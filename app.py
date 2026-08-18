"""
app.py
======
FastAPI — PPG Arrhythmia Detection Server
Receives raw PPG data from a mobile app as JSON.

POST /predict  →  { "ppg": [...], "time_ms": [...], "fs": 100 }
GET  /health   →  server + model status
"""

import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.hr         import classify_heart_rate, compute_heart_rate
from pipeline.model      import VGGModel
from pipeline.preprocess import bandpass_filter, minmax_normalize, moving_average
from pipeline.resampler  import uniform_resample
from pipeline.sqi        import compute_sqi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_FS = 100
SEG_LEN    = 1000   # 10 s × 100 Hz

app = FastAPI(
    title="PPG Arrhythmia Detection",
    description="Send raw PPG numbers → get BPM, diagnosis, and arrhythmia type.\n\n**Classes:** SR · PVC · PAC · VT · SVT · AF",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/vgg_arrhythmia.pt")
_model     = VGGModel(MODEL_PATH, num_classes=6)


# ── Schemas ───────────────────────────────────────────────────────

class PPGRequest(BaseModel):
    ppg: list[float] = Field(..., description="Raw PPG ADC values from sensor")
    time_ms: list[float] | None = Field(default=None, description="Timestamps in ms (optional)")
    fs: float | None = Field(default=None, description="Sample rate Hz (optional, default=100)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ppg": [91706, 91712, 91724, 91760, 91754, 91804, 91870],
                "time_ms": None,
                "fs": None
            }
        }
    }


class SegmentResult(BaseModel):
    index:         int
    accepted:      bool
    sqi_score:     float
    bpm:           float | None
    hr_status:     str
    arrhythmia:    str | None
    reject_reason: str | None


class PredictResponse(BaseModel):
    # ── Overall diagnosis ──────────────────────────────────────────
    overall_diagnosis:   str          # final verdict across all segments
    overall_bpm:         float | None # average BPM from accepted segments
    arrhythmia_type:     str | None   # VGG result (if arrhythmic)
    # ── Detail ────────────────────────────────────────────────────
    fs_used:             float
    total_segments:      int
    accepted_segments:   int
    segments:            list[SegmentResult]


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "model_loaded": _model.is_loaded}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(body: PPGRequest):
    """
    Send raw PPG samples from the mobile app.

    - **ppg** is the only required field.
    - **time_ms** optional — send when sensor timing is jittered.
    - **fs** optional — sample rate in Hz. If not sent, **100 Hz is assumed**.
    """
    ppg = np.asarray(body.ppg, dtype=np.float32)

    if len(ppg) == 0:
        raise HTTPException(400, "ppg array is empty")

    # ── Resample to 100 Hz ────────────────────────────────────────
    if body.time_ms is not None:
        if len(body.time_ms) != len(ppg):
            raise HTTPException(
                400,
                f"ppg length ({len(ppg)}) != time_ms length ({len(body.time_ms)})"
            )
        time_ms = np.asarray(body.time_ms, dtype=np.float64)
        uniform = uniform_resample(time_ms, ppg, target_fs=DEFAULT_FS)
        fs_used = float(DEFAULT_FS)
    else:
        fs_src  = float(body.fs) if (body.fs is not None and body.fs > 0) else DEFAULT_FS
        fs_used = float(DEFAULT_FS)
        if abs(fs_src - DEFAULT_FS) < 0.5:
            uniform = ppg
        else:
            duration_ms = len(ppg) / fs_src * 1000.0
            time_ms     = np.linspace(0.0, duration_ms, len(ppg), dtype=np.float64)
            uniform     = uniform_resample(time_ms, ppg, target_fs=DEFAULT_FS)

    # ── Segment ───────────────────────────────────────────────────
    n_segs = len(uniform) // SEG_LEN
    if n_segs == 0:
        return PredictResponse(
            overall_diagnosis  = "Uncertain — signal too short (need ≥10 seconds)",
            overall_bpm        = None,
            arrhythmia_type    = None,
            fs_used            = fs_used,
            total_segments     = 0,
            accepted_segments  = 0,
            segments           = [],
        )

    raw_segs = uniform[: n_segs * SEG_LEN].reshape(n_segs, SEG_LEN)

    # ── Run per-segment pipeline ──────────────────────────────────
    segments, overall = _process_segments(raw_segs)

    return PredictResponse(
        overall_diagnosis  = overall["diagnosis"],
        overall_bpm        = overall["bpm"],
        arrhythmia_type    = overall["arrhythmia_type"],
        fs_used            = fs_used,
        total_segments     = len(segments),
        accepted_segments  = sum(1 for s in segments if s["accepted"]),
        segments           = segments,
    )


# ── Core pipeline ─────────────────────────────────────────────────

def _process_segments(raw_segs: np.ndarray) -> tuple[list, dict]:
    results  = []
    all_bpms = []
    arrhythmia_votes = []   # collect VGG labels across segments

    for idx, seg in enumerate(raw_segs):
        out = {
            "index":         idx,
            "accepted":      False,
            "sqi_score":     0.0,
            "bpm":           None,
            "hr_status":     "Uncertain",
            "arrhythmia":    None,
            "reject_reason": None,
        }

        # 1. Saturation / clipping check
        if not _saturation_ok(seg):
            out["reject_reason"] = "ADC saturation or clipping detected"
            results.append(out)
            continue

        # 2. Bandpass 0.5–10 Hz
        seg_f = bandpass_filter(seg, fs=DEFAULT_FS)

        # 3. Moving average w=5
        seg_f = moving_average(seg_f, window=5)

        # 4. SQI
        sqi_score, sqi_ok = compute_sqi(seg_f, fs=DEFAULT_FS)
        out["sqi_score"] = round(float(sqi_score), 4)

        if not sqi_ok:
            out["reject_reason"] = "Failed SQI check"
            results.append(out)
            continue

        # 5. Min-Max normalise [0,1]
        seg_norm     = minmax_normalize(seg_f)
        out["accepted"] = True

        # 6. Heart rate
        bpm       = compute_heart_rate(seg_norm, fs=DEFAULT_FS)
        hr_status = classify_heart_rate(bpm)

        if bpm > 0:
            out["bpm"]       = round(float(bpm), 1)
            out["hr_status"] = hr_status
            all_bpms.append(bpm)
        else:
            out["bpm"]       = None
            out["hr_status"] = "Uncertain (no detectable pulse)"

        # 7. VGG model — only for arrhythmic segments
        if bpm > 0 and "Arrhythmia" in hr_status:
            label, conf      = _model.predict_single(seg_norm)
            out["arrhythmia"] = f"{label} (confidence: {conf:.1%})"
            arrhythmia_votes.append(label)

        results.append(out)

    # ── Overall diagnosis ─────────────────────────────────────────
    overall = _overall_diagnosis(all_bpms, arrhythmia_votes, len(results))
    return results, overall


def _overall_diagnosis(
    all_bpms: list[float],
    arrhythmia_votes: list[str],
    total: int,
) -> dict:
    """
    Summarise across all accepted segments into a single diagnosis.
    """
    if not all_bpms:
        return {
            "diagnosis":      "Uncertain — no acceptable signal segments found",
            "bpm":            None,
            "arrhythmia_type": None,
        }

    avg_bpm   = round(float(np.mean(all_bpms)), 1)
    hr_label  = classify_heart_rate(avg_bpm)

    if "Arrhythmia" not in hr_label:
        return {
            "diagnosis":       f"Normal — Average BPM: {avg_bpm}",
            "bpm":             avg_bpm,
            "arrhythmia_type": None,
        }

    # Pick the most common VGG label across arrhythmic segments
    if arrhythmia_votes:
        from collections import Counter
        top_label = Counter(arrhythmia_votes).most_common(1)[0][0]
        arrhythmia_type = top_label
    else:
        arrhythmia_type = "Unknown (model unavailable)"

    subtype = hr_label.replace("Arrhythmia", "").strip("() ")   # e.g. Bradycardia / Tachycardia

    return {
        "diagnosis":       f"Arrhythmia Detected — {subtype} | BPM: {avg_bpm} | Type: {arrhythmia_type}",
        "bpm":             avg_bpm,
        "arrhythmia_type": arrhythmia_type,
    }


def _saturation_ok(seg: np.ndarray, reject_pct: float = 0.05) -> bool:
    mn, mx = seg.min(), seg.max()
    if (mx - mn) < 1e-6:
        return False
    if np.mean(seg == mn) > reject_pct or np.mean(seg == mx) > reject_pct:
        return False
    q25, q75 = np.percentile(seg, 25), np.percentile(seg, 75)
    iqr      = q75 - q25 + 1e-8
    if np.any(np.abs(seg - np.median(seg)) > 6 * iqr):
        return False
    return True