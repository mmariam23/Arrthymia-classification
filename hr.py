"""
pipeline/hr.py
──────────────
Heart-rate estimation and clinical classification.

hr_status values
----------------
  "Uncertain (no detectable pulse)"   bpm == 0
  "Arrhythmia (Bradycardia)"          bpm < 60
  "Normal"                            60 ≤ bpm < 100
  "Arrhythmia (Tachycardia)"          bpm ≥ 100
"""

import numpy as np
from scipy.signal import find_peaks


def compute_heart_rate(
    seg_norm: np.ndarray,
    fs:       float = 100.0,
) -> float:
    """Return BPM from a [0,1]-normalised segment. Returns 0.0 if undetectable."""
    peaks, _ = find_peaks(
        seg_norm,
        distance=int(fs * 0.30),
        prominence=0.20,
        height=0.20,
    )
    if len(peaks) < 2:
        return 0.0
    mean_rr = np.mean(np.diff(peaks) / fs)
    return float(60.0 / mean_rr) if mean_rr > 1e-6 else 0.0


def classify_heart_rate(bpm: float) -> str:
    if bpm <= 0:
        return "Uncertain"
    elif bpm < 60:
        return "Arrhythmia (Bradycardia)"
    elif bpm < 100:
        return "Normal"
    else:
        return "Arrhythmia (Tachycardia)"