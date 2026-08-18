"""
pipeline/sqi.py
───────────────
compute_sqi() — copied verbatim from the training notebook.

Checks:
  1. Flat / dead signal       std < 1e-3
  2. ADC clipping             > 3% at min or max
  3. BPM outside [40, 180]
  4. Inter-peak CV > 0.40     (irregular rhythm)

Returns (score: float [0–1], is_acceptable: bool)
"""

import numpy as np
from scipy.signal import find_peaks


def compute_sqi(segment: np.ndarray, fs: float = 100.0) -> tuple[float, bool]:
    seg = np.asarray(segment, dtype=np.float32).copy()

    # 1. Flat signal
    if np.std(seg) < 1e-3:
        return 0.0, False

    # 2. Clipping
    mn, mx = seg.min(), seg.max()
    if np.mean(seg == mn) > 0.03 or np.mean(seg == mx) > 0.03:
        return 0.0, False

    # 3. Peak-based rhythmicity
    seg_n    = (seg - mn) / (mx - mn + 1e-8)
    min_dist = int(fs * 0.30)

    peaks, _ = find_peaks(seg_n, distance=min_dist, prominence=0.20, height=0.20)

    n_peaks    = len(peaks)
    bpm        = n_peaks * (60.0 / (len(seg) / fs))

    if bpm < 40 or bpm > 180:
        return 0.0, False

    if n_peaks >= 3:
        ipi = np.diff(peaks).astype(float)
        cv  = ipi.std() / (ipi.mean() + 1e-8)
        if cv > 0.40:
            return float(np.clip(1.0 - cv, 0.0, 1.0)), False
        bpm_score = float(np.clip(1.0 - abs(bpm - 75) / 100.0, 0.0, 1.0))
        cv_score  = float(np.clip(1.0 - cv, 0.0, 1.0))
        return 0.5 * bpm_score + 0.5 * cv_score, True

    return 0.3, True
