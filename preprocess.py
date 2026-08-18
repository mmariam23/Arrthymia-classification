"""
pipeline/preprocess.py
──────────────────────
Matches the training notebook exactly:
  bandpass_filter  : 4th-order zero-phase Butterworth, 0.5–10 Hz
  moving_average   : np.convolve same-mode, window=5
  minmax_normalize : scale to [0, 1]
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs:     float = 100.0,
    lo:     float = 0.5,
    hi:     float = 10.0,
    order:  int   = 4,
) -> np.ndarray:
    nyq  = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(
        np.asarray(signal, dtype=np.float32), kernel, mode="same"
    ).astype(np.float32)


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    mn, mx = np.min(signal), np.max(signal)
    rng    = mx - mn
    if rng < 1e-8:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - mn) / rng).astype(np.float32)