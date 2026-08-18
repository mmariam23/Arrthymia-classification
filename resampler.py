"""
pipeline/resampler.py
─────────────────────
Resample a jittered PPG time-series to a uniform 100 Hz grid.
Matches _uniform_resample() from the training notebook exactly.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def uniform_resample(
    time_ms:        np.ndarray,
    ppg:            np.ndarray,
    target_fs:      float = 100.0,
    gap_ms:         float = 200.0,
    min_duration_s: float = 10.0,
) -> np.ndarray:
    """
    Resample a (possibly jittered / irregular) PPG signal to `target_fs` Hz.

    Parameters
    ----------
    time_ms        : timestamps in milliseconds
    ppg            : raw ADC values, same length as time_ms
    target_fs      : desired output sample rate (Hz) — default 100
    gap_ms         : gaps larger than this are treated as recording breaks
    min_duration_s : chunks shorter than this are discarded

    Returns
    -------
    1-D float32 array at target_fs Hz
    """
    t = np.asarray(time_ms, dtype=np.float64) / 1000.0   # ms → seconds
    x = np.asarray(ppg,     dtype=np.float32)

    # Sort and deduplicate timestamps
    order = np.argsort(t)
    t, x  = t[order], x[order]
    df    = (pd.DataFrame({"t": t, "x": x})
               .groupby("t", as_index=False)
               .mean())
    t = df["t"].to_numpy(np.float64)
    x = df["x"].to_numpy(np.float32)

    # Split on large gaps (device disconnects / pauses)
    dt_ms  = np.diff(t) * 1000.0
    cuts   = np.where(dt_ms > gap_ms)[0]
    starts = np.r_[0,        cuts + 1]
    ends   = np.r_[cuts + 1, len(t)]

    chunks = []
    for s, e in zip(starts, ends):
        tc, xc = t[s:e], x[s:e]
        if e - s < 5 or (tc[-1] - tc[0]) < min_duration_s:
            continue
        t_uni = np.arange(tc[0], tc[-1], 1.0 / target_fs)
        x_uni = interp1d(
            tc, xc, kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )(t_uni).astype(np.float32)
        chunks.append(x_uni)

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
