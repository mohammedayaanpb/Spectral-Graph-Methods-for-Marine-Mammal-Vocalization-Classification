"""V2 feature extractors. Replaces 12-dim chroma at Level 2 with 12 dims of
MFCC variability (6 MFCC std + 6 delta-MFCC std).

Rationale (REPORT_v2 fix 6): chroma features were designed for music — they
fold the spectrum into 12 pitch classes assuming a periodic harmonic
structure that dolphin clicks/whistles don't follow. MFCC and delta-MFCC
*standard deviations* across frames are far more meaningful for this domain:
clicks are impulsive (high MFCC std, high dMFCC std), whistles are sustained
(low std), so the feature directly tracks the impulsive-vs-tonal axis.
"""

from __future__ import annotations

import numpy as np
import librosa

# Re-export everything else from the original features module.
from utils.features import (  # noqa: F401
    TARGET_SR,
    _safe,
    _stats_or_zero,
    _is_silent,
    compute_mel_spectrogram,
    _common_spectral,
    extract_features_level1,
    extract_features_level3,
)


def _std_or_zero(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Std over axis, zero if input empty/silent."""
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    return _safe(x.std(axis=axis)).astype(np.float32)


def extract_features_level2_v2(window: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """51-dim feature vector with the 12 chroma dims swapped out.

    Layout (51 total):
      [0:13]  13 MFCC means
      [13:26] 13 dMFCC means
      [26:32] 6 spectral stats (centroid, bandwidth, rolloff, flatness, ZCR, RMS)
      [32:39] 7-band spectral contrast means
      [39:45] 6 MFCC stds (first 6 coefficients) — NEW
      [45:51] 6 dMFCC stds (first 6 coefficients) — NEW
    """
    base = extract_features_level1(window, sr=sr)  # 32 dims
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    if window.size == 0 or _is_silent(window):
        return np.concatenate([base, np.zeros(7 + 6 + 6, dtype=np.float32)])

    contrast = librosa.feature.spectral_contrast(y=window, sr=sr, n_bands=6)

    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    dmfcc = librosa.feature.delta(mfcc)

    mfcc_std = _std_or_zero(mfcc[:6, :])
    dmfcc_std = _std_or_zero(dmfcc[:6, :])

    feat = np.concatenate([
        base,
        _stats_or_zero(contrast),
        mfcc_std,
        dmfcc_std,
    ])
    return _safe(feat).astype(np.float32)
