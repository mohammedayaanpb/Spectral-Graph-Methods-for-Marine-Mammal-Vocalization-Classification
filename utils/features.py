"""Feature extractors for the three classification levels.

Level 1 — 32 dims (MFCC + Delta + spectral stats)
Level 2 — 51 dims (Level 1 + spectral contrast + chroma)
Level 3 — 46 dims (temporal/pulse-aware feature set)

Also: log-Mel spectrogram for the deep-learning baselines.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.stats import kurtosis


TARGET_SR = 22050


def _safe(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with finite values."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _stats_or_zero(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Mean over an axis, but zero if input is empty/silent."""
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    return _safe(x.mean(axis=axis)).astype(np.float32)


def _is_silent(window: np.ndarray, eps: float = 1e-7) -> bool:
    return float(np.max(np.abs(window))) < eps


def compute_mel_spectrogram(
    window: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """log-Mel spectrogram, shape (n_mels, T). float32."""
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    S = librosa.feature.melspectrogram(
        y=window, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    S = np.log(S + 1e-9)
    return _safe(S).astype(np.float32)


def _common_spectral(window: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Frame-level basic spectral stats used by multiple levels.

    Returns dict of frame-level arrays so callers can take mean/std as needed.
    """
    return {
        "centroid": librosa.feature.spectral_centroid(y=window, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=window, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=window, sr=sr),
        "flatness": librosa.feature.spectral_flatness(y=window),
        "zcr": librosa.feature.zero_crossing_rate(y=window),
        "rms": librosa.feature.rms(y=window),
    }


def extract_features_level1(window: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """32-dim feature vector: MFCC(13) + dMFCC(13) + 6 spectral stats."""
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    if window.size == 0 or _is_silent(window):
        return np.zeros(32, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    dmfcc = librosa.feature.delta(mfcc)
    spec = _common_spectral(window, sr)
    feat = np.concatenate([
        _stats_or_zero(mfcc),
        _stats_or_zero(dmfcc),
        _stats_or_zero(spec["centroid"]),
        _stats_or_zero(spec["bandwidth"]),
        _stats_or_zero(spec["rolloff"]),
        _stats_or_zero(spec["flatness"]),
        _stats_or_zero(spec["zcr"]),
        _stats_or_zero(spec["rms"]),
    ])
    return _safe(feat).astype(np.float32)


def extract_features_level2(window: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """51-dim: Level 1 features + 7-band spectral contrast + 12-bin chroma."""
    base = extract_features_level1(window, sr=sr)
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    if window.size == 0 or _is_silent(window):
        return np.concatenate([base, np.zeros(7 + 12, dtype=np.float32)])
    contrast = librosa.feature.spectral_contrast(y=window, sr=sr, n_bands=6)
    chroma = librosa.feature.chroma_stft(y=window, sr=sr)
    feat = np.concatenate([base, _stats_or_zero(contrast), _stats_or_zero(chroma)])
    return _safe(feat).astype(np.float32)


def _temporal_entropy(window: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    rms = librosa.feature.rms(y=window, frame_length=frame_length, hop_length=hop_length).reshape(-1)
    energy = rms ** 2
    s = energy.sum()
    if s <= 0 or not np.isfinite(s):
        return 0.0
    p = energy / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _autocorr_peak(window: np.ndarray, sr: int) -> tuple[float, float]:
    """Lag (in seconds) and height of first significant autocorrelation peak.

    Searches lags between ~1 ms and 100 ms, which spans plausible inter-click
    intervals for dolphin pulse trains.
    """
    if window.size < 4:
        return 0.0, 0.0
    x = window - window.mean()
    norm = np.dot(x, x)
    if norm <= 0 or not np.isfinite(norm):
        return 0.0, 0.0
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n] / norm
    lo = max(1, int(0.001 * sr))
    hi = min(n - 1, int(0.1 * sr))
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi]
    if seg.size == 0:
        return 0.0, 0.0
    idx = int(np.argmax(seg))
    lag_s = (lo + idx) / float(sr)
    height = float(seg[idx])
    if not np.isfinite(lag_s):
        lag_s = 0.0
    if not np.isfinite(height):
        height = 0.0
    return lag_s, height


def extract_features_level3(window: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """46-dim feature vector tuned for vocalization-type clustering.

    MFCC(13) + dMFCC(13) + 4 spectral stats (centroid, bw, rolloff, flatness mean)
    + 7-band spectral contrast + ZCR(mean,std) + RMS(mean,std)
    + envelope kurtosis + temporal entropy + autocorr lag + autocorr height
    + flatness_std (std of frame-level flatness, the post-hoc validation feature
      mentioned in docs/METHODS.md).
    Total: 13 + 13 + 4 + 7 + 2 + 2 + 1 + 1 + 1 + 1 + 1 = 46.
    """
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    if window.size == 0 or _is_silent(window):
        return np.zeros(46, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    dmfcc = librosa.feature.delta(mfcc)
    spec = _common_spectral(window, sr)
    contrast = librosa.feature.spectral_contrast(y=window, sr=sr, n_bands=6)

    zcr = spec["zcr"].reshape(-1)
    rms = spec["rms"].reshape(-1)
    flat = spec["flatness"].reshape(-1)
    flat_std = float(flat.std()) if flat.size else 0.0
    if not np.isfinite(flat_std):
        flat_std = 0.0

    envelope = np.abs(window)
    env_kurt = float(kurtosis(envelope, fisher=True, bias=False)) if envelope.size > 3 else 0.0
    if not np.isfinite(env_kurt):
        env_kurt = 0.0

    t_entropy = _temporal_entropy(window)
    ac_lag, ac_h = _autocorr_peak(window, sr)

    feat = np.concatenate([
        _stats_or_zero(mfcc),
        _stats_or_zero(dmfcc),
        _stats_or_zero(spec["centroid"]),
        _stats_or_zero(spec["bandwidth"]),
        _stats_or_zero(spec["rolloff"]),
        _stats_or_zero(spec["flatness"]),
        _stats_or_zero(contrast),
        np.array([zcr.mean() if zcr.size else 0.0, zcr.std() if zcr.size else 0.0], dtype=np.float32),
        np.array([rms.mean() if rms.size else 0.0, rms.std() if rms.size else 0.0], dtype=np.float32),
        np.array([env_kurt], dtype=np.float32),
        np.array([t_entropy], dtype=np.float32),
        np.array([ac_lag], dtype=np.float32),
        np.array([ac_h], dtype=np.float32),
        np.array([flat_std], dtype=np.float32),
    ])
    return _safe(feat).astype(np.float32)
