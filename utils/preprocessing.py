"""Audio preprocessing utilities for WMMD pipeline.

Resampling to 22050 Hz, fixed 2-second windowing with symmetric zero-pad
(short clips) or non-overlapping segmentation (long clips), and a 4th-order
Butterworth bandpass between 100 Hz and 10 kHz.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


TARGET_SR = 22050
WINDOW_SEC = 2.0


def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample mono audio to target_sr. Returns float32 1-D array."""
    audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)


def segment_audio(audio_array: np.ndarray, sr: int, window_sec: float = WINDOW_SEC) -> list[np.ndarray]:
    """Split a 1-D audio array into fixed-length windows.

    - Short clips (< window): symmetric zero-pad up to one window.
    - Exactly one window: returned as-is.
    - Long clips: non-overlapping segmentation; the trailing tail is dropped if
      it is shorter than half a window, otherwise it's symmetrically padded.
    """
    audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    win = int(round(sr * window_sec))
    if win <= 0:
        raise ValueError(f"window length must be positive, got {win}")
    n = audio.size

    if n == 0:
        return [np.zeros(win, dtype=np.float32)]

    if n <= win:
        pad_total = win - n
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        return [np.pad(audio, (pad_l, pad_r), mode="constant").astype(np.float32)]

    n_full = n // win
    windows = [audio[i * win:(i + 1) * win].copy() for i in range(n_full)]
    tail = audio[n_full * win:]
    if tail.size >= win // 2:
        pad_total = win - tail.size
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        windows.append(np.pad(tail, (pad_l, pad_r), mode="constant").astype(np.float32))
    return windows


def bandpass_filter(
    audio_array: np.ndarray,
    sr: int,
    lowcut: float = 100.0,
    highcut: float = 10000.0,
    order: int = 4,
) -> np.ndarray:
    """4th-order Butterworth bandpass via SOS + zero-phase filtfilt.

    If sr is too low for the requested band, the band is clamped just below
    the Nyquist frequency. Returns a float32 array of the same length.
    """
    audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    nyq = 0.5 * sr
    low = max(1.0, lowcut) / nyq
    high = min(highcut, nyq * 0.99) / nyq
    if high <= low:
        return audio
    sos = butter(order, [low, high], btype="band", output="sos")
    padlen = min(audio.size - 1, 3 * (2 * order))
    if padlen < 1:
        return audio
    return sosfiltfilt(sos, audio, padlen=padlen).astype(np.float32)


def preprocess_clip(audio_array: np.ndarray, orig_sr: int) -> list[np.ndarray]:
    """Full pipeline: resample to 22050 Hz, segment into 2-sec windows, bandpass.

    Each returned window is 44100 samples long (TARGET_SR * WINDOW_SEC).
    Returns an empty list only if input is empty after coercion.
    """
    audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return []
    resampled = resample_audio(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
    windows = segment_audio(resampled, sr=TARGET_SR, window_sec=WINDOW_SEC)
    out = []
    for w in windows:
        if not np.any(w):
            out.append(w.astype(np.float32))
        else:
            out.append(bandpass_filter(w, sr=TARGET_SR).astype(np.float32))
    return out
