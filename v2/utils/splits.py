"""Clip-level labeled/unlabeled splits for graph SSL (V2 fix 1).

V1 sampled labels at the window level. A 10-second clip contributing 5
windows could end up with 3 labeled + 2 unlabeled, and the graph would
trivially propagate the label to the near-identical sister windows. This
module enforces splits at the *clip* level: every window from a given
clip is either entirely labeled or entirely unlabeled.
"""

from __future__ import annotations

import numpy as np


def majority_clip_labels(
    clip_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_clips, clip_label) where clip_label is the majority label.

    For Level 1 / Level 2 every window from one clip already shares its label
    (a clip belongs to one species), so the majority is just the first
    window's label. We compute the majority for safety in case that ever
    changes.
    """
    clip_ids = np.asarray(clip_ids)
    labels = np.asarray(labels)
    unique_clips = np.unique(clip_ids)
    clip_label = np.empty(unique_clips.shape[0], dtype=labels.dtype)
    for i, c in enumerate(unique_clips):
        sel = labels[clip_ids == c]
        vals, counts = np.unique(sel, return_counts=True)
        clip_label[i] = vals[int(np.argmax(counts))]
    return unique_clips, clip_label


def stratified_clip_mask(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `frac` of clips per class, then label ALL their windows.

    Returns
    -------
    mask : (N,) bool array — True for windows whose clip was selected.
    """
    clip_ids = np.asarray(clip_ids)
    labels = np.asarray(labels)
    unique_clips, clip_label = majority_clip_labels(clip_ids, labels)

    chosen_clips: list = []
    for c in np.unique(clip_label):
        idx = np.where(clip_label == c)[0]
        n_pick = max(1, int(round(frac * idx.size)))
        n_pick = min(n_pick, idx.size)
        pick = rng.choice(idx, size=n_pick, replace=False)
        chosen_clips.extend(unique_clips[pick].tolist())

    chosen_set = set(int(c) for c in chosen_clips)
    return np.isin(clip_ids, list(chosen_set))
