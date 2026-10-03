"""Protect the clip-level split that prevents sister-window label leakage."""

import numpy as np
from numpy.testing import assert_array_equal

from v2.utils.splits import majority_clip_labels, stratified_clip_mask


def test_majority_labels_follow_clip_ids():
    clips, labels = majority_clip_labels(
        np.array([7, 2, 7, 2, 7]), np.array([4, 9, 5, 9, 4])
    )
    assert_array_equal(clips, [2, 7])
    assert_array_equal(labels, [9, 4])


def test_stratified_selection_keeps_each_clip_together_and_is_reproducible():
    clips = np.repeat(np.arange(20), 3)
    labels = np.repeat([2, 8], 30)
    mask = stratified_clip_mask(clips, labels, 0.3, np.random.default_rng(17))
    repeated = stratified_clip_mask(clips, labels, 0.3, np.random.default_rng(17))
    assert mask.dtype == np.bool_
    assert_array_equal(mask, repeated)
    for clip in np.unique(clips):
        assert np.unique(mask[clips == clip]).size == 1
    for label in [2, 8]:
        assert np.unique(clips[mask & (labels == label)]).size == 3


def test_small_label_fraction_keeps_a_seed_for_each_class():
    clips = np.repeat(np.arange(4), 2)
    labels = np.repeat([2, 2, 8, 8], 2)
    mask = stratified_clip_mask(clips, labels, 0.01, np.random.default_rng(0))
    assert_array_equal(np.unique(labels[mask]), [2, 8])
    assert np.unique(clips[mask]).size == 2
