"""Small numerical examples for the graph methods, independent of the dataset."""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

from utils.graph import (
    build_knn_graph,
    compute_laplacian,
    diffusion_map_embedding,
    label_propagation,
    laplacian_regularization,
    spectral_clustering,
    spectral_gap_analysis,
    spectral_gap_from_W,
)
from v2.utils.graph import ensure_connected, n_connected_components


@pytest.fixture
def path_graph():
    return sp.diags([np.ones(5), np.ones(5)], [-1, 1], shape=(6, 6), format="csr")


def test_knn_gaussian_weights_and_symmetric_union():
    graph = build_knn_graph(np.array([[0.0], [1.0], [3.0]]), k=1, sigma=2.0)
    near, far = np.exp(-1.0 / 8.0), np.exp(-4.0 / 8.0)
    assert sp.isspmatrix_csr(graph)
    assert_allclose(graph.toarray(), [[0, near, 0], [near, 0, far], [0, far, 0]])


@pytest.mark.parametrize("sigma", ["median", "self_tuning", 0.5])
def test_knn_bandwidths_produce_valid_affinities(sigma):
    features = np.array([[0, 0], [1, 0], [1, 2], [3, 1]], dtype=float)
    graph = build_knn_graph(features, k=10, sigma=sigma)
    assert_allclose(graph.toarray(), graph.toarray().T)
    assert_array_equal(graph.diagonal(), np.zeros(4))
    assert np.isfinite(graph.data).all()
    assert ((graph.data > 0) & (graph.data <= 1)).all()
    assert graph.nnz == 12  # k is capped at the number of other samples.


@pytest.mark.parametrize("size", [0, 1])
def test_knn_handles_trivial_inputs(size):
    graph = build_knn_graph(np.zeros((size, 2)))
    assert graph.shape == (size, size)
    assert graph.nnz == 0


def test_laplacians_annihilate_constants(path_graph):
    unnormalized = compute_laplacian(path_graph, normalized=False)
    random_walk = compute_laplacian(path_graph)
    assert_allclose(unnormalized @ np.ones(6), 0)
    assert_allclose(random_walk @ np.ones(6), 0)
    assert_allclose(unnormalized.toarray(), unnormalized.toarray().T)
    assert_allclose(random_walk.diagonal(), 1)
    assert_allclose(random_walk.toarray()[1, [0, 2]], [-0.5, -0.5])


def test_spectral_gap_matches_known_path_spectrum(path_graph):
    expected = 1 - np.cos(np.pi * np.arange(3) / 5)
    assert_allclose(spectral_gap_from_W(path_graph, 3), expected, atol=1e-12)
    assert_allclose(
        spectral_gap_analysis(compute_laplacian(path_graph), 3), expected, atol=1e-12
    )


def test_diffusion_coordinates_are_nontrivial_eigenfunctions(path_graph):
    coordinates = diffusion_map_embedding(path_graph, n_components=2, t=0)
    degrees = np.asarray(path_graph.sum(axis=1)).ravel()
    transition = sp.diags(1 / degrees) @ path_graph
    eigenvalues = np.cos(np.pi * np.arange(1, 3) / 5)
    assert coordinates.shape == (6, 2)
    assert_allclose(transition @ coordinates, coordinates * eigenvalues, atol=1e-12)
    assert_allclose(coordinates.T @ (degrees[:, None] * coordinates), np.eye(2), atol=1e-12)
    assert_allclose(degrees @ coordinates, 0, atol=1e-12)


@pytest.mark.parametrize("method", [label_propagation, laplacian_regularization])
def test_ssl_recovers_path_labels_and_preserves_seeds(path_graph, method):
    labels = np.array([3, -1, -1, -1, -1, 9])
    mask = np.array([True, False, False, False, False, True])
    predictions = method(path_graph, labels, mask)
    assert_array_equal(predictions, [3, 3, 3, 9, 9, 9])
    assert_array_equal(labels, [3, -1, -1, -1, -1, 9])


def test_spectral_clustering_separates_disconnected_groups():
    triangle = np.ones((3, 3)) - np.eye(3)
    graph = sp.block_diag([triangle, triangle], format="csr")
    labels = spectral_clustering(graph, k=2, random_state=0)
    assert np.unique(labels[:3]).size == 1
    assert np.unique(labels[3:]).size == 1
    assert labels[0] != labels[3]


def test_connectivity_repair_preserves_existing_edges():
    pair = np.array([[0.0, 1.0], [1.0, 0.0]])
    original = sp.block_diag([pair, pair, pair], format="csr")
    original.eliminate_zeros()
    features = np.array([[0], [1], [10], [11], [20], [21]], dtype=float)
    repaired = ensure_connected(original, features, fallback_weight=1e-4)
    assert n_connected_components(original) == 3
    assert n_connected_components(repaired) == 1
    assert_allclose(repaired.toarray(), repaired.toarray().T)
    assert_array_equal(repaired.diagonal(), np.zeros(6))
    rows, columns = original.nonzero()
    assert_allclose(np.asarray(repaired[rows, columns]).ravel(), 1)
    added = repaired - original
    assert added.nnz == 4  # Two undirected bridges connect three components.
    assert_allclose(added.data, 1e-4)
    assert original.nnz == 6


def test_connected_graph_does_not_change(path_graph):
    repaired = ensure_connected(path_graph, np.arange(6).reshape(-1, 1))
    assert_allclose(repaired.toarray(), path_graph.toarray())
