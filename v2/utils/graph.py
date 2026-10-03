"""V2 graph helpers — wraps the original `utils.graph` with a connectivity fix.

Adds `ensure_connected(W, X, fallback_weight)` which checks if the graph has
multiple connected components and, if so, splices in weak fallback edges
between the closest pair of nodes in each component pair. This is needed for
Level 2 (Fix 5) where the best (k, σ) pair often produces a tiny disconnected
sub-component (e.g. 2787 + 6 nodes), which makes label propagation undefined
for the disconnected nodes.

Also re-exports the original utility functions so v2 notebooks can import
everything from one place.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

# Re-export the original functions so v2 notebooks have one import surface.
from utils.graph import (  # noqa: F401
    build_knn_graph,
    compute_laplacian,
    spectral_gap_analysis,
    spectral_gap_from_W,
    diffusion_map_embedding,
    label_propagation,
    laplacian_regularization,
    spectral_clustering,
    _build_label_matrix,  # noqa: F401  (used by tests / notebooks)
)


def n_connected_components(W: sp.spmatrix) -> int:
    """Number of connected components in the (symmetric) similarity graph W."""
    W = sp.csr_matrix(W)
    n_comp, _ = connected_components(W, directed=False, return_labels=True)
    return int(n_comp)


def ensure_connected(
    W: sp.spmatrix,
    X: np.ndarray,
    fallback_weight: float = 1e-6,
) -> sp.csr_matrix:
    """If W has multiple components, add weak edges between them.

    For every pair of components (i, j) we find the closest pair of nodes
    (one in each) by Euclidean distance in `X` and connect them with a
    symmetric edge of weight `fallback_weight`. The original edges are
    preserved.

    The fallback weight is intentionally small so the new edges do not
    perturb tightly connected components, but they are large enough that
    label propagation can flow across them.

    Returns
    -------
    W_new : (N, N) sparse symmetric CSR matrix.
    """
    W = sp.csr_matrix(W).astype(np.float64)
    n_comp, comp_lab = connected_components(W, directed=False, return_labels=True)
    if n_comp <= 1:
        return W

    X = np.ascontiguousarray(X, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    # For efficiency, grow a spanning tree across components: pick component 0
    # as the "anchor" pool and connect each other component to its nearest
    # neighbor in the anchor pool. Then merge.
    in_anchor = comp_lab == 0
    anchor_idx = np.where(in_anchor)[0]
    anchor_X = X[anchor_idx]
    nbrs = NearestNeighbors(n_neighbors=1).fit(anchor_X)

    for c in range(1, n_comp):
        other_idx = np.where(comp_lab == c)[0]
        if other_idx.size == 0:
            continue
        other_X = X[other_idx]
        dists, locs = nbrs.kneighbors(other_X)
        j = int(np.argmin(dists[:, 0]))
        u = int(other_idx[j])
        v = int(anchor_idx[int(locs[j, 0])])
        rows.append(u); cols.append(v); data.append(float(fallback_weight))
        rows.append(v); cols.append(u); data.append(float(fallback_weight))

    if not rows:
        return W
    W_extra = sp.coo_matrix(
        (data, (rows, cols)), shape=W.shape, dtype=np.float64
    ).tocsr()
    W_new = (W + W_extra).tocsr()
    W_new.setdiag(0.0)
    W_new.eliminate_zeros()
    return W_new
