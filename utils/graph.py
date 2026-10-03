"""Graph construction and spectral methods.

Implements:
- k-NN graph with Gaussian weights (median, self-tuning, or scalar sigma)
- Normalized random-walk Laplacian
- Diffusion map embedding
- Label propagation (Calder & Olver, Section 9.9.2)
- Laplacian regularization (Section 9.9.1)
- Spectral clustering (normalized Laplacian eigenvectors + k-means)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
    sigma: str | float = "median",
) -> sp.csr_matrix:
    """Build a symmetric k-NN graph with Gaussian-kernel weights.

    Parameters
    ----------
    X : (N, d) feature matrix.
    k : neighbors per point (excluding self).
    sigma :
        - 'median'      : single global sigma = median of all NN distances
        - 'self_tuning' : Zelnik-Manor & Perona local sigma_i, weight uses sigma_i*sigma_j
        - float         : explicit scalar sigma

    Returns
    -------
    W : (N, N) sparse symmetric CSR matrix, zero diagonal.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    N = X.shape[0]
    if N <= 1:
        return sp.csr_matrix((N, N), dtype=np.float32)
    k = min(int(k), N - 1)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist = dist[:, 1:]
    idx = idx[:, 1:]

    rows = np.repeat(np.arange(N), k)
    cols = idx.reshape(-1)
    d_flat = dist.reshape(-1).astype(np.float64)

    if sigma == "median":
        med = float(np.median(dist))
        sig2 = max(med ** 2, 1e-12)
        w = np.exp(-(d_flat ** 2) / (2.0 * sig2))
    elif sigma == "self_tuning":
        sigma_i = dist[:, -1].astype(np.float64)
        sigma_i = np.maximum(sigma_i, 1e-12)
        sig_row = np.repeat(sigma_i, k)
        sig_col = sigma_i[cols]
        w = np.exp(-(d_flat ** 2) / (sig_row * sig_col + 1e-12))
    else:
        s = float(sigma)
        sig2 = max(s ** 2, 1e-12)
        w = np.exp(-(d_flat ** 2) / (2.0 * sig2))

    A = sp.coo_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float64)
    W = A.maximum(A.T).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W.astype(np.float64)


def compute_laplacian(W: sp.spmatrix, normalized: bool = True) -> sp.spmatrix:
    """Graph Laplacian.

    normalized=True  -> random-walk L_rw = I - D^{-1} W
    normalized=False -> unnormalized L = D - W
    """
    W = sp.csr_matrix(W)
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    if normalized:
        d_safe = np.where(deg > 0, deg, 1.0)
        D_inv = sp.diags(1.0 / d_safe)
        L = sp.eye(W.shape[0], format="csr") - D_inv @ W
    else:
        D = sp.diags(deg)
        L = D - W
    return L.tocsr()


def spectral_gap_analysis(L: sp.spmatrix, n_eigenvalues: int = 10) -> np.ndarray:
    """Smallest n_eigenvalues of L (sorted ascending), real parts only.

    L_rw = I - D^{-1} W is asymmetric but similar to the symmetric L_sym, so its
    spectrum is real. We use general eigvals and take the real part rather than
    symmetrizing (which would change the spectrum).
    """
    n = L.shape[0]
    n_eig = min(int(n_eigenvalues), max(1, n - 2))
    L_dense = L.toarray() if sp.issparse(L) else np.asarray(L)
    vals = np.linalg.eigvals(L_dense)
    vals = np.real(vals)
    vals.sort()
    return vals[:n_eig].astype(np.float64)


def spectral_gap_from_W(W: sp.spmatrix, n_eigenvalues: int = 10) -> np.ndarray:
    """Smallest n_eigenvalues of L_sym = I - D^{-1/2} W D^{-1/2} (sparse, fast).

    L_sym is symmetric and shares its spectrum with the random-walk Laplacian,
    so this is the recommended fast path for the parameter sweeps.
    """
    W = sp.csr_matrix(W).astype(np.float64)
    n = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    L_sym = sp.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) * 0.5

    n_eig = min(int(n_eigenvalues), max(1, n - 2))
    if n <= 400:
        vals = np.linalg.eigvalsh(L_sym.toarray())
        vals = np.sort(np.real(vals))
        return vals[:n_eig].astype(np.float64)
    try:
        vals, _ = eigsh(L_sym, k=n_eig, which="SM", tol=1e-4)
    except Exception:
        try:
            sigma = -1e-3
            vals, _ = eigsh(L_sym, k=n_eig, sigma=sigma, which="LM", tol=1e-4)
        except Exception:
            vals = np.linalg.eigvalsh(L_sym.toarray())[:n_eig]
    vals = np.sort(np.real(vals))
    return vals.astype(np.float64)


def diffusion_map_embedding(
    W: sp.spmatrix,
    n_components: int = 3,
    t: int = 5,
) -> np.ndarray:
    """Diffusion map embedding (Coifman & Lafon).

    Returns (N, n_components) array — the leading non-trivial eigenvectors of
    P = D^{-1} W weighted by lambda^t. Computed via the symmetric similar
    matrix M = D^{-1/2} W D^{-1/2}, whose eigenvectors map back to those of P.
    """
    W = sp.csr_matrix(W).astype(np.float64)
    n = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    M = D_inv_sqrt @ W @ D_inv_sqrt
    M = (M + M.T) * 0.5

    k_eig = min(n_components + 1, n - 1)
    if k_eig < 1:
        return np.zeros((n, n_components), dtype=np.float64)

    if n <= 400:
        vals_all, vecs_all = np.linalg.eigh(M.toarray())
        order = np.argsort(-vals_all)
        vals = vals_all[order][:k_eig]
        vecs = vecs_all[:, order][:, :k_eig]
    else:
        try:
            vals, vecs = eigsh(M, k=k_eig, which="LA")
            order = np.argsort(-vals)
            vals = vals[order]
            vecs = vecs[:, order]
        except Exception:
            vals_all, vecs_all = np.linalg.eigh(M.toarray())
            order = np.argsort(-vals_all)
            vals = vals_all[order][:k_eig]
            vecs = vecs_all[:, order][:, :k_eig]

    phi = D_inv_sqrt @ vecs
    if k_eig <= 1:
        return np.zeros((n, n_components), dtype=np.float64)
    take = min(n_components, k_eig - 1)
    coords = phi[:, 1:1 + take] * (vals[1:1 + take] ** float(t))[None, :]
    if take < n_components:
        coords = np.concatenate([coords, np.zeros((n, n_components - take))], axis=1)
    return np.asarray(coords, dtype=np.float64)


def _build_label_matrix(labels: np.ndarray, labeled_mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Build N x C indicator matrix Y over the union of labeled classes."""
    labels = np.asarray(labels)
    classes = sorted(set(int(c) for c in labels[labeled_mask]))
    C = len(classes)
    N = labels.shape[0]
    Y = np.zeros((N, C), dtype=np.float64)
    cls_to_col = {c: i for i, c in enumerate(classes)}
    for i in np.where(labeled_mask)[0]:
        Y[i, cls_to_col[int(labels[i])]] = 1.0
    return Y, classes


def label_propagation(
    W: sp.spmatrix,
    labels: np.ndarray,
    labeled_mask: np.ndarray,
) -> np.ndarray:
    """Solve f_u = -L_uu^{-1} L_ul f_l (Section 9.9.2).

    Returns predicted integer label per node (labeled nodes preserved).
    Uses the unnormalized Laplacian on the labeled/unlabeled partition.
    """
    W = sp.csr_matrix(W).astype(np.float64)
    N = W.shape[0]
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    L = compute_laplacian(W, normalized=False).tolil()

    l_idx = np.where(labeled_mask)[0]
    u_idx = np.where(~labeled_mask)[0]
    Y, classes = _build_label_matrix(labels, labeled_mask)
    if u_idx.size == 0:
        return np.asarray(labels).copy()

    L = L.tocsr()
    L_uu = L[u_idx, :][:, u_idx]
    L_ul = L[u_idx, :][:, l_idx]
    F_l = Y[l_idx, :]

    L_uu = L_uu + sp.eye(L_uu.shape[0]) * 1e-8
    rhs = -L_ul @ F_l
    if L_uu.shape[0] == 0:
        F_u = np.zeros((0, F_l.shape[1]))
    else:
        try:
            F_u = spsolve(L_uu.tocsc(), rhs)
            if F_u.ndim == 1:
                F_u = F_u.reshape(-1, 1)
        except Exception:
            F_u = np.linalg.solve(L_uu.toarray(), rhs)

    pred = np.asarray(labels).copy()
    if F_u.shape[0] > 0:
        cls_arr = np.array(classes, dtype=int)
        pred_u = cls_arr[np.argmax(F_u, axis=1)]
        pred[u_idx] = pred_u
    return pred


def laplacian_regularization(
    W: sp.spmatrix,
    labels: np.ndarray,
    labeled_mask: np.ndarray,
    mu: float = 1.0,
) -> np.ndarray:
    """Solve (L + mu*J) F = mu * Y for one-hot label matrix Y (Section 9.9.1).

    L is the (unnormalized) Laplacian. J is diag(labeled_mask).
    Returns predicted integer label per node.
    """
    W = sp.csr_matrix(W).astype(np.float64)
    N = W.shape[0]
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    L = compute_laplacian(W, normalized=False)
    Y, classes = _build_label_matrix(labels, labeled_mask)

    J = sp.diags(labeled_mask.astype(np.float64))
    A = L + mu * J + sp.eye(N) * 1e-8
    rhs = mu * Y
    try:
        F = spsolve(A.tocsc(), rhs)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
    except Exception:
        F = np.linalg.solve(A.toarray(), rhs)

    pred = np.asarray(labels).copy()
    if F.size:
        cls_arr = np.array(classes, dtype=int)
        pred = cls_arr[np.argmax(F, axis=1)]
        for i in np.where(labeled_mask)[0]:
            pred[i] = labels[i]
    return pred


def spectral_clustering(
    W: sp.spmatrix,
    k: int,
    n_init: int = 10,
    random_state: int | None = 0,
) -> np.ndarray:
    """Normalized-Laplacian spectral clustering (Ng-Jordan-Weiss style).

    Computes the bottom-k eigenvectors of L_sym = I - D^{-1/2} W D^{-1/2},
    row-normalizes them to unit length, and runs k-means.
    """
    W = sp.csr_matrix(W).astype(np.float64)
    N = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    L_sym = sp.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) * 0.5

    k = max(1, int(k))
    n_eig = min(k, N - 1) if N > 1 else 1

    if N <= 400:
        vals, vecs = np.linalg.eigh(L_sym.toarray())
        order = np.argsort(vals)
        U = vecs[:, order][:, :n_eig]
    else:
        try:
            vals, vecs = eigsh(L_sym, k=n_eig, which="SM")
            order = np.argsort(vals)
            U = vecs[:, order]
        except Exception:
            vals, vecs = np.linalg.eigh(L_sym.toarray())
            order = np.argsort(vals)
            U = vecs[:, order][:, :n_eig]

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    U = U / norms

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    return km.fit_predict(U)
