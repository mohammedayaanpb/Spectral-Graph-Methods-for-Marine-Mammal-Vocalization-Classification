"""Generate spectrogram grid figures for the k=3 spectral clusters (V2 capped pool)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph import build_knn_graph, spectral_clustering

FEATURES_PATH = PROJECT_ROOT / "Level3" / "Data" / "level3_features.npz"
SPECS_PATH = PROJECT_ROOT / "Level3" / "Data" / "level3_spectrograms.npz"
FIG_DIR = PROJECT_ROOT / "v2" / "Level3" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_WINDOWS_PER_CLIP = 3
K_GRAPH = 7
K_CLUSTERS = 3
FLATNESS_COL = 29  # mean spectral flatness in the Level 3 46-dim feature vector
PER_CLUSTER_SAMPLES = 8

feat = np.load(FEATURES_PATH, allow_pickle=True)
spec = np.load(SPECS_PATH, allow_pickle=True)
X_full = feat["X"]
clip_ids_full = feat["clip_ids"].astype(np.int64)
S_full = spec["S"]

assert X_full.shape[0] == S_full.shape[0] == clip_ids_full.shape[0]

RNG_CAP = np.random.default_rng(42)
keep = []
for cid in np.unique(clip_ids_full):
    rows = np.where(clip_ids_full == cid)[0]
    if rows.size <= MAX_WINDOWS_PER_CLIP:
        keep.append(rows)
    else:
        pick = RNG_CAP.choice(rows, size=MAX_WINDOWS_PER_CLIP, replace=False)
        keep.append(np.sort(pick))
keep_idx = np.concatenate(keep)

X = X_full[keep_idx]
S = S_full[keep_idx]
clip_ids = clip_ids_full[keep_idx]
print(f"Capped pool: {X.shape[0]} windows from {len(np.unique(clip_ids))} clips")

Xz = StandardScaler().fit_transform(X).astype(np.float32)
W = build_knn_graph(Xz, k=K_GRAPH, sigma="median")
labels = spectral_clustering(W, k=K_CLUSTERS, random_state=0)

unique, counts = np.unique(labels, return_counts=True)
print("Cluster sizes:", dict(zip(unique.tolist(), counts.tolist())))

flatness = X[:, FLATNESS_COL]
print("Mean spectral flatness per cluster:")
for c in range(K_CLUSTERS):
    members = np.where(labels == c)[0]
    if members.size:
        print(f"  cluster {c}: {flatness[members].mean():.4f}  (n={members.size})")
    else:
        print(f"  cluster {c}: empty")

RNG_PLOT = np.random.default_rng(0)


def sample_indices(members: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if members.size <= n:
        return members
    return np.sort(rng.choice(members, size=n, replace=False))


for c in range(K_CLUSTERS):
    members = np.where(labels == c)[0]
    pick = sample_indices(members, PER_CLUSTER_SAMPLES, RNG_PLOT)
    n_total = members.size
    n_show = pick.size
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
    axes = np.atleast_2d(axes)
    for ax_idx in range(rows * cols):
        ax = axes.flat[ax_idx]
        if ax_idx < n_show:
            wi = pick[ax_idx]
            ax.imshow(S[wi], aspect="auto", origin="lower")
            ax.set_title(f"clip {int(clip_ids[wi])} (win {int(wi)})", fontsize=9)
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Mel bin")
        else:
            ax.axis("off")
    fig.suptitle(f"Cluster {c} — {n_total} windows", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / f"cluster_{c}_spectrograms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

OVERVIEW_COLS = 4
fig, axes = plt.subplots(K_CLUSTERS, OVERVIEW_COLS, figsize=(14, 8))
axes = np.atleast_2d(axes)
RNG_OVERVIEW = np.random.default_rng(1)
for c in range(K_CLUSTERS):
    members = np.where(labels == c)[0]
    pick = sample_indices(members, OVERVIEW_COLS, RNG_OVERVIEW)
    n_total = members.size
    for j in range(OVERVIEW_COLS):
        ax = axes[c, j]
        if j < pick.size:
            wi = pick[j]
            ax.imshow(S[wi], aspect="auto", origin="lower")
            ax.set_title(f"clip {int(clip_ids[wi])}", fontsize=9)
            ax.set_xlabel("Time frame")
            if j == 0:
                ax.set_ylabel(f"Cluster {c}\n(n={n_total})\nMel bin")
            else:
                ax.set_ylabel("Mel bin")
        else:
            ax.axis("off")
fig.suptitle("Level 3 V2 (capped) — spectral clustering k=3", fontsize=13)
plt.tight_layout()
out_overview = FIG_DIR / "cluster_spectrograms_overview.png"
fig.savefig(out_overview, dpi=150)
plt.close(fig)
print(f"wrote {out_overview}")
