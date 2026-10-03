"""Evaluation metrics and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoids display issues when notebooks save figures
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def classification_report_custom(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | np.ndarray,
) -> dict:
    """Accuracy + macro F1 + per-class precision/recall/F1.

    Returns a dict the notebooks can display or write to disk.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_names = list(label_names)
    labels_idx = list(range(len(label_names)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_idx, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred,
        labels=labels_idx,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {label_names[i]: report[label_names[i]] for i in labels_idx},
        "raw": report,
    }


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | np.ndarray,
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> None:
    label_names = list(label_names)
    labels_idx = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_disp = cm.astype(np.float64)
    if normalize:
        row_sums = cm_disp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_disp = cm_disp / row_sums

    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 0.5), max(5, len(label_names) * 0.5)))
    im = ax.imshow(cm_disp, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0 if normalize else cm.max())
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if len(label_names) <= 12:
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                v = cm_disp[i, j]
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(cm[i, j])}",
                        ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_ensure_parent(save_path), dpi=150)
    plt.close(fig)


def plot_spectral_gap(eigenvalues: np.ndarray, save_path: str, title: str = "Spectral gap") -> None:
    eigenvalues = np.asarray(eigenvalues).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, eigenvalues.size + 1), eigenvalues, marker="o")
    ax.set_xlabel("Eigenvalue index k")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_parent(save_path), dpi=150)
    plt.close(fig)


def plot_embedding_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | np.ndarray,
    title: str,
    save_path: str,
) -> None:
    coords = np.asarray(coords)[:, :2]
    labels = np.asarray(labels)
    label_names = list(label_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = sorted(set(int(l) for l in labels))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    for i, lab in enumerate(uniq):
        m = labels == lab
        name = label_names[lab] if 0 <= lab < len(label_names) else str(lab)
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.7, color=cmap(i), label=name)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if len(uniq) <= 12:
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(_ensure_parent(save_path), dpi=150)
    plt.close(fig)


def plot_silhouette_vs_k(silhouette_scores: list[float] | np.ndarray, k_range: list[int] | np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), list(silhouette_scores), marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette score")
    ax.set_title("Silhouette score vs k")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_parent(save_path), dpi=150)
    plt.close(fig)


def plot_parameter_heatmap(
    k_values: list[int] | np.ndarray,
    sigma_values: list,
    accuracy_matrix: np.ndarray,
    save_path: str,
    title: str = "Accuracy over (k, sigma)",
) -> None:
    accuracy_matrix = np.asarray(accuracy_matrix)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(accuracy_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(k_values)))
    ax.set_yticks(range(len(sigma_values)))
    ax.set_xticklabels(list(k_values))
    ax.set_yticklabels([str(s) for s in sigma_values])
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title(title)
    for i in range(len(sigma_values)):
        for j in range(len(k_values)):
            ax.text(j, i, f"{accuracy_matrix[i, j]:.2f}",
                    ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_ensure_parent(save_path), dpi=150)
    plt.close(fig)
