"""ResNet-18 baseline for log-Mel spectrogram classification.

Single-channel input (1 x 128 x T). Used by Level 1 and Level 2 notebooks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18


class SpecDataset(TensorDataset):
    pass


def make_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 with single-channel first conv and a fresh FC head."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _normalize_specs(S: np.ndarray) -> np.ndarray:
    """Per-spectrogram zero-mean unit-variance, then a single global rescale.

    Models train more stably when each input has comparable energy.
    """
    S = S.astype(np.float32)
    mean = S.mean(axis=(1, 2), keepdims=True)
    std = S.std(axis=(1, 2), keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return ((S - mean) / std).astype(np.float32)


def train_resnet_baseline(
    S_train: np.ndarray,
    y_train: np.ndarray,
    S_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Train ResNet-18 on Mel spectrograms and return predictions on the test split."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = _normalize_specs(S_train)[:, None, :, :]
    Xte = _normalize_specs(S_test)[:, None, :, :]
    ytr = np.asarray(y_train, dtype=np.int64)
    yte = np.asarray(y_test, dtype=np.int64)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(Xte)
    yte_t = torch.from_numpy(yte)

    ds_tr = TensorDataset(Xtr_t, ytr_t)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    model = make_resnet18(num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        losses.append(ep_loss / max(n, 1))
        if verbose:
            print(f"  epoch {ep+1}/{epochs}: train_loss={losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, Xte_t.shape[0], batch_size):
            xb = Xte_t[i:i+batch_size].to(device)
            out = model(xb)
            preds.append(out.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)

    return {
        "y_pred": y_pred,
        "y_true": yte,
        "losses": losses,
        "device": device,
    }
