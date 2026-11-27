"""Model L: LSTM on 24 h windows of 8 channels (protocol sections 7 and 7.2).

Inputs and the target are standardized with statistics of the training rows only. The loss is
MSE on the standardized target, which has the same minimizer as MSE in nT; the validation
error used for early stopping is reported in nT^2. With early stopping the weights of the best
epoch are restored, and n_epochs is that epoch's number (1-based).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import nn

BATCH = 256
MAX_EPOCHS = 100
PATIENCE = 5
PREDICT_BATCH = 4096

GRID = [
    {"hidden": h, "lr": lr, "layers": n}
    for h, lr, n in product((32, 64, 128), (1e-3, 3e-3), (1, 2))
]

torch.set_num_threads(os.cpu_count() or 1)


class Net(nn.Module):
    def __init__(self, n_channels: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(1)


@dataclass
class Scaler:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def from_train(cls, x: np.ndarray, y: np.ndarray) -> "Scaler":
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        return cls(flat.mean(0), flat.std(0), float(np.mean(y)), float(np.std(y)))

    def x(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((x - self.x_mean) / self.x_std).astype(np.float32))

    def y(self, y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((y - self.y_mean) / self.y_std).astype(np.float32))


@dataclass
class FittedL:
    net: Net
    scaler: Scaler
    n_epochs: int
    val_mse: list[float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _predict(self.net, self.scaler, x)

    def save(self, path) -> None:
        torch.save({
            "state": self.net.state_dict(),
            "n_channels": self.net.lstm.input_size,
            "hidden": self.net.lstm.hidden_size,
            "layers": self.net.lstm.num_layers,
            "scaler": {k: np.asarray(v).tolist() for k, v in vars(self.scaler).items()},
            "n_epochs": self.n_epochs,
        }, path)


def load(path) -> FittedL:
    d = torch.load(path, weights_only=True)
    net = Net(d["n_channels"], d["hidden"], d["layers"])
    net.load_state_dict(d["state"])
    s = d["scaler"]
    scaler = Scaler(np.array(s["x_mean"]), np.array(s["x_std"]), float(s["y_mean"]), float(s["y_std"]))
    return FittedL(net, scaler, d["n_epochs"], [])


def _check(x: np.ndarray, y: np.ndarray | None = None) -> None:
    if x.ndim != 3 or np.isnan(x).any():
        raise ValueError("x must be complete windows of shape (n, hours, channels)")
    if y is not None and (len(y) != len(x) or np.isnan(y).any()):
        raise ValueError("y must be a complete target for every window")


@torch.no_grad()
def _predict(net: Net, scaler: Scaler, x: np.ndarray) -> np.ndarray:
    _check(x)
    net.eval()
    xt = scaler.x(x)
    out = [net(xt[b:b + PREDICT_BATCH]) for b in range(0, len(xt), PREDICT_BATCH)]
    return torch.cat(out).numpy().astype(np.float64) * scaler.y_std + scaler.y_mean


def fit(x: np.ndarray, y: np.ndarray, config: dict, seed: int, *,
        n_epochs: int | None = None, x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None, max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE) -> FittedL:
    """Fit with early stopping on (x_val, y_val), or for a fixed number of epochs.

    Exactly one of `n_epochs` and the validation pair must be given. The caller passes only
    complete windows with a known target. The seed controls initialization and batch order.
    """
    _check(x, y)
    if (n_epochs is None) == (x_val is None) or (x_val is None) != (y_val is None):
        raise ValueError("give either n_epochs or a validation set, not both")
    if x_val is not None:
        _check(x_val, y_val)

    scaler = Scaler.from_train(x, y)
    xt, yt = scaler.x(x), scaler.y(y)
    torch.manual_seed(seed)
    net = Net(x.shape[-1], config["hidden"], config["layers"])
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    order = torch.Generator().manual_seed(seed)

    epochs = n_epochs if n_epochs is not None else max_epochs
    best, best_epoch, best_state, val_mse = np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(xt), generator=order)
        for b in range(0, len(xt), BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if x_val is None:
            continue
        mse = float(np.mean((_predict(net, scaler, x_val) - y_val) ** 2))
        val_mse.append(mse)
        if mse < best:
            best, best_epoch, best_state = mse, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if x_val is None:
        return FittedL(net, scaler, epochs, val_mse)
    net.load_state_dict(best_state)
    return FittedL(net, scaler, best_epoch, val_mse)
