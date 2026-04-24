"""Train DeepCalibNet: Heston params → IV surface.

Heavy data generation step (Heston semi-analytic × 25 points × N samples). Use small N
for smoke test; production run 100k on GPU / overnight.

Usage:
    python -m ai_pricing.deep_calib.train --n 2000 --epochs 40 \
        --out models/deep_calib.pt --data data/deep_calib_cache.npz
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from ai_pricing.deep_calib.model import DeepCalibNet, DeepCalibConfig
from ai_pricing.deep_calib.sampler import sample_heston_params
from ai_pricing.deep_calib.surface import N_POINTS, iv_surface_batch


def build_dataset(n: int, seed: int, data_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        z = np.load(data_path)
        print(f"[deep_calib] loaded cache {data_path}: {len(z['X'])} samples")
        return z["X"], z["Y"]

    print(f"[deep_calib] sampling {n} Heston param tuples + computing IV surfaces...")
    t0 = time.time()
    X = sample_heston_params(n, seed=seed)
    Y = iv_surface_batch(X)
    # drop rows with NaN (IV solver failure)
    valid = ~np.isnan(Y).any(axis=1)
    X, Y = X[valid], Y[valid]
    print(f"[deep_calib] data ready: {len(X)} valid / {n} total in {time.time() - t0:.1f}s")

    if data_path:
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, X=X, Y=Y)
    return X, Y


def train(
    n: int = 2_000,
    epochs: int = 40,
    batch: int = 256,
    lr: float = 1e-3,
    out: str = "models/deep_calib.pt",
    data_path: str = "data/deep_calib_cache.npz",
    device: str | None = None,
    seed: int = 42,
):
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError("pip install torch") from e

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, Y = build_dataset(n, seed, data_path)

    # Normalize inputs and outputs
    x_mean, x_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = Y.mean(0), Y.std(0) + 1e-8
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    n_train = int(0.9 * len(Xn))
    X_tr = torch.from_numpy(Xn[:n_train]).float()
    Y_tr = torch.from_numpy(Yn[:n_train]).float()
    X_va = torch.from_numpy(Xn[n_train:]).float().to(device)
    Y_va = torch.from_numpy(Yn[n_train:]).float().to(device)

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch, shuffle=True)

    model = DeepCalibNet(DeepCalibConfig()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        tot, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n_seen += len(xb)

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), Y_va).item()

        tag = ""
        if val < best_val:
            best_val = val
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": model.cfg.__dict__,
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
            }, out)
            tag = " ✓"
        print(f"[epoch {ep+1:3d}/{epochs}] train={tot/n_seen:.2e} val={val:.2e}{tag}")

    # Rough IV RMSE in vol points (de-normalized)
    model.load_state_dict(torch.load(out, map_location=device)["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(X_va).cpu().numpy() * y_std + y_mean
    true = Y_va.cpu().numpy() * y_std + y_mean
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    print(f"[TEST] IV RMSE = {rmse*100:.3f} vol points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/deep_calib.pt")
    ap.add_argument("--data", dest="data_path", type=str, default="data/deep_calib_cache.npz")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    train(args.n, args.epochs, args.batch, args.lr, args.out, args.data_path, args.device)


if __name__ == "__main__":
    main()
