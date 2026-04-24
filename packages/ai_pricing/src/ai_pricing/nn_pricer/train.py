"""Train the NN Pricer.

Usage (from repo root, after `pip install -e packages/pricing packages/ai_pricing torch`):

    python -m ai_pricing.nn_pricer.train --n 100000 --epochs 30 --out models/nn_pricer.pt

Small n and epochs for CPU smoke test; full run on GPU uses n=500k, epochs=50.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from ai_pricing.nn_pricer.data import generate_training_set
from ai_pricing.nn_pricer.model import NNPricer, NNPricerConfig


def train(
    n: int = 100_000,
    epochs: int = 30,
    batch: int = 2048,
    lr: float = 1e-3,
    out_path: str = "models/nn_pricer.pt",
    device: str | None = None,
    seed: int = 42,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError("pip install torch") from e

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[nn_pricer] Generating {n} samples...")
    t0 = time.time()
    data = generate_training_set(n=n, seed=seed)
    print(f"[nn_pricer] Data ready in {time.time() - t0:.1f}s")

    X_tr = torch.from_numpy(data["X_train"])
    y_tr = torch.from_numpy(data["y_train"])
    X_va = torch.from_numpy(data["X_val"]).to(device)
    y_va = torch.from_numpy(data["y_val"]).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"))

    model = NNPricer(NNPricerConfig()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for ep in range(epochs):
        model.train()
        tot = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = tot / n_seen
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg.__dict__}, out_path)
            tag = " ✓"
        print(f"[epoch {ep+1:3d}/{epochs}] train_mse={train_loss:.2e} "
              f"val_mse={val_loss:.2e}{tag}")

    # Final test evaluation
    X_te = torch.from_numpy(data["X_test"]).to(device)
    y_te = torch.from_numpy(data["y_test"]).to(device)
    model.load_state_dict(torch.load(out_path, map_location=device)["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te).cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    rel_err = np.abs(pred_te - y_te_np) / np.maximum(y_te_np, 1e-4)
    print(f"[TEST] MSE={np.mean((pred_te - y_te_np)**2):.2e}  "
          f"mean rel err={rel_err.mean():.4f}  "
          f"p95 rel err={np.quantile(rel_err, 0.95):.4f}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/nn_pricer.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.n, args.epochs, args.batch, args.lr, args.out, args.device, args.seed)


if __name__ == "__main__":
    main()
