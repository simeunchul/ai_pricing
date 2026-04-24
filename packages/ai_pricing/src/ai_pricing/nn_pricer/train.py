"""Train the NN Pricer.

Usage (from repo root, after `pip install -e packages/pricing packages/ai_pricing torch`):

    # Classic MSE (original Hutchinson 1994)
    python -m ai_pricing.nn_pricer.train --n 100000 --epochs 30 --loss mse --out models/nn_pricer_mse.pt

    # Relative MSE (Ferguson-Green 2018) — best bang-for-buck for underfit problem
    python -m ai_pricing.nn_pricer.train --n 80000 --epochs 40 --loss rel --out models/nn_pricer_rel.pt

    # Log-space MSE
    python -m ai_pricing.nn_pricer.train --n 80000 --epochs 40 --loss log --out models/nn_pricer_log.pt

Small n and epochs for CPU smoke test; full run on GPU uses n=500k, epochs=50.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ai_pricing.nn_pricer.data import generate_training_set
from ai_pricing.nn_pricer.model import NNPricer, NNPricerConfig


LOSS_CHOICES = ("mse", "rel", "log", "hybrid")


def make_loss(name: str, eps: float = 1e-2, hybrid_alpha: float = 0.5):
    """Return (train_fn, eval_fn). eval_fn is always raw MSE for comparability.

    eps chosen to match "meaningful price floor": target y = C/K lives in [0, ~1.4].
    eps=0.01 treats prices below 1% of K as noise, preventing deep OTM from
    dominating the gradient (which would be the mirror image of the MSE problem).

    hybrid = alpha * log_mse + (1 - alpha) * mse. Balances log loss's scale-
    invariance with MSE's preservation of absolute fit (helps deep ITM trade-off
    observed with pure log-MSE).
    """
    import torch

    mse = torch.nn.MSELoss()

    def log_mse(pred, y):
        return ((torch.log(pred.clamp_min(eps)) - torch.log(y.clamp_min(eps))) ** 2).mean()

    if name == "mse":
        return mse, mse
    if name == "rel":
        def rel_mse(pred, y):
            return (((pred - y) / (y.abs() + eps)) ** 2).mean()
        return rel_mse, mse
    if name == "log":
        return log_mse, mse
    if name == "hybrid":
        def combo(pred, y):
            return hybrid_alpha * log_mse(pred, y) + (1.0 - hybrid_alpha) * mse(pred, y)
        return combo, mse

    raise ValueError(f"Unknown loss: {name}. Choose from {LOSS_CHOICES}")


def train(
    n: int = 100_000,
    epochs: int = 30,
    batch: int = 2048,
    lr: float = 1e-3,
    out_path: str = "models/nn_pricer.pt",
    device: str | None = None,
    seed: int = 42,
    loss: str = "mse",
    hybrid_alpha: float = 0.5,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError("pip install torch") from e

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[nn_pricer] Generating {n} samples... (loss={loss}, device={device})")
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
    train_loss_fn, eval_loss_fn = make_loss(loss, hybrid_alpha=hybrid_alpha)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mse": []}

    for ep in range(epochs):
        model.train()
        tot = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss_val = train_loss_fn(pred, yb)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            tot += loss_val.item() * len(xb)
            n_seen += len(xb)
        train_loss = tot / n_seen
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = train_loss_fn(val_pred, y_va).item()          # loss-fn specific
            val_mse = eval_loss_fn(val_pred, y_va).item()             # always MSE (comparable)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)

        # Checkpoint on the training objective (val_loss) so each run saves the
        # model that minimizes its own loss. Cross-run comparison happens in bench
        # script using the universal IV-space metric.
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg.__dict__,
                        "loss_name": loss}, out_path)
            tag = " [best]"
        print(f"[epoch {ep+1:3d}/{epochs}] train_{loss}={train_loss:.2e} "
              f"val_{loss}={val_loss:.2e}  val_mse={val_mse:.2e}{tag}")

    # Final test evaluation
    X_te = torch.from_numpy(data["X_test"]).to(device)
    y_te = torch.from_numpy(data["y_test"]).to(device)
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te).cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    mse = float(np.mean((pred_te - y_te_np) ** 2))
    rel_err = np.abs(pred_te - y_te_np) / np.maximum(y_te_np, 1e-4)
    print(f"[TEST] MSE={mse:.2e}  "
          f"mean rel err={rel_err.mean():.4f}  "
          f"p95 rel err={np.quantile(rel_err, 0.95):.4f}")
    print(f"[TEST] NOTE: run scripts/bench_nn_pricer.py for IV-space + strata report.")
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
    ap.add_argument("--loss", type=str, default="mse", choices=LOSS_CHOICES,
                    help="Training loss: mse | rel | log | hybrid")
    ap.add_argument("--hybrid-alpha", type=float, default=0.5,
                    help="Weight of log-MSE in hybrid loss (0..1). Ignored if --loss != hybrid.")
    args = ap.parse_args()
    train(args.n, args.epochs, args.batch, args.lr, args.out, args.device, args.seed,
          args.loss, args.hybrid_alpha)


if __name__ == "__main__":
    main()
