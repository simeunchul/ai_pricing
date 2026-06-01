"""Multi-symbol LSTM 시계열 모델 — 외국인+기관 + 차트 features 시퀀스 학습.

설계:
  - Symbol embedding (각 종목 8차원 vector)
  - LSTM (hidden=64, 2 layers, dropout 0.2)
  - 입력: (batch, seq_len=20, n_features=15)
  - 출력: 다음날 수익률 ≥ +0.5% 확률 (binary)

이 모델은 시계열 의존성을 자동 학습 (LightGBM 대비 장점).
종목별 차이는 symbol embedding 으로 학습.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from autotrader.ml.dual_signal_classifier import FEATURE_COLS, build_features

logger = logging.getLogger(__name__)


SEQ_LEN = 20
EMBED_DIM = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class LSTMSignalModel(nn.Module):
    def __init__(
        self,
        n_features: int = len(FEATURE_COLS),
        n_symbols: int = 50,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.n_features = n_features
        self.symbol_embed = nn.Embedding(n_symbols, embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, sym_idx: torch.Tensor) -> torch.Tensor:
        """
        x:       (batch, seq_len, n_features)
        sym_idx: (batch,) long
        returns: (batch,) logit
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                  # (batch, hidden)
        sym_emb = self.symbol_embed(sym_idx)   # (batch, embed)
        combined = torch.cat([last_hidden, sym_emb], dim=-1)
        logit = self.head(combined).squeeze(-1)
        return logit


class SequenceDataset(Dataset):
    """종목별 sliding window sequences.

    각 sample: (X[t-SEQ_LEN:t], symbol_idx) → y[t+1] (다음날 +0.5% binary)
    """
    def __init__(
        self,
        features_df: pd.DataFrame,
        symbol_to_idx: dict[str, int],
        seq_len: int = SEQ_LEN,
    ):
        self.seq_len = seq_len
        self.symbol_to_idx = symbol_to_idx
        self.samples = []   # list of (X, sym_idx, y)

        for sym, sub in features_df.groupby("symbol"):
            sub = sub.sort_values("date").reset_index(drop=True)
            X = sub[FEATURE_COLS].values.astype(np.float32)
            y = sub["target"].values.astype(np.float32)
            sym_idx = symbol_to_idx[sym]

            # sliding window: window 끝점 t 에서 y[t+1] 예측
            # target 은 이미 build_features 에서 next-day 정의됨 (T+1 수익률)
            # → window [t-seq_len+1 : t+1] (= seq_len 개) → predict y[t]
            for t in range(seq_len - 1, len(X)):
                window = X[t - seq_len + 1: t + 1]
                if window.shape[0] != seq_len:
                    continue
                if np.isnan(window).any():
                    continue
                self.samples.append((window, sym_idx, y[t]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, sym_idx, y = self.samples[idx]
        return (
            torch.from_numpy(X),
            torch.tensor(sym_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


@dataclass
class LSTMTrainResult:
    model: nn.Module
    symbol_to_idx: dict[str, int]
    train_auc: float
    test_auc: float
    test_accuracy: float
    n_train: int
    n_test: int
    test_predictions: pd.DataFrame    # date, symbol, pred_proba, target
    train_history: list[dict]


def train_lstm(
    features_df: pd.DataFrame,
    train_split_date: str | None = None,
    train_frac: float = 0.8,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seq_len: int = SEQ_LEN,
    device: str | None = None,
) -> LSTMTrainResult:
    from sklearn.metrics import roc_auc_score, accuracy_score

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [LSTM] device={device}")

    df = features_df.sort_values("date").reset_index(drop=True)
    if train_split_date:
        cutoff = pd.Timestamp(train_split_date)
        train_df = df[df["date"] < cutoff].copy()
        test_df = df[df["date"] >= cutoff].copy()
    else:
        n_train = int(len(df) * train_frac)
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()

    symbols = sorted(set(features_df["symbol"]))
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    train_ds = SequenceDataset(train_df, symbol_to_idx, seq_len=seq_len)
    test_ds = SequenceDataset(test_df, symbol_to_idx, seq_len=seq_len)
    print(f"  [LSTM] train sequences: {len(train_ds)}, test sequences: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LSTMSignalModel(
        n_features=len(FEATURE_COLS),
        n_symbols=len(symbols),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_test_auc = 0.0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, sym, y in train_loader:
            x, sym, y = x.to(device), sym.to(device), y.to(device)
            optim.zero_grad()
            logit = model(x, sym)
            loss = loss_fn(logit, y)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        # eval
        model.eval()
        all_train_pred, all_train_y = [], []
        all_test_pred, all_test_y = [], []
        with torch.no_grad():
            for x, sym, y in train_loader:
                x, sym = x.to(device), sym.to(device)
                p = torch.sigmoid(model(x, sym)).cpu().numpy()
                all_train_pred.extend(p.tolist())
                all_train_y.extend(y.numpy().tolist())
            for x, sym, y in test_loader:
                x, sym = x.to(device), sym.to(device)
                p = torch.sigmoid(model(x, sym)).cpu().numpy()
                all_test_pred.extend(p.tolist())
                all_test_y.extend(y.numpy().tolist())

        train_auc = roc_auc_score(all_train_y, all_train_pred) if len(set(all_train_y)) > 1 else float("nan")
        test_auc = roc_auc_score(all_test_y, all_test_pred) if len(set(all_test_y)) > 1 else float("nan")
        avg_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch + 1, "train_loss": avg_loss,
                        "train_auc": train_auc, "test_auc": test_auc})
        if test_auc > best_test_auc:
            best_test_auc = test_auc
        print(f"  epoch {epoch+1:>2}/{epochs}  loss={avg_loss:.4f}  "
              f"train_auc={train_auc:.4f}  test_auc={test_auc:.4f}")

    # 최종 test predictions
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for x, sym, y in test_loader:
            x_d, sym_d = x.to(device), sym.to(device)
            p = torch.sigmoid(model(x_d, sym_d)).cpu().numpy()
            test_preds_list.extend(p.tolist())

    test_acc = accuracy_score(
        [int(v >= 0.5) for v in all_test_y],
        [int(v >= 0.5) for v in all_test_pred],
    ) if all_test_y else 0.0

    # test_predictions DataFrame — sequence sample 별 (date 매핑)
    # sample order matches test_ds samples (sorted by sym, then date)
    pred_rows = []
    for (X, sym_idx, y), p in zip(test_ds.samples, test_preds_list):
        # Note: test_ds samples are unordered; we rebuild date by re-iterating
        pred_rows.append({"sym_idx": int(sym_idx), "target": float(y), "pred_proba": float(p)})
    pred_df = pd.DataFrame(pred_rows)

    return LSTMTrainResult(
        model=model.cpu(),
        symbol_to_idx=symbol_to_idx,
        train_auc=train_auc,
        test_auc=test_auc,
        test_accuracy=test_acc,
        n_train=len(train_ds),
        n_test=len(test_ds),
        test_predictions=pred_df,
        train_history=history,
    )


def predict_lstm(
    model: LSTMSignalModel,
    features_df: pd.DataFrame,
    symbol_to_idx: dict[str, int],
    seq_len: int = SEQ_LEN,
    device: str = "cpu",
) -> pd.DataFrame:
    """학습된 LSTM 으로 features_df 의 모든 (sym, date) 점수 예측.

    Returns: features_df + pred_proba 컬럼.
    """
    model = model.to(device).eval()
    out_rows = []
    with torch.no_grad():
        for sym, sub in features_df.groupby("symbol"):
            if sym not in symbol_to_idx:
                continue
            sub = sub.sort_values("date").reset_index(drop=True)
            X = sub[FEATURE_COLS].values.astype(np.float32)
            for t in range(seq_len - 1, len(X)):
                window = X[t - seq_len + 1: t + 1]
                if np.isnan(window).any():
                    continue
                x_t = torch.from_numpy(window).unsqueeze(0).to(device)
                s_t = torch.tensor([symbol_to_idx[sym]], dtype=torch.long).to(device)
                p = float(torch.sigmoid(model(x_t, s_t)).item())
                out_rows.append({
                    "symbol": sym,
                    "date": sub.iloc[t]["date"],
                    "pred_proba": p,
                })
    return pd.DataFrame(out_rows)


def save_lstm(result: LSTMTrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": result.model.state_dict(),
        "symbol_to_idx": result.symbol_to_idx,
        "n_features": len(FEATURE_COLS),
        "n_symbols": len(result.symbol_to_idx),
    }, path)


def load_lstm(path: Path) -> tuple[LSTMSignalModel, dict[str, int]]:
    ckpt = torch.load(path, weights_only=False)
    model = LSTMSignalModel(
        n_features=ckpt["n_features"],
        n_symbols=ckpt["n_symbols"],
    )
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["symbol_to_idx"]


__all__ = [
    "LSTMSignalModel", "SequenceDataset", "LSTMTrainResult",
    "train_lstm", "predict_lstm", "save_lstm", "load_lstm",
    "SEQ_LEN", "EMBED_DIM", "HIDDEN_DIM",
]
