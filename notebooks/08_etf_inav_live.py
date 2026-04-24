"""Notebook W8 — ETF iNAV live runner review."""

# %%
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

log_path = Path("data/runner_log.json")
if not log_path.exists():
    print("Run first: python -m autotrader.runner --dry --seconds 60")
else:
    df = pd.DataFrame(json.loads(log_path.read_text(encoding="utf-8")))
    df["ts"] = pd.to_datetime(df["ts"])
    print(df.describe())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["ts"], df["dev_bps"], "-o")
    ax.set_ylabel("Deviation (bps)")
    ax.set_title("ETF vs iNAV deviation")
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("inav_deviation.png")
