"""Notebook W7 — All 5 methods comparison.

Run:
    python -m experiments.compare_all --out data
Then this notebook renders the CSVs.
"""

# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# %%
for name in ("all_methods_pricing.csv", "all_methods_hedging.csv"):
    p = Path("data") / name
    if p.exists():
        print(f"\n=== {name} ===")
        print(pd.read_csv(p).to_string(index=False))
    else:
        print(f"[missing] run compare_all first to produce {p}")
