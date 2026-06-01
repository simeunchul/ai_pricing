"""C++ fastmc production-scale benchmark.

기존 단일 옵션 1M paths (14.8x) 가 production 부하를 대표하지 못함.
실 데스크 시나리오: 야간 re-pricing 으로 보유 옵션 1k~100k 건 동시.

벤치 시나리오:
  N_opts ∈ {100, 1000, 10000}, n_paths=10000, n_steps=63 (3M 만기)
  - numpy (vectorized over options)
  - fastmc C++ (loop over options, OpenMP intra-option)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

import fastmc  # noqa: E402


def numpy_batch_mc(S0_arr, K_arr, T_arr, r=0.03, q=0.0, sigma=0.2,
                   n_paths=10_000, n_steps=63, seed=42):
    """Vectorized over options + paths (single big array)."""
    rng = np.random.default_rng(seed)
    N = len(S0_arr)
    dt_arr = T_arr / n_steps  # (N,)
    drift = (r - q - 0.5 * sigma ** 2) * dt_arr
    diff = sigma * np.sqrt(dt_arr)

    half = n_paths // 2
    Z = rng.standard_normal((half, n_steps))  # shared Z across options
    logS_anti_pos = (drift[:, None, None] * np.arange(1, n_steps + 1)[None, None, :]
                     + diff[:, None, None] * np.cumsum(Z, axis=1)[None, :, :]
                     )[:, :, -1]
    logS_anti_neg = (drift[:, None, None] * np.arange(1, n_steps + 1)[None, None, :]
                     - diff[:, None, None] * np.cumsum(Z, axis=1)[None, :, :]
                     )[:, :, -1]
    S_pos = S0_arr[:, None] * np.exp(logS_anti_pos)
    S_neg = S0_arr[:, None] * np.exp(logS_anti_neg)
    disc = np.exp(-r * T_arr)
    p = 0.5 * (np.maximum(S_pos - K_arr[:, None], 0)
               + np.maximum(S_neg - K_arr[:, None], 0)) * disc[:, None]
    return p.mean(axis=1)


def cpp_loop_mc(S0_arr, K_arr, T_arr, r=0.03, q=0.0, sigma=0.2,
                n_paths=10_000, n_steps=63, seed_base=42):
    """Loop over options, fastmc handles per-option OpenMP."""
    prices = np.empty(len(S0_arr))
    for i, (S0, K, T) in enumerate(zip(S0_arr, K_arr, T_arr)):
        res = fastmc.mc_euro_call(float(S0), float(K), float(T), r, q, sigma,
                                   n_paths=n_paths, n_steps=n_steps,
                                   seed=seed_base + i)
        prices[i] = res.price
    return prices


def main():
    print("=" * 75)
    print("  fastmc production-scale benchmark — N options × 10k paths × 63 steps")
    print(f"  C++ extension active: _HAS_NATIVE={fastmc._HAS_NATIVE}")
    print("=" * 75)

    rng = np.random.default_rng(0)
    rows = []

    for N in [100, 1000, 10_000]:
        S0_arr = rng.uniform(80, 120, size=N)
        K_arr = rng.uniform(80, 120, size=N)
        T_arr = rng.uniform(0.1, 1.0, size=N)
        n_paths = 10_000
        n_steps = 63

        print(f"\n[N={N:>5}] preparing inputs ...")

        if N <= 1000:
            print(f"  numpy   (vectorized over {N} opts × {n_paths} paths × {n_steps} steps) ...")
            t0 = time.perf_counter()
            p_np = numpy_batch_mc(S0_arr, K_arr, T_arr,
                                   n_paths=n_paths, n_steps=n_steps, seed=42)
            t_np = time.perf_counter() - t0
            print(f"    {t_np:>6.2f}s  ({t_np/N*1000:.2f} ms/opt)  mean price = {p_np.mean():.4f}")
        else:
            print(f"  numpy   skipped (memory) ...")
            t_np = None

        print(f"  C++     (loop over {N} opts, OpenMP intra-option) ...")
        t0 = time.perf_counter()
        p_cpp = cpp_loop_mc(S0_arr, K_arr, T_arr,
                              n_paths=n_paths, n_steps=n_steps, seed_base=42)
        t_cpp = time.perf_counter() - t0
        print(f"    {t_cpp:>6.2f}s  ({t_cpp/N*1000:.2f} ms/opt)  mean price = {p_cpp.mean():.4f}")

        if t_np is not None:
            speedup = t_np / t_cpp
            print(f"    speedup = {speedup:.2f}×")
        else:
            speedup = None

        rows.append({
            "N_options": N,
            "n_paths": n_paths,
            "n_steps": n_steps,
            "numpy_seconds": round(t_np, 3) if t_np else None,
            "cpp_seconds": round(t_cpp, 3),
            "speedup": round(speedup, 2) if speedup else None,
            "ms_per_opt_cpp": round(t_cpp / N * 1000, 3),
        })

    print("\n" + "=" * 75)
    print("  Summary")
    print("=" * 75)
    print(f"{'N_opts':>8}  {'numpy(s)':>10}  {'cpp(s)':>8}  {'speedup':>8}  {'cpp ms/opt':>12}")
    for r in rows:
        print(f"{r['N_options']:>8}  "
              f"{str(r['numpy_seconds'] or '—'):>10}  "
              f"{r['cpp_seconds']:>8.2f}  "
              f"{str(r['speedup'] or '—'):>8}  "
              f"{r['ms_per_opt_cpp']:>12.3f}")

    print(f"\n[Production extrapolation]")
    big = rows[-1]
    if big["N_options"] >= 1000:
        per_opt_ms = big["ms_per_opt_cpp"]
        for target in [10_000, 100_000]:
            est_min = target * per_opt_ms / 1000 / 60
            print(f"    N={target:>6} opts × 10k paths × 63 steps ≈ {est_min:.1f}분 (CPU 단일 머신, OpenMP)")

    out = ROOT / "data" / "bench_fastmc_scale.json"
    import json
    out.write_text(json.dumps({"rows": rows, "host": "Windows 11 / MSVC + OpenMP",
                                "_HAS_NATIVE": fastmc._HAS_NATIVE},
                                 indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
