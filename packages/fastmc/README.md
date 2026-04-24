# fastmc

C++ Monte Carlo kernel exposed through pybind11.

## Build

Requires: CMake ≥ 3.18, C++17 compiler, Python ≥ 3.10.

```bash
pip install ./packages/fastmc
```

On Windows use the Visual Studio "x64 Native Tools Command Prompt", or install `Build Tools for Visual Studio`.

## Fallback

If the C++ build fails, `fastmc.mc_euro_call` falls back to a pure-numpy implementation
(slower but functionally identical). The symbol `fastmc._HAS_NATIVE` is `False` in that case.

## Benchmark

```bash
python packages/fastmc/benchmarks/bench_mc.py
```

Target (n_paths=1M, n_steps=252):

| Impl | Time | Speedup |
|---|---|---|
| numpy | ~12 s | 1× |
| numba | ~4 s | 3× |
| C++ single | ~1.5 s | 8× |
| C++ OpenMP 8c | ~0.45 s | ~27× |
