"""Root conftest — tells pytest to treat each package's tests dir as an isolated rootdir.

Without this, pytest collides on duplicate `tests` module names across packages.
"""

import sys
from pathlib import Path

# Ensure each package src is on sys.path (editable install normally handles this,
# but this guards against CI installs without editable mode).
ROOT = Path(__file__).parent
for pkg in ("pricing", "ai_pricing", "ai_hedging", "experiments", "autotrader"):
    src = ROOT / "packages" / pkg / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
