"""dual_trading.log 파싱해서 dual_history.parquet 의 5/12~5/17 갭 backfill.

원인: MDD 분기에서 history append 가 빠져서 5/12 09:32 이후 row 없음.
근본 fix 는 이미 적용 (MDD 분기에 history append 추가) — 이 스크립트는 그 사이의
silent 누락 구간을 trading.log 의 portfolio 라인에서 재구성한다.

trading.log 형식:
  "→ portfolio {total:,}원 ({pct:+.2f}%), 보유 {n}종"
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "dual_trading.log"
HIST = ROOT / "data" / "dual_history.parquet"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


_TS = r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})"
_RE = re.compile(
    _TS + r"[^\n]*portfolio ([\d,]+)원 \([+\-][\d.]+%\), 보유 (\d+)종"
)


def main():
    if not LOG.exists():
        print(f"[err] log not found: {LOG}")
        return
    text = LOG.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in _RE.finditer(text):
        d, t, tot, n = m.groups()
        ts = f"{d}T{t}"
        rows.append({
            "timestamp": ts,
            "cash": None,
            "n_positions": int(n),
            "portfolio_total": int(tot.replace(",", "")),
            "portfolio_peak": None,
            "drawdown": None,
            "cooldown_remaining": 0,
            "n_buy_candidates": 0,
            "n_sell_candidates": 0,
            "mdd_event": False,
            "backfilled": True,
        })
    print(f"trading.log 에서 portfolio row {len(rows)}건 추출")
    if not rows:
        return

    import pandas as pd
    df_new = pd.DataFrame(rows)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

    if HIST.exists():
        df_old = pd.read_parquet(HIST)
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"])
        # 이미 있는 timestamp 는 skip — backfill 만 추가
        existing_ts = set(df_old["timestamp"])
        df_new = df_new[~df_new["timestamp"].isin(existing_ts)]
        print(f"기존 history {len(df_old)}건, 추가 {len(df_new)}건 backfill")
        # peak / drawdown 컬럼은 기존 행에서 cummax 흉내내야 정확. 단순화: forward fill 후 cummax.
        merged = pd.concat([df_old, df_new], ignore_index=True).sort_values("timestamp")
    else:
        merged = df_new

    # peak / drawdown 재계산 (정확 보장 — 누락 행 있어도 OK)
    merged["portfolio_peak"] = merged["portfolio_total"].cummax()
    merged["drawdown"] = (
        (merged["portfolio_total"] - merged["portfolio_peak"]) / merged["portfolio_peak"]
    ).fillna(0)
    merged = merged.reset_index(drop=True)
    merged.to_parquet(HIST)
    print(f"[ok] dual_history.parquet 총 {len(merged)} 건 (저장 완료)")
    print(f"  range: {merged['timestamp'].min()} ~ {merged['timestamp'].max()}")
    print(f"  peak: {int(merged['portfolio_peak'].max()):,}원")
    print(f"  최저: {int(merged['portfolio_total'].min()):,}원")


if __name__ == "__main__":
    main()
