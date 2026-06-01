"""Dual confirmation 전략 vs 단순 벤치마크 비교 분석.

사용자 가설: "국장에 알파 없음 — 삼전+하닉만 사면 장땡"
이를 정직하게 검증한다.

비교 대상 (운영 기간 2026-04-30 ~ 2026-05-26):
  1. dual 전략 실제 운영 곡선 (history parquet + 일별 로그 파싱)
  2. 삼성전자 100% buy-and-hold
  3. SK하이닉스 100% buy-and-hold
  4. 삼전 50% + 하닉 50%
  5. 코스피 종합지수 (1001)

지표: 총수익률, MDD, 일수익률 변동성(연율화), Sharpe(rf=0), Calmar.

데이터 소스:
  - 운영 곡선: data/dual_history.parquet (4/30~5/18) + data/dual_trading.log* (5/19~5/26)
  - 벤치마크: pykrx (KRX 공식 일봉)
"""

from __future__ import annotations

import glob
import json
import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

START = "20260430"
END = "20260526"


# ---------------------------------------------------------------
# 1. 전략 운영 곡선
# ---------------------------------------------------------------
def build_strategy_curve() -> pd.Series:
    """일자별 portfolio_total (KRW). history parquet + 로그 파싱 병합."""
    points: dict[date, float] = {}

    # (a) history parquet — 4/30 ~ 5/18
    p = DATA / "dual_history.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["timestamp"])
        df["d"] = df["ts"].dt.date
        for d, v in df.groupby("d")["portfolio_total"].last().items():
            points[d] = float(v)

    # (b) 일별 로그 — 5/19 ~ 5/26 (history append 가 timestamp 버그로 끊긴 구간)
    pat = re.compile(r"portfolio\s+([0-9,]+)\s*원")
    for logf in glob.glob(str(DATA / "dual_trading.log*")):
        name = Path(logf).name
        # 파일명에서 날짜 추출 (dual_trading.log.2026-05-25), 본문 last 값 사용
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
        try:
            text = Path(logf).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        vals = pat.findall(text)
        if not vals:
            continue
        last_val = float(vals[-1].replace(",", ""))
        if m:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        else:
            # 회전 안 된 현재 로그 (dual_trading.log) → 마지막 timestamp 의 날짜
            tsm = re.findall(r"(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}", text)
            if not tsm:
                continue
            d = date.fromisoformat(tsm[-1])
        # 로그 값이 있으면 우선 (history 보다 최신/정확)
        points[d] = last_val

    s = pd.Series(points).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


# ---------------------------------------------------------------
# 2. 벤치마크 곡선
# ---------------------------------------------------------------
def fetch_benchmarks() -> pd.DataFrame:
    from pykrx import stock

    sam = stock.get_market_ohlcv(START, END, "005930")["종가"]
    hyx = stock.get_market_ohlcv(START, END, "000660")["종가"]
    # 코스피 지수 직접 조회(get_index_ohlcv)는 KRX 서버 응답 파싱 불안정 →
    # KODEX 200 ETF(069500) 를 코스피 프록시로 사용 (코스피200 추종).
    kospi = stock.get_market_ohlcv(START, END, "069500")["종가"]

    df = pd.DataFrame({
        "삼성전자": sam,
        "SK하이닉스": hyx,
        "코스피(KODEX200)": kospi,
    })
    df.index = pd.to_datetime(df.index)
    # 삼전 50 + 하닉 50 (정규화 후 평균)
    norm = df[["삼성전자", "SK하이닉스"]] / df[["삼성전자", "SK하이닉스"]].iloc[0]
    df["삼전50+하닉50"] = norm.mean(axis=1) * df["삼성전자"].iloc[0]
    return df


# ---------------------------------------------------------------
# 3. 지표
# ---------------------------------------------------------------
def metrics(curve: pd.Series) -> dict:
    curve = curve.dropna()
    if len(curve) < 2:
        return {}
    total_ret = curve.iloc[-1] / curve.iloc[0] - 1.0

    # MDD
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    mdd = dd.min()

    # 일수익률
    rets = curve.pct_change().dropna()
    n_days = len(curve)
    ann_factor = 252
    vol_daily = rets.std()
    vol_ann = vol_daily * np.sqrt(ann_factor)

    # 연율화 수익률 (기간 길이 기반 CAGR)
    span_days = max(1, (curve.index[-1] - curve.index[0]).days)
    cagr = (1 + total_ret) ** (365.0 / span_days) - 1.0

    sharpe = (cagr / vol_ann) if vol_ann > 0 else float("nan")
    calmar = (cagr / abs(mdd)) if mdd < 0 else float("nan")

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_ann_pct": round(vol_ann * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe == sharpe else None,
        "calmar": round(calmar, 2) if calmar == calmar else None,
        "n_points": n_days,
    }


def main():
    strat = build_strategy_curve()
    bench = fetch_benchmarks()

    print("=" * 60)
    print("전략 운영 곡선 (일자별 portfolio, KRW)")
    print("=" * 60)
    print(strat.map(lambda x: f"{x:,.0f}").to_string())

    # 전략을 1 기준으로 정규화하여 벤치마크와 동일 스케일 비교
    curves = {"dual전략": strat}
    for col in bench.columns:
        curves[col] = bench[col]

    print("\n" + "=" * 60)
    print("지표 비교 (운영 기간 4/30~5/26)")
    print("=" * 60)
    rows = {}
    for name, c in curves.items():
        rows[name] = metrics(c)
    result = pd.DataFrame(rows).T
    print(result.to_string())

    # 같은 날짜 포인트로 전략 vs 벤치 누적수익 비교 (전략 곡선 날짜에 맞춰 reindex)
    aligned = pd.DataFrame(index=strat.index)
    aligned["dual전략"] = strat
    for col in bench.columns:
        aligned[col] = bench[col].reindex(strat.index, method="ffill")
    norm = aligned / aligned.iloc[0]
    print("\n" + "=" * 60)
    print("누적 수익 곡선 (시작=1.00, 전략 날짜 기준 정렬)")
    print("=" * 60)
    print(norm.round(3).to_string())

    out = {
        "metrics": {k: v for k, v in rows.items()},
        "normalized_curve": norm.round(4).reset_index().rename(
            columns={"index": "date"}
        ).assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
    }
    outpath = DATA / "strategy_vs_benchmark.json"
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {outpath}")


if __name__ == "__main__":
    main()
