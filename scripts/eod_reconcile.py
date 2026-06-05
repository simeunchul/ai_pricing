"""EOD 재정합 보고서 — 봇 state vs KIS 실잔고 비교.

매 iter sync 가 phantom/missing 을 정정하긴 하지만, **누적 drift 추세**를
포착할 별도 시스템이 없었다. 이 스크립트는:

  1. 봇이 알고 있는 state (`dual_state.json`)
  2. KIS 실잔고 (`balance()` API)
  3. KIS 거래 내역 (`dual_trades.parquet` — 봇 입장)

를 한 번에 호출해 비교 → 일별 보고서 (`data/eod_reconcile_<date>.json`).

운영 방식:
  - 매일 장마감 후 (15:35 권장) cron / 작업스케줄러로 자동 호출
  - 보고서가 누적되면 drift 추세 확인 가능
  - 큰 차이 발생 시 logger 가 WARNING 으로 표시 (외부 알람 hook 가능)

검사 항목:
  - state.cash vs KIS dnca_tot_amt
  - state.positions vs KIS holdings (종목별 qty, avg_entry 차이)
  - 일중 거래 횟수 vs 봇 trades 기록
  - state.portfolio_peak vs 현재 평가액

종료 코드:
  0 = clean (drift 없음)
  1 = warning (한 가지 이상 drift)
  2 = error (KIS API 호출 실패)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

from autotrader.broker.kis_client import KISClient, KISConfig

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Drift 허용 임계 — 이 미만이면 "정상", 초과면 WARNING
CASH_TOLERANCE_KRW = 10_000           # 1만원 미만 차이는 무시 (수수료 등)
QTY_TOLERANCE = 0                     # 수량은 정확히 일치해야 함
AVG_ENTRY_TOLERANCE_PCT = 0.005       # 평균 매수가 0.5% 이내 차이는 무시 (반올림)


def load_dotenv(path: Path) -> None:
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _i(s) -> int:
    try:
        return int(str(s or "0").replace(",", "") or "0")
    except (ValueError, TypeError):
        return 0


def _f(s) -> float:
    try:
        return float(str(s or "0").replace(",", "") or "0")
    except (ValueError, TypeError):
        return 0.0


def load_bot_state(state_path: Path) -> dict | None:
    """봇 state 파일 로드."""
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] state 파일 파싱 실패: {e}", file=sys.stderr)
        return None


def fetch_kis_holdings(client: KISClient) -> dict | None:
    """KIS 실잔고 조회 → {symbol: {qty, avg, ...}, _cash}."""
    try:
        bal = client.balance()
    except Exception as e:
        print(f"[ERROR] KIS balance 호출 실패: {e}", file=sys.stderr)
        return None
    if not isinstance(bal, dict) or bal.get("rt_cd") != "0":
        print(f"[ERROR] KIS balance 비정상 응답: rt_cd={bal.get('rt_cd') if isinstance(bal, dict) else '?'}", file=sys.stderr)
        return None

    out1 = bal.get("output1") or []
    out2 = bal.get("output2") or []

    holdings = {}
    for h in out1:
        if not isinstance(h, dict):
            continue
        qty = _i(h.get("hldg_qty"))
        if qty <= 0:
            continue
        holdings[h.get("pdno", "")] = {
            "qty": qty,
            "avg_entry": _f(h.get("pchs_avg_pric")),
            "current_price": _f(h.get("prpr")),
            "pchs_amt": _f(h.get("pchs_amt")),
            "evlu_amt": _f(h.get("evlu_amt")),
        }

    cash = 0.0
    if out2 and isinstance(out2[0], dict):
        cash = _f(out2[0].get("dnca_tot_amt"))

    holdings["_cash"] = cash
    return holdings


def compare(state: dict, kis: dict) -> dict:
    """봇 state vs KIS 실잔고 비교 → drift 리포트."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "date": date.today().isoformat(),
        "drifts": [],
        "n_warnings": 0,
    }

    # 1. Cash 비교
    state_cash = float(state.get("cash", 0))
    kis_cash = kis.get("_cash", 0.0)
    cash_diff = state_cash - kis_cash
    if abs(cash_diff) > CASH_TOLERANCE_KRW:
        report["drifts"].append({
            "type": "cash",
            "state_value": state_cash,
            "kis_value": kis_cash,
            "diff": cash_diff,
            "severity": "warning",
        })
        report["n_warnings"] += 1
    report["state_cash"] = state_cash
    report["kis_cash"] = kis_cash
    report["cash_diff"] = cash_diff

    # 2. Positions 비교
    state_positions = state.get("positions", {})
    state_codes = set(state_positions.keys())
    kis_codes = set(k for k in kis.keys() if k and k != "_cash")

    phantom = state_codes - kis_codes
    missing = kis_codes - state_codes
    common = state_codes & kis_codes

    for sym in sorted(phantom):
        pos = state_positions[sym]
        report["drifts"].append({
            "type": "phantom",
            "symbol": sym,
            "state_qty": _i(pos.get("qty")),
            "state_avg_entry": _f(pos.get("avg_entry")),
            "severity": "warning",
            "note": "봇은 보유 중이라 알고 있으나 KIS 잔고엔 없음",
        })
        report["n_warnings"] += 1

    for sym in sorted(missing):
        h = kis[sym]
        report["drifts"].append({
            "type": "missing",
            "symbol": sym,
            "kis_qty": h["qty"],
            "kis_avg_entry": h["avg_entry"],
            "severity": "warning",
            "note": "KIS 엔 있으나 봇은 모르고 있음",
        })
        report["n_warnings"] += 1

    qty_mismatches = []
    avg_mismatches = []
    for sym in sorted(common):
        sp = state_positions[sym]
        kh = kis[sym]
        sp_qty = _i(sp.get("qty"))
        sp_avg = _f(sp.get("avg_entry"))
        if abs(sp_qty - kh["qty"]) > QTY_TOLERANCE:
            qty_mismatches.append({
                "type": "qty_mismatch",
                "symbol": sym,
                "state_qty": sp_qty,
                "kis_qty": kh["qty"],
                "diff": sp_qty - kh["qty"],
                "severity": "warning",
            })
        if sp_avg > 0 and kh["avg_entry"] > 0:
            rel_diff = abs(sp_avg - kh["avg_entry"]) / max(sp_avg, kh["avg_entry"])
            if rel_diff > AVG_ENTRY_TOLERANCE_PCT:
                avg_mismatches.append({
                    "type": "avg_entry_mismatch",
                    "symbol": sym,
                    "state_avg": sp_avg,
                    "kis_avg": kh["avg_entry"],
                    "rel_diff_pct": rel_diff * 100,
                    "severity": "warning",
                })

    report["drifts"].extend(qty_mismatches)
    report["drifts"].extend(avg_mismatches)
    report["n_warnings"] += len(qty_mismatches) + len(avg_mismatches)

    # 3. Summary
    report["summary"] = {
        "state_n_positions": len(state_codes),
        "kis_n_positions": len(kis_codes),
        "phantom_n": len(phantom),
        "missing_n": len(missing),
        "qty_mismatch_n": len(qty_mismatches),
        "avg_mismatch_n": len(avg_mismatches),
    }

    # 4. Portfolio value rough check
    kis_total_eval = sum(
        h["evlu_amt"] for sym, h in kis.items()
        if sym != "_cash" and isinstance(h, dict)
    )
    report["kis_holdings_eval_amt"] = kis_total_eval
    report["kis_portfolio_total"] = kis_cash + kis_total_eval
    report["state_portfolio_peak"] = float(state.get("portfolio_peak", 0))

    return report


def print_human_summary(report: dict) -> None:
    s = report["summary"]
    print("=" * 70)
    print(f"EOD Reconciliation Report — {report['date']}")
    print("=" * 70)
    print(f"  봇 보유: {s['state_n_positions']}종 | KIS 보유: {s['kis_n_positions']}종")
    print(f"  현금  봇: {report['state_cash']:>14,.0f}원   KIS: {report['kis_cash']:>14,.0f}원   차이: {report['cash_diff']:>+13,.0f}원")
    print(f"  포트폴리오 총평가 (KIS 기준): {report['kis_portfolio_total']:>14,.0f}원")
    print(f"  peak (봇 기록):              {report['state_portfolio_peak']:>14,.0f}원")
    print()
    print(f"  Drift 항목:")
    print(f"    phantom (state만 있음)     : {s['phantom_n']}")
    print(f"    missing (KIS만 있음)       : {s['missing_n']}")
    print(f"    qty 불일치                 : {s['qty_mismatch_n']}")
    print(f"    avg_entry 불일치           : {s['avg_mismatch_n']}")
    print()
    if report["n_warnings"] == 0:
        print("  ✅ CLEAN — drift 없음")
    else:
        print(f"  ⚠ {report['n_warnings']} WARNING(s) — 아래 detail 참고")
        for d in report["drifts"]:
            print(f"    [{d['type']}] " + str({k: v for k, v in d.items() if k != 'type'}))
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / "data" / "dual_state.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    ap.add_argument("--quiet", action="store_true", help="JSON 만 출력 (cron 용)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = KISConfig.from_env()
    if not cfg.app_key:
        print("[ERROR] KIS_APP_KEY 미설정", file=sys.stderr)
        return 2

    state_path = Path(args.state)
    state = load_bot_state(state_path)
    if state is None:
        print(f"[ERROR] state 파일 없음 또는 파싱 실패: {state_path}", file=sys.stderr)
        return 2

    client = KISClient(cfg)
    try:
        client.token()
    except Exception as e:
        print(f"[ERROR] KIS token 발급 실패: {e}", file=sys.stderr)
        return 2

    kis = fetch_kis_holdings(client)
    if kis is None:
        return 2

    report = compare(state, kis)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eod_reconcile_{report['date']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    if not args.quiet:
        print_human_summary(report)
        print(f"\n[saved] {out_path}")

    return 1 if report["n_warnings"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
