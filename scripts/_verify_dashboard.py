"""Dashboard 자체 검증 — streamlit 없이 함수 직접 호출.

목적:
  1. add_vline / add_shape 에러 재발 안 하는지
  2. 표시되는 값들 (현재 자산, 보유 종목 등) 이 실제 KIS / state 와 일치하는지
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

# streamlit cache decorator 가 동작하려면 streamlit 가 import 되어야 함.
# 하지만 streamlit 컨텍스트 없이도 함수 자체는 호출 가능 (cache 만 비활성화 됨).
import streamlit as st


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ===== 1. Plot generation 검증 =====
section("1. Plot generation (add_vline/add_shape) — 에러 안 나는가")

import importlib.util
_spec = importlib.util.spec_from_file_location("dual_dashboard", ROOT / "scripts" / "dual_dashboard.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

import pandas as pd
import plotly.graph_objects as go

mt = dd.TRADING_LOG.stat().st_mtime
try:
    trades = dd.load_trades.__wrapped__(mt)   # cache 우회
except Exception:
    trades = dd.load_trades(mt)
try:
    mdd = dd.load_mdd_events.__wrapped__(mt)
except Exception:
    mdd = dd.load_mdd_events(mt)

print(f"  trades shape : {trades.shape}")
print(f"  mdd shape    : {mdd.shape}")
print(f"  mdd unique dates: {len(mdd['timestamp'].dt.date.unique()) if not mdd.empty else 0}")

# 실제 dashboard 와 똑같이 plot 생성
if trades.empty:
    print("  [skip] no trades")
else:
    daily = trades.copy()
    daily["date"] = daily["timestamp"].dt.date
    agg = daily.groupby(["date", "side"]).size().unstack(fill_value=0)
    for col in ("BUY", "SELL", "FORCE_SELL"):
        if col not in agg.columns:
            agg[col] = 0
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(x=agg.index, y=agg["BUY"], name="BUY"))
    fig_d.add_trace(go.Bar(x=agg.index, y=agg["SELL"], name="SELL"))
    fig_d.add_trace(go.Bar(x=agg.index, y=agg["FORCE_SELL"], name="FORCE_SELL"))

    if not mdd.empty:
        for d in mdd["timestamp"].dt.strftime("%Y-%m-%d").unique():
            try:
                fig_d.add_shape(
                    type="line", x0=d, x1=d, y0=0, y1=1,
                    yref="paper", xref="x",
                    line=dict(color="red", dash="dot", width=1),
                )
                fig_d.add_annotation(
                    x=d, y=1, yref="paper", xref="x",
                    text="MDD", showarrow=False,
                    font=dict(color="red", size=10),
                    yanchor="bottom",
                )
            except Exception as e:
                print(f"  ❌ FAIL on {d}: {type(e).__name__}: {e}")
                raise
    fig_d.update_layout(barmode="stack", height=240)
    # render 까지 강제 — plotly figure → json → 에러 발생 여부 최종 확인
    try:
        _ = fig_d.to_json()
        print(f"  ✓ figure to_json OK ({len(fig_d.layout.shapes or [])} shapes, "
              f"{len(fig_d.layout.annotations or [])} annotations)")
    except Exception as e:
        print(f"  ❌ to_json FAIL: {e}")
        raise


# ===== 2. State 파일 검증 =====
section("2. dual_state.json — 대시보드 'Drawdown' 표시 기준값")

state = json.loads(dd.STATE_PATH.read_text(encoding="utf-8"))
print(f"  initial_cash      : {state['initial_cash']:,.0f}원")
print(f"  cash (state 기록) : {state['cash']:,.0f}원")
print(f"  positions (state) : {len(state.get('positions', {}))}종 — {list(state.get('positions', {}).keys())}")
print(f"  portfolio_peak    : {state['portfolio_peak']:,.0f}원")
print(f"  cooldown_remaining: {state['cooldown_remaining']}")
print(f"  last_run          : {state['last_run']}")


# ===== 3. KIS balance 직접 호출 — 대시보드의 truth source =====
section("3. KIS balance — 대시보드의 '현재 자산' / '보유 종목' truth source")

import os
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from autotrader.broker.kis_client import KISClient, KISConfig

try:
    cfg = KISConfig.from_env()
    client = KISClient(cfg)
    client.token()
    b = client.balance()
    if b.get("rt_cd") == "0":
        out2 = b.get("output2", [{}])[0] if b.get("output2") else {}
        out1 = b.get("output1") or []

        def _i(s): return int(str(s or "0").replace(",", "") or "0")
        def _f(s): return float(str(s or "0").replace(",", "") or "0")

        tot_eval = _i(out2.get("tot_evlu_amt"))
        cash = _i(out2.get("dnca_tot_amt"))
        nxdy = _i(out2.get("nxdy_excc_amt"))
        d2 = _i(out2.get("prvs_rcdl_excc_amt"))
        bfdy = _i(out2.get("bfdy_tot_asst_evlu_amt"))
        scts = _i(out2.get("scts_evlu_amt"))

        print(f"  총평가금액 (tot_evlu_amt)  : {tot_eval:,}원  ← 대시보드 'Portfolio Total'")
        print(f"  전일 총자산 (bfdy_tot...)  : {bfdy:,}원      ← 일일 PnL 기준선")
        print(f"  당일 손익                  : {tot_eval-bfdy:+,}원  ({(tot_eval-bfdy)/bfdy*100:+.2f}% if bfdy>0)")
        print(f"  예수금 (dnca_tot_amt)      : {cash:,}원      ← 대시보드 '예수금'")
        print(f"  익일 정산 (nxdy)           : {nxdy:,}원")
        print(f"  D+2 정산 (prvs_rcdl)       : {d2:,}원")
        print(f"  T+2 후 가용 (cash+nxdy+d2) : {cash+nxdy+d2:,}원  ← 대시보드 'T+2 후 추정'")
        print(f"  유가증권 평가 (scts)       : {scts:,}원")
        print(f"  보유 종목 (output1 raw)    : {sum(1 for h in out1 if _i(h.get('hldg_qty'))!=0)}종")
        for h in out1:
            q = _i(h.get("hldg_qty"))
            if q == 0:
                continue
            print(f"    - {h.get('pdno')} {h.get('prdt_name','').strip()}: {q}주 "
                  f"@매입 {_f(h.get('pchs_avg_pric')):,.0f}원 → 현재 {_f(h.get('prpr')):,.0f}원 "
                  f"= {_i(h.get('evlu_amt')):,}원 ({_f(h.get('evlu_pfls_rt')):+.2f}%)")
    else:
        print(f"  ⚠ KIS balance rt_cd={b.get('rt_cd')} msg={b.get('msg1','')}")
        print(f"  → 장외 시간이면 정상 (vts 가 일부 응답을 거부)")
except Exception as e:
    print(f"  ❌ KIS balance 호출 실패: {type(e).__name__}: {e}")


# ===== 4. 매매 KPI 검증 =====
section("4. 매매 이력 KPI — load_trades 가 추출한 값")

if trades.empty:
    print("  no trades")
else:
    n_buy = (trades["side"] == "BUY").sum()
    n_sell = (trades["side"] == "SELL").sum()
    n_force = (trades["side"] == "FORCE_SELL").sum()
    print(f"  매수 {n_buy}건 / 매도 {n_sell}건 / 강제청산 {n_force}건")
    print(f"  총 거래대금     : {int(trades['value'].sum()):,}원")
    print(f"  MDD 발동 이벤트 : {len(mdd)}회")

    sell_mask = trades["side"].isin(["SELL", "FORCE_SELL"])
    sell_with_pnl = trades[sell_mask & trades["pnl_krw"].notna()].copy()
    if not sell_with_pnl.empty:
        pnl_total = float(sell_with_pnl["pnl_krw"].sum())
        n_win = int((sell_with_pnl["pnl_krw"] > 0).sum())
        n_lose = int((sell_with_pnl["pnl_krw"] < 0).sum())
        best = float(sell_with_pnl["pnl_pct"].max()) * 100
        worst = float(sell_with_pnl["pnl_pct"].min()) * 100
        print(f"  매도 PnL 합계   : {pnl_total:+,.0f}원")
        print(f"  평균 매도 수익률: {float(sell_with_pnl['pnl_pct'].mean())*100:+.2f}%")
        print(f"  승률            : {n_win}/{n_win+n_lose} = "
              f"{n_win/(n_win+n_lose)*100 if n_win+n_lose>0 else 0:.1f}%")
        print(f"  최고 / 최저     : {best:+.2f}% / {worst:+.2f}%")

print("\n[verify_dashboard] done.")
