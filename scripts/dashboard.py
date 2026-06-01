"""KIS Paper Trading 실시간 대시보드 — 다중 종목 + 탭 레이아웃.

매 5초 자동 새로고침.
- 상단: KIS 잔고 (always visible)
- 탭1 🌐 전체: 모든 종목 합산 / 비교 차트
- 탭2 🔍 종목별: 선택 종목의 시계열 + 매매 디테일

Usage:
  pip install streamlit plotly
  streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
REFRESH_SEC = 5
BALANCE_TTL_SEC = 30   # KIS API quota 보호: 잔고 30초 캐시

# load env + KIS path
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

SYMBOL_NAMES = {
    # KOSPI 200 추종 ETF (운영 중)
    "069500": "KODEX 200",
    "102110": "TIGER 200",
    "148020": "KBSTAR 200",
    "152100": "ARIRANG 200",
    "278530": "KODEX 200TR",
    "105190": "KINDEX 200",
    # 코스닥 / 섹터 ETF (운영 중)
    "229200": "KODEX 코스닥150",
    "091160": "KODEX 반도체",
    "305720": "KODEX 2차전지산업",
    "266420": "KODEX 헬스케어",
    # 인버스 ETF (합성 short 라우팅 대상)
    "252670": "KODEX 200 인버스",
    "251340": "KODEX 코스닥150 선물인버스",
    "252710": "KODEX 200선물인버스2X",
    # 기타 (참고용)
    "232080": "TIGER 코스피200 레버리지",
    "226490": "KODEX 코스피",
}


@st.cache_data(ttl=BALANCE_TTL_SEC)
def fetch_kis_balance():
    """KIS 잔고 조회 (30초 캐시)."""
    try:
        from autotrader.broker.kis_client import KISClient, KISConfig
        client = KISClient(KISConfig.from_env())
        b = client.balance()
        if b.get("rt_cd") != "0":
            return {"error": b.get("msg1", "?")}
        out2 = b.get("output2", [{}])[0] if b.get("output2") else {}
        out1 = b.get("output1", [])
        return {
            "cash": int(out2.get("dnca_tot_amt", 0)),
            "total_eval": int(out2.get("tot_evlu_amt", 0)),
            "pnl": int(out2.get("asst_icdc_amt", 0)),
            "pnl_rate": float(out2.get("asst_icdc_erng_rt", 0)) / 100,
            "buy_today": int(out2.get("thdt_buy_amt", 0)),
            "sell_today": int(out2.get("thdt_sll_amt", 0)),
            "fee_today": int(out2.get("thdt_tlex_amt", 0)),
            "holdings": [
                {
                    "code": h.get("pdno"),
                    "name": h.get("prdt_name", "").strip(),
                    "qty": int(h.get("hldg_qty", 0)),
                    "avg_price": float(h.get("pchs_avg_pric", 0)),
                    "cur_price": float(h.get("prpr", 0)),
                    "eval_amt": int(h.get("evlu_amt", 0)),
                    "pnl_amt": int(h.get("evlu_pfls_amt", 0)),
                    "pnl_pct": float(h.get("evlu_pfls_rt", 0)),
                }
                for h in out1
                if int(h.get("hldg_qty", 0)) != 0
            ],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


st.set_page_config(
    page_title="KIS Paper Trading · quant-lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 KIS Paper Trading — 실시간 대시보드")
st.caption(f"Auto-refresh every {REFRESH_SEC}s · 🌐 전체 / 🔍 종목별 탭으로 전환")

# -------------------------------------------------------------------------
# Sidebar — log file picker
# -------------------------------------------------------------------------
log_dir = ROOT / "data"
log_files = sorted(log_dir.glob("kis_trading_log_*.json"), reverse=True)
if not log_files:
    st.warning("No log files yet. Run: `python scripts/run_kis_paper_trading.py`")
    st.stop()

selected_log = st.sidebar.selectbox(
    "Log file",
    options=log_files,
    format_func=lambda p: p.name,
)
st.sidebar.caption(f"Path: `{selected_log}`")

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
try:
    raw = selected_log.read_text(encoding="utf-8")
    if not raw.strip():
        st.info("Log file is empty (runner just started)")
        st.stop()
    records = json.loads(raw)
except json.JSONDecodeError:
    st.info("Log file mid-write, retry next tick")
    time.sleep(REFRESH_SEC)
    st.rerun()
    st.stop()

if not records:
    st.info("No ticks yet")
    st.stop()

df_all = pd.DataFrame(records)
df_all["ts"] = pd.to_datetime(df_all["ts"])
df_all = df_all.sort_values("ts").reset_index(drop=True)

# Backward-compat: 단일 종목 로그는 symbol 필드가 없을 수 있음
if "symbol" not in df_all.columns:
    df_all["symbol"] = "069500"
    df_all["name"] = "KODEX 200"
df_all["symbol"] = df_all["symbol"].astype(str)

available_syms = sorted(df_all["symbol"].unique())

# 종목명 resolver: 1) 로그에 기록된 name 필드 우선 (runner 의 진실)
#                  2) SYMBOL_NAMES dict 폴백
#                  3) 그래도 없으면 종목코드 자체
NAME_FROM_LOG: dict[str, str] = {}
if "name" in df_all.columns:
    for sym in available_syms:
        sub = df_all[df_all["symbol"] == sym]
        names = sub["name"].dropna().unique()
        if len(names) > 0 and str(names[0]).strip():
            NAME_FROM_LOG[sym] = str(names[0])

def name_of(sym: str) -> str:
    return NAME_FROM_LOG.get(sym) or SYMBOL_NAMES.get(sym) or sym

def _label(s: str) -> str:
    return f"{s} — {name_of(s)}"

# -------------------------------------------------------------------------
# Sidebar — symbol selector (종목별 탭에서 사용)
# -------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 종목별 탭에서 표시할 종목**")
selected_sym = st.sidebar.radio(
    "차트 표시 종목",
    options=available_syms,
    format_func=_label,
    index=0,
    label_visibility="collapsed",
)

# -------------------------------------------------------------------------
# KIS 계좌 잔고 (always visible, above tabs)
# -------------------------------------------------------------------------
st.subheader("💰 KIS 모의투자 계좌")
bal = fetch_kis_balance()
if bal.get("error"):
    st.warning(f"잔고 조회 실패: {bal['error']}")
else:
    # 항등식 역산: 진짜 가용 현금 = 총평가 - 보유평가합 (음수 가능 = phantom debt)
    raw_cash = bal["cash"]
    total_eval = bal["total_eval"]
    invested = sum(h["eval_amt"] for h in bal["holdings"]) if bal["holdings"] else 0
    cash = total_eval - invested
    cash_drift = raw_cash - cash
    over_committed = cash < 0
    denom = max(1, abs(invested) + abs(cash)) if cash >= 0 else max(1, invested)
    inv_pct = invested / denom * 100
    cash_pct = cash / denom * 100
    pnl = bal["pnl"]
    pnl_color = "🟢" if pnl >= 0 else "🔴"

    if over_committed:
        st.error(
            f"⚠️ **PHANTOM MARGIN DEBT 감지** — 보유 평가금({invested:,}원) 이 "
            f"총자산({total_eval:,}원) 을 초과. KIS vts 가 가용 현금을 체크하지 않고 "
            f"매수를 받아준 상태 (실 prod 였으면 거절). "
            f"실제 가용 현금 = **{cash:+,}원**. 일부 매도하여 정리 필요."
        )

    b1, b2, b3 = st.columns([1.2, 1.2, 1])
    b1.metric("💼 ETF 투자 (평가금액)", f"{invested:,}원",
              delta=f"{inv_pct:.1f}% of 총자산", delta_color="off")
    b2.metric("💵 가용 현금 (예수금)", f"{cash:,}원",
              delta=f"{cash_pct:.1f}% of 총자산", delta_color="off")
    b3.metric("🏦 총자산 (평가)", f"{total_eval:,}원",
              delta=f"{pnl:+,}원 ({bal['pnl_rate']:+.4%})")

    # 시각 바
    bar_fig = go.Figure()
    if over_committed:
        bar_fig.add_trace(go.Bar(
            x=[invested], y=["자산"], orientation="h",
            name=f"ETF 투자 {invested:,}원", marker=dict(color="#2c5aa0"),
            text=f"{invested:,}원", textposition="inside", insidetextanchor="middle"))
        bar_fig.add_trace(go.Bar(
            x=[abs(cash)], y=["자산"], orientation="h",
            name=f"⚠ phantom debt {cash:,}원",
            marker=dict(color="#c0392b", pattern_shape="x"),
            text=f"{cash:,}원 (debt)", textposition="inside", insidetextanchor="middle"))
    else:
        bar_fig.add_trace(go.Bar(
            x=[invested], y=["자산"], orientation="h",
            name=f"ETF 투자 {invested:,}원", marker=dict(color="#2c5aa0"),
            text=f"{inv_pct:.1f}%", textposition="inside", insidetextanchor="middle"))
        bar_fig.add_trace(go.Bar(
            x=[cash], y=["자산"], orientation="h",
            name=f"가용 현금 {cash:,}원", marker=dict(color="#7cb342"),
            text=f"{cash_pct:.1f}%", textposition="inside", insidetextanchor="middle"))
    bar_fig.update_layout(barmode="stack", height=80,
                          margin=dict(t=10, b=10, l=20, r=20),
                          showlegend=True,
                          legend=dict(orientation="h", yanchor="top", y=-0.5),
                          xaxis=dict(showticklabels=False),
                          yaxis=dict(showticklabels=False))
    st.plotly_chart(bar_fig, use_container_width=True)

    b4, b5, b6 = st.columns(3)
    b4.metric("당일 매수", f"{bal['buy_today']:,}원")
    b5.metric("당일 매도", f"{bal['sell_today']:,}원")
    b6.metric("당일 수수료", f"{bal['fee_today']:,}원")

    if bal["holdings"]:
        st.markdown("**📦 보유 종목 (구성)**")
        st.dataframe(
            pd.DataFrame(bal["holdings"]).rename(columns={
                "code": "종목코드", "name": "종목명", "qty": "수량",
                "avg_price": "평균단가", "cur_price": "현재가",
                "eval_amt": "평가금액", "pnl_amt": "평가손익", "pnl_pct": "수익률 %",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption(f"보유 종목 없음 (모두 청산). {pnl_color} {bal['pnl_rate']:+.4%} 수익률")
    st.caption(f"⏱ 잔고 캐시 {BALANCE_TTL_SEC}초 갱신")

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab_overview, tab_detail = st.tabs(["🌐 전체 종목", "🔍 종목별 상세"])

# =========================================================================
# 🌐 전체 종목 탭
# =========================================================================
with tab_overview:
    all_trades = df_all[df_all["signal"].isin(["BUY", "SELL"])]
    n_records_all = len(df_all)
    ok_orders_all = sum(
        1 for r in df_all.to_dict("records")
        if isinstance(r.get("order_response"), dict)
        and r["order_response"].get("rt_cd") == "0"
    )
    fail_orders_all = sum(
        1 for r in df_all.to_dict("records")
        if isinstance(r.get("order_response"), dict)
        and r["order_response"].get("rt_cd") not in (None, "0")
    )

    # ── 헤더 metrics
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("운영 종목", f"{len(available_syms)}")
    o2.metric("총 ticks", f"{n_records_all:,}")
    o3.metric("총 매매", f"{len(all_trades):,}")
    o4.metric("주문 OK / FAIL", f"{ok_orders_all} / {fail_orders_all}")

    # ── 종목별 현황 table
    st.subheader("🎯 종목별 현황")
    summary_rows = []
    for sym in available_syms:
        sub = df_all[df_all["symbol"] == sym]
        if sub.empty:
            continue
        last_row = sub.iloc[-1]
        trade_sub = sub[sub["signal"].isin(["BUY", "SELL"])]
        n_buy = int((trade_sub["signal"] == "BUY").sum())
        n_sell = int((trade_sub["signal"] == "SELL").sum())
        etf_pos = int(last_row.get("etf_position", last_row.get("position", 0)) or 0)
        inv_pos = int(last_row.get("inverse_position", 0) or 0)
        summary_rows.append({
            "종목": sym,
            "이름": name_of(sym),
            "ETF": int(last_row["etf_price"]),
            "iNAV": int(last_row["inav"]),
            "dev(bps)": float(last_row["dev_bps"]),
            "ETF 포지션": etf_pos,
            "인버스 포지션": inv_pos,
            "BUY": n_buy,
            "SELL": n_sell,
            "Trades": n_buy + n_sell,
        })
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True, hide_index=True,
    )

    # ── 종목별 포지션 가치 시간 추이 (stacked area)
    st.subheader("📊 종목별 ETF 포지션 가치 (시간 추이)")
    df_for_area = df_all.copy()
    df_for_area["pos_value"] = (
        df_for_area.get("etf_position", df_for_area.get("position", 0))
        .astype(float).fillna(0) * df_for_area["etf_price"].astype(float)
    )
    if df_for_area["pos_value"].abs().sum() > 0:
        pivot = df_for_area.pivot_table(
            index="ts", columns="symbol", values="pos_value", aggfunc="last"
        ).ffill().fillna(0)
        fig_area = go.Figure()
        for sym in pivot.columns:
            fig_area.add_trace(go.Scatter(
                x=pivot.index, y=pivot[sym],
                mode="lines", stackgroup="one",
                name=f"{sym} {name_of(sym)}",
                hovertemplate="%{y:,.0f}원<extra>" + name_of(sym) + "</extra>",
            ))
        fig_area.update_layout(
            height=300, hovermode="x unified",
            margin=dict(t=20, b=20),
            yaxis_title="포지션 평가 (원)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.caption("아직 포지션 진입 없음 — 매매 발생 후 표시")

    # ── 전 종목 deviation overlay
    st.subheader("📉 종목별 Deviation 비교 (bps)")
    fig_dev = go.Figure()
    for sym in available_syms:
        sub = df_all[df_all["symbol"] == sym]
        fig_dev.add_trace(go.Scatter(
            x=sub["ts"], y=sub["dev_bps"],
            mode="lines", name=f"{sym} {name_of(sym)[:10]}",
            line=dict(width=1.2),
        ))
    fig_dev.add_hline(y=0, line_color="black", line_width=1)
    fig_dev.add_hline(y=1, line_dash="dash", line_color="green",
                      opacity=0.5, annotation_text="±1bps enter")
    fig_dev.add_hline(y=-1, line_dash="dash", line_color="green", opacity=0.5)
    fig_dev.update_layout(
        height=350, hovermode="x unified",
        margin=dict(t=20, b=20),
        yaxis_title="dev (bps)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    # ── 종목별 매매 빈도 막대
    st.subheader("📊 종목별 매매 빈도")
    if len(all_trades) > 0:
        trade_counts = (all_trades.groupby(["symbol", "signal"])
                        .size().unstack(fill_value=0).reset_index())
        x_labels = [f"{s}<br>{name_of(s)[:10]}" for s in trade_counts["symbol"]]
        fig_bar = go.Figure()
        if "BUY" in trade_counts.columns:
            fig_bar.add_trace(go.Bar(
                x=x_labels, y=trade_counts["BUY"],
                name="BUY", marker_color="#0b8043",
                text=trade_counts["BUY"], textposition="auto",
            ))
        if "SELL" in trade_counts.columns:
            fig_bar.add_trace(go.Bar(
                x=x_labels, y=trade_counts["SELL"],
                name="SELL", marker_color="#c0392b",
                text=trade_counts["SELL"], textposition="auto",
            ))
        fig_bar.update_layout(
            barmode="group", height=300, margin=dict(t=20, b=40),
            yaxis_title="매매 건수",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.caption("아직 매매 발생 없음")

    # ── 전 종목 매매 통합 table
    if len(all_trades) > 0:
        st.subheader(f"📋 전 종목 매매 통합 (총 {len(all_trades)}건, 최근 50)")
        recent_all = all_trades.tail(50).copy()
        recent_all["ts_str"] = recent_all["ts"].dt.strftime("%H:%M:%S")
        recent_all["이름"] = recent_all["symbol"].apply(name_of)
        recent_all["order_status"] = recent_all["order_response"].apply(
            lambda r: "✓" if isinstance(r, dict) and r.get("rt_cd") == "0"
            else ("✗ " + (r.get("msg1", "")[:30] if isinstance(r, dict) else "?"))
        )
        # qty 컬럼 (신스키마면 order_qty, 없으면 1로 가정)
        if "order_qty" in recent_all.columns:
            recent_all["qty"] = recent_all["order_qty"].fillna(1).astype(int)
        else:
            recent_all["qty"] = 1
        cols = ["ts_str", "symbol", "이름", "signal", "qty",
                "etf_price", "dev_bps", "order_status"]
        st.dataframe(
            recent_all[cols].iloc[::-1].rename(columns={
                "ts_str": "시간", "symbol": "종목", "signal": "신호",
                "qty": "수량", "etf_price": "ETF",
                "dev_bps": "dev(bps)", "order_status": "주문",
            }),
            use_container_width=True, height=400, hide_index=True,
        )

# =========================================================================
# 🔍 종목별 상세 탭
# =========================================================================
with tab_detail:
    df = df_all[df_all["symbol"] == selected_sym].reset_index(drop=True)

    if df.empty:
        st.info(f"[{selected_sym}] 기록 없음")
    else:
        last = df.iloc[-1]
        trade_df = df[df["signal"].isin(["BUY", "SELL"])]
        ok_orders = sum(
            1 for r in df.to_dict("records")
            if isinstance(r.get("order_response"), dict)
            and r["order_response"].get("rt_cd") == "0"
        )
        fail_orders = sum(
            1 for r in df.to_dict("records")
            if isinstance(r.get("order_response"), dict)
            and r["order_response"].get("rt_cd") not in (None, "0")
        )

        # ── per-symbol metrics
        st.subheader(f"📍 {_label(selected_sym)}")
        etf_pos = int(last.get("etf_position", last.get("position", 0)) or 0)
        inv_pos = int(last.get("inverse_position", 0) or 0)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Total Ticks", f"{len(df):,}")
        c2.metric("Latest ETF", f"{last['etf_price']:,.0f}")
        c3.metric("Latest iNAV", f"{last['inav']:,.0f}")
        c4.metric("Latest dev", f"{last['dev_bps']:+.2f} bps")
        c5.metric("ETF 포지션", etf_pos)
        c6.metric("인버스 포지션", inv_pos)
        c7.metric("주문 OK / FAIL", f"{ok_orders} / {fail_orders}")

        elapsed_sec = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds()
        st.caption(
            f"Session: {df['ts'].iloc[0].strftime('%H:%M:%S')} → "
            f"{df['ts'].iloc[-1].strftime('%H:%M:%S')}  ({elapsed_sec/60:.1f} min)"
        )

        # ── ETF + iNAV overlay
        st.subheader(f"📈 {_label(selected_sym)} — ETF vs iNAV")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["ts"], y=df["etf_price"], name="ETF",
            line=dict(color="#2c5aa0", width=2),
        ))
        fig1.add_trace(go.Scatter(
            x=df["ts"], y=df["inav"], name="iNAV (이론가)",
            line=dict(color="#a6820d", width=2, dash="dot"),
        ))
        buys = trade_df[trade_df["signal"] == "BUY"]
        sells = trade_df[trade_df["signal"] == "SELL"]
        if len(buys) > 0:
            fig1.add_trace(go.Scatter(
                x=buys["ts"], y=buys["etf_price"], mode="markers", name="BUY",
                marker=dict(color="#0b8043", size=12, symbol="triangle-up"),
            ))
        if len(sells) > 0:
            fig1.add_trace(go.Scatter(
                x=sells["ts"], y=sells["etf_price"], mode="markers", name="SELL",
                marker=dict(color="#c0392b", size=12, symbol="triangle-down"),
            ))
        fig1.update_layout(height=350, hovermode="x unified", margin=dict(t=30, b=20))
        st.plotly_chart(fig1, use_container_width=True)

        # ── Deviation chart
        st.subheader("📉 Deviation (bps) + 매매 임계")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["ts"], y=df["dev_bps"], name="dev (bps)",
            line=dict(color="#6c3483", width=2),
            fill="tozeroy", fillcolor="rgba(108, 52, 131, 0.1)",
        ))
        ENTER_BPS = 1.0
        EXIT_BPS = 0.2
        fig2.add_hline(y=ENTER_BPS, line_dash="dash", line_color="green",
                       annotation_text=f"+enter {ENTER_BPS} bps")
        fig2.add_hline(y=-ENTER_BPS, line_dash="dash", line_color="green",
                       annotation_text=f"-enter {ENTER_BPS} bps")
        fig2.add_hline(y=EXIT_BPS, line_dash="dot", line_color="gray", opacity=0.5)
        fig2.add_hline(y=-EXIT_BPS, line_dash="dot", line_color="gray", opacity=0.5)
        fig2.add_hline(y=0, line_color="black", line_width=1)
        if len(buys) > 0:
            fig2.add_trace(go.Scatter(
                x=buys["ts"], y=buys["dev_bps"], mode="markers", name="BUY",
                marker=dict(color="#0b8043", size=12, symbol="triangle-up"),
            ))
        if len(sells) > 0:
            fig2.add_trace(go.Scatter(
                x=sells["ts"], y=sells["dev_bps"], mode="markers", name="SELL",
                marker=dict(color="#c0392b", size=12, symbol="triangle-down"),
            ))
        fig2.update_layout(height=350, hovermode="x unified", margin=dict(t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        # ── Position chart (ETF + 인버스 둘 다)
        st.subheader("📦 Position (보유 주식 수) — ETF vs 인버스")
        fig3 = go.Figure()
        if "etf_position" in df.columns:
            fig3.add_trace(go.Scatter(
                x=df["ts"], y=df["etf_position"].fillna(0),
                line=dict(shape="hv", color="#2c5aa0", width=2),
                fill="tozeroy", fillcolor="rgba(44, 90, 160, 0.15)",
                name="ETF 포지션",
            ))
        else:
            fig3.add_trace(go.Scatter(
                x=df["ts"], y=df["position"],
                line=dict(shape="hv", color="#2c5aa0", width=2),
                fill="tozeroy", fillcolor="rgba(44, 90, 160, 0.15)",
                name="포지션",
            ))
        if "inverse_position" in df.columns:
            fig3.add_trace(go.Scatter(
                x=df["ts"], y=df["inverse_position"].fillna(0),
                line=dict(shape="hv", color="#c0392b", width=2),
                fill="tozeroy", fillcolor="rgba(192, 57, 43, 0.10)",
                name="인버스 포지션 (synthetic short)",
            ))
        fig3.add_hline(y=0, line_color="black", line_width=1)
        fig3.update_layout(
            height=240, margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── 매매 + 통계
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader(f"📋 [{selected_sym}] 최근 매매 (최대 30건)")
            if len(trade_df) > 0:
                recent = trade_df.tail(30).copy()
                recent["ts_str"] = recent["ts"].dt.strftime("%H:%M:%S")
                recent["order_status"] = recent["order_response"].apply(
                    lambda r: "✓" if isinstance(r, dict) and r.get("rt_cd") == "0"
                    else ("✗ " + (r.get("msg1", "")[:30] if isinstance(r, dict) else "?"))
                )
                if "order_qty" in recent.columns:
                    recent["qty"] = recent["order_qty"].fillna(1).astype(int)
                else:
                    recent["qty"] = 1
                pos_col = "etf_position" if "etf_position" in recent.columns else "position"
                st.dataframe(
                    recent[["ts_str", "signal", "qty", "etf_price",
                            "dev_bps", pos_col, "order_status"]]
                        .iloc[::-1]
                        .rename(columns={
                            "ts_str": "시간", "signal": "신호", "qty": "수량",
                            "etf_price": "ETF", "dev_bps": "dev(bps)",
                            pos_col: "포지션", "order_status": "주문",
                        }),
                    use_container_width=True, height=400, hide_index=True,
                )
            else:
                st.info(f"[{selected_sym}] 매매 발생 안 함 — 임계 (±{ENTER_BPS}bps) 도달 대기")

        with col_right:
            st.subheader("📊 통계 (선택 종목)")
            st.metric("Mean dev (bps)", f"{df['dev_bps'].mean():+.2f}")
            st.metric("Std dev (bps)", f"{df['dev_bps'].std():.2f}")
            st.metric("Max dev (bps)", f"{df['dev_bps'].max():+.2f}")
            st.metric("Min dev (bps)", f"{df['dev_bps'].min():+.2f}")
            if len(trade_df) > 0:
                n_buy = sum(trade_df["signal"] == "BUY")
                n_sell = sum(trade_df["signal"] == "SELL")
                st.metric("BUY / SELL", f"{n_buy} / {n_sell}")

# -------------------------------------------------------------------------
# Risk halts (after tabs)
# -------------------------------------------------------------------------
risk_halts = df_all[~df_all.get("risk_ok", True).fillna(True)]
if len(risk_halts) > 0:
    with st.expander(f"⚠️ Risk halts ({len(risk_halts)} records)"):
        st.dataframe(
            risk_halts[["ts", "symbol", "risk_reason"]].tail(20),
            use_container_width=True, hide_index=True,
        )

# -------------------------------------------------------------------------
# Auto refresh
# -------------------------------------------------------------------------
st.caption(f"⏱  다음 새로고침: {REFRESH_SEC}초 후")
time.sleep(REFRESH_SEC)
st.rerun()
