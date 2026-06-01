"""외국인 순매수 추종 전략 일별 backtest.

Long-only (공매도 불가). 사후정보 누설 회피를 위해:
  - T일 EOD 외국인 데이터 관측 → T+1일 *시가* 진입
  - 청산도 시가 기준
  - 거래비용 (cost_bps) 은 entry + exit 모두에 적용 (왕복)

가설 ("외국인 따라가면 돈 번다") 검증을 위해 각 종목별 매수후보유 (buy &
hold) 수익률을 baseline 으로 함께 계산한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from autotrader.data.krx_investor import (
    fetch_combined, add_flow_features, InvestorFlowCache,
)
from autotrader.strategies.foreign_flow import ForeignFlowFollow
from autotrader.strategies.etf_inav_arb import Signal


@dataclass
class ForeignFlowBacktestConfig:
    symbols: list[str]
    enter_threshold: float = 0.05
    exit_after_days: int = 2
    qty_per_signal: int = 10
    max_position_per_symbol: int = 50
    cost_bps: float = 25.0                   # 왕복 거래비용 (basis points)
    max_pages: int = 60                      # Naver 페이지네이션 깊이
    use_cache: bool = True
    cache_dir: Path | None = None
    start: str | None = None                 # YYYY-MM-DD (None = 데이터 시작점)
    end: str | None = None                   # YYYY-MM-DD (None = 데이터 끝점)


def _load_symbol_data(symbol: str, cfg: ForeignFlowBacktestConfig) -> pd.DataFrame:
    cache = InvestorFlowCache(cache_dir=cfg.cache_dir) if cfg.cache_dir else None
    df = fetch_combined(
        symbol,
        max_pages=cfg.max_pages,
        use_cache=cfg.use_cache,
        cache=cache,
    )
    if df.empty:
        return df
    df = add_flow_features(df)
    if cfg.start:
        df = df[df.index >= pd.Timestamp(cfg.start)]
    if cfg.end:
        df = df[df.index <= pd.Timestamp(cfg.end)]
    return df


def _common_date_range(sym_data: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """모든 종목 공통 거래 가능 기간 [max(starts), min(ends)] 반환.

    종목별 데이터 시작/종료일이 다를 때 portfolio sum 이 점프하지 않도록
    공통 구간만 사용하기 위함.
    """
    if not sym_data:
        return None
    starts = [df.index.min() for df in sym_data.values() if not df.empty]
    ends = [df.index.max() for df in sym_data.values() if not df.empty]
    if not starts or not ends:
        return None
    return max(starts), min(ends)


def run_foreign_flow_backtest(
    cfg: ForeignFlowBacktestConfig,
    strategy=None,
) -> pd.DataFrame:
    """심볼별 일별 backtest 실행. 거래 단위 DataFrame 반환.

    Args:
        cfg: 백테스트 config
        strategy: duck-typed 전략 객체 (decide / apply / advance_day 인터페이스).
                  None 이면 cfg 기반 ForeignFlowFollow 인스턴스 생성.

    Returns DataFrame with columns:
        date_signal, date_entry, date_exit, symbol,
        entry_open, exit_open, hold_days,
        gross_return, net_return, qty_avg
    """
    cost_per_side = cfg.cost_bps / 10_000.0 / 2.0   # 왕복 → 한 쪽 비용
    trades: list[dict] = []

    # 종목별 데이터 로드
    sym_data: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df

    if not sym_data:
        return pd.DataFrame()

    # 모든 종목의 거래일을 union 해서 시계열 진행
    rng = _common_date_range(sym_data)
    if rng is None:
        return pd.DataFrame()
    common_start, common_end = rng
    all_dates = sorted([
        d for d in set().union(*[set(df.index) for df in sym_data.values()])
        if common_start <= d <= common_end
    ])
    if strategy is not None:
        strat = strategy
    else:
        strat = ForeignFlowFollow(
            enter_threshold=cfg.enter_threshold,
            exit_after_days=cfg.exit_after_days,
            qty_per_signal=cfg.qty_per_signal,
            max_position_per_symbol=cfg.max_position_per_symbol,
        )

    # 진입 가격 추적 (FIFO 단순화: 평균가 1개)
    entry_state: dict[str, dict] = {}    # symbol -> {avg_entry, qty, entry_date}

    for i, today in enumerate(all_dates):
        # 1) 보유 중인 포지션 → 만기/매도 신호 청산을 먼저 결정
        # 2) 새 매수 신호는 같은 날 EOD 관측 후 T+1 시가 진입을 의미
        # 따라서 today 의 행위는: T-1 의 신호 → T(=today) 시가 매수/매도

        # 이전일 데이터 기준 신호 → today 시가에 실행
        if i == 0:
            strat.advance_day()
            continue
        prev = all_dates[i - 1]

        for sym, df in sym_data.items():
            if today not in df.index or prev not in df.index:
                continue
            prev_row = df.loc[prev]
            today_row = df.loc[today]
            today_open = float(today_row["open"])
            if today_open <= 0:
                continue

            flow_ratio_prev = float(prev_row["flow_ratio"])
            sig, qty = strat.decide(sym, flow_ratio_prev)

            if sig == Signal.BUY and qty > 0:
                # 진입
                state = entry_state.get(sym, {"avg_entry": 0.0, "qty": 0, "entry_date": today})
                old_qty = state["qty"]
                old_cost = state["avg_entry"] * old_qty
                new_qty = old_qty + qty
                new_avg = (old_cost + today_open * qty) / new_qty if new_qty > 0 else 0.0
                entry_state[sym] = {
                    "avg_entry": new_avg,
                    "qty": new_qty,
                    "entry_date": state.get("entry_date") if old_qty > 0 else today,
                }
                strat.apply(sym, sig, qty)
            elif sig == Signal.SELL and qty > 0:
                # 청산 (부분 청산 지원: sell_qty <= state['qty'])
                state = entry_state.get(sym)
                if state is None or state["qty"] == 0:
                    continue
                sell_qty = min(qty, state["qty"])
                avg_entry = state["avg_entry"]
                hold_days = (today - state["entry_date"]).days
                gross = (today_open - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                net = gross - 2 * cost_per_side  # 왕복 비용 (부분 청산도 동일 비율)
                trades.append({
                    "date_signal": prev,
                    "date_entry": state["entry_date"],
                    "date_exit": today,
                    "symbol": sym,
                    "entry_open": avg_entry,
                    "exit_open": today_open,
                    "hold_days": hold_days,
                    "qty": sell_qty,
                    "gross_return": gross,
                    "net_return": net,
                })
                strat.apply(sym, sig, sell_qty)
                state["qty"] -= sell_qty
                if state["qty"] <= 0:
                    entry_state.pop(sym, None)
                # else: avg_entry / entry_date 유지 (다음 부분청산 또는 재매수 대비)

        strat.advance_day()

    return pd.DataFrame(trades)


def buy_and_hold_baseline(cfg: ForeignFlowBacktestConfig) -> dict[str, float]:
    """종목별 시작-끝 매수후보유 수익률 (등가중 평균 baseline 산출용)."""
    out: dict[str, float] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if df.empty or len(df) < 2:
            continue
        first_open = float(df["open"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        if first_open <= 0:
            continue
        out[sym] = (last_close - first_open) / first_open - cfg.cost_bps / 10_000.0
    return out


def summarize_trades(trades: pd.DataFrame, baseline: dict[str, float] | None = None) -> dict:
    """집계 통계: 승률, 평균 P&L, Sharpe, 종목별 분해, 매수후보유 대비 초과수익."""
    if trades.empty:
        return {"n": 0, "note": "no trades"}

    n = len(trades)
    hit = float((trades["net_return"] > 0).mean())
    mean = float(trades["net_return"].mean())
    std = float(trades["net_return"].std())
    sharpe_pertrade = mean / std if std > 0 else float("nan")
    # 252 trading days 기준 거래당 Sharpe → 연 환산 (가정: 종목당 ~50 trades/year)
    avg_hold = float(trades["hold_days"].mean()) if "hold_days" in trades.columns else 1.0
    trades_per_year_per_sym = 252 / max(avg_hold, 1)
    ann_factor = math.sqrt(trades_per_year_per_sym)
    sharpe_ann = sharpe_pertrade * ann_factor if not math.isnan(sharpe_pertrade) else float("nan")

    by_symbol = (
        trades.groupby("symbol")
              .agg(n=("net_return", "count"),
                   hit=("net_return", lambda s: (s > 0).mean()),
                   mean=("net_return", "mean"),
                   total=("net_return", "sum"),
                   avg_hold=("hold_days", "mean"))
              .reset_index()
    )

    summary = {
        "n_trades": n,
        "hit_rate": hit,
        "mean_return": mean,
        "std_return": std,
        "sharpe_per_trade": sharpe_pertrade,
        "sharpe_annualized": sharpe_ann,
        "total_return_sum": float(trades["net_return"].sum()),
        "best_trade": float(trades["net_return"].max()),
        "worst_trade": float(trades["net_return"].min()),
        "avg_hold_days": avg_hold,
        "by_symbol": by_symbol.to_dict("records"),
    }

    if baseline:
        bh_mean = sum(baseline.values()) / len(baseline)
        summary["buyhold_mean_return"] = float(bh_mean)
        summary["buyhold_per_symbol"] = baseline
        # 전략의 종목별 누적 수익률 vs 매수후보유
        sym_total = trades.groupby("symbol")["net_return"].sum().to_dict()
        excess = {s: sym_total.get(s, 0) - baseline.get(s, 0) for s in baseline}
        summary["excess_vs_buyhold_per_symbol"] = excess
        summary["excess_vs_buyhold_mean"] = float(sum(excess.values()) / len(excess))

    return summary


@dataclass
class PortfolioBacktestResult:
    """Portfolio-level backtest 결과.

    history: index=date, columns=MultiIndex[(symbol, field)] where field ∈
             {cash, position, value}. value = cash + position × close.
    summary: 종목별 final_value / return / max_drawdown.
    overall: 전체 portfolio (5종 합) final_value / return / vs buy-hold.
    """
    history: pd.DataFrame
    summary: pd.DataFrame
    overall: dict


def run_portfolio_backtest(
    cfg: ForeignFlowBacktestConfig,
    strategy=None,
    initial_cash: float = 10_000_000.0,
    stop_loss_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    mdd_cap: float | None = None,
    vol_target_daily: float | None = None,
    vol_lookback: int = 20,
    cooldown_days: int = 20,
    signal_keys: tuple[str, ...] = ("flow_ratio",),
) -> PortfolioBacktestResult:
    """Portfolio-level backtest — 진짜 자본 흐름 시뮬레이션.

    종목당 동일 cash quota (initial_cash / N) 로 시작해서, 매수 시 cash 차감,
    매도 시 cash 회수, 매일 종가로 portfolio value 마크.

    Args:
        stop_loss_pct:    종목별 절대 손절 비율 (avg_entry 대비). None 이면 비활성.
                          예: 0.10 → 진입가 -10% 도달 시 다음날 시가 청산.
        trailing_stop_pct: 종목별 추적 손절 비율 (peak_close 대비). None 이면 비활성.
                          예: 0.07 → 진입 후 고점 대비 -7% 떨어지면 다음날 청산.
                          stop_loss 와 OR — 어느 한쪽이라도 hit 되면 청산.
        mdd_cap:          Portfolio drawdown 안전망. 전체 자본 -X% 도달 시 모든
                          포지션 강제 청산 + cooldown_days 일 동안 신규 진입 차단.
                          None 이면 비활성.
        vol_target_daily: 변동성 기반 사이징 목표 일별 std. None 이면 비활성.
        vol_lookback:     변동성 측정 lookback 일수 (default 20).
        cooldown_days:    MDD cap 발동 후 신규 진입 차단 일수 (default 20).

    매수 수량은 strategy 가 반환하는 target qty 와 cash 한도 중 작은 값.
    부분 매도 지원 (strategy.decide() qty 만큼만 매도).
    """
    cost_per_side = cfg.cost_bps / 10_000.0 / 2.0

    # 데이터 로드
    sym_data: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df
    if not sym_data:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    n_sym = len(sym_data)
    quota = initial_cash / n_sym
    rng = _common_date_range(sym_data)
    if rng is None:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})
    common_start, common_end = rng
    all_dates = sorted([
        d for d in set().union(*[set(df.index) for df in sym_data.values()])
        if common_start <= d <= common_end
    ])

    if strategy is not None:
        strat = strategy
    else:
        strat = ForeignFlowFollow(
            enter_threshold=cfg.enter_threshold,
            exit_after_days=cfg.exit_after_days,
            qty_per_signal=cfg.qty_per_signal,
            max_position_per_symbol=cfg.max_position_per_symbol,
        )

    # 종목별 자본/포지션 상태
    # avg_entry: stop loss 기준가 (가중평균 매수가)
    # peak_close: 진입 후 종가 고점 (trailing stop 기준)
    # stop_armed / trailing_armed: 다음날 시가 청산 트리거
    state: dict[str, dict] = {
        sym: {
            "cash": quota, "position": 0,
            "avg_entry": 0.0, "peak_close": 0.0,
            "stop_armed": False, "trailing_armed": False,
        }
        for sym in sym_data
    }

    # MDD cap state — 발동 시 cooldown 카운터 양수, 매일 -1
    mdd_cooldown_remaining = 0
    mdd_armed = False    # 직전일 portfolio drawdown 이 cap 도달 → 다음날 청산

    # 시계열 history 기록
    history_rows: list[dict] = []

    for i, today in enumerate(all_dates):
        if i == 0:
            # 첫 날: 시그널 없음, 초기 상태 기록
            for sym, df in sym_data.items():
                if today in df.index:
                    close = float(df.loc[today, "close"])
                    val = state[sym]["cash"] + state[sym]["position"] * close
                    history_rows.append({
                        "date": today, "symbol": sym,
                        "cash": state[sym]["cash"],
                        "position": state[sym]["position"],
                        "value": val,
                    })
            strat.advance_day()
            continue

        prev = all_dates[i - 1]

        # ===== MDD Cap 발동 (전일 drawdown 이 cap 도달했다면 오늘 시가에 전 종목 청산) =====
        if mdd_armed:
            for sym, df in sym_data.items():
                if today not in df.index:
                    continue
                today_open_s = float(df.loc[today, "open"])
                pos_s = state[sym]["position"]
                if pos_s > 0 and today_open_s > 0:
                    proceeds = pos_s * today_open_s * (1 - cost_per_side)
                    state[sym]["cash"] += proceeds
                    state[sym]["position"] = 0
                    state[sym]["avg_entry"] = 0.0
                    state[sym]["peak_close"] = 0.0
                    state[sym]["stop_armed"] = False
                    state[sym]["trailing_armed"] = False
                    strat.apply(sym, Signal.SELL, pos_s)
            mdd_armed = False
            mdd_cooldown_remaining = cooldown_days

        for sym, df in sym_data.items():
            if today not in df.index or prev not in df.index:
                continue
            prev_row = df.loc[prev]
            today_row = df.loc[today]
            today_open = float(today_row["open"])
            today_close = float(today_row["close"])
            if today_open <= 0 or today_close <= 0:
                continue

            # Strategy signal extraction (multi-signal 지원)
            sig_values = tuple(float(prev_row[k]) for k in signal_keys)
            cash = state[sym]["cash"]
            pos = state[sym]["position"]
            avg_entry = state[sym]["avg_entry"]

            # ===== Stop loss / Trailing stop 우선 처리 (OR — 어느 하나라도 hit) =====
            stop_triggered = False
            armed = False
            if stop_loss_pct is not None and pos > 0 and state[sym].get("stop_armed"):
                armed = True
            if trailing_stop_pct is not None and pos > 0 and state[sym].get("trailing_armed"):
                armed = True
            if armed:
                proceeds = pos * today_open * (1 - cost_per_side)
                state[sym]["cash"] = cash + proceeds
                state[sym]["position"] = 0
                state[sym]["avg_entry"] = 0.0
                state[sym]["peak_close"] = 0.0
                state[sym]["stop_armed"] = False
                state[sym]["trailing_armed"] = False
                strat.apply(sym, Signal.SELL, pos)
                stop_triggered = True
                cash = state[sym]["cash"]
                pos = 0

            if not stop_triggered and mdd_cooldown_remaining <= 0:
                sig, qty = strat.decide(sym, *sig_values)

                if sig == Signal.BUY and qty > 0:
                    # 변동성 기반 사이징 (옵션)
                    if vol_target_daily is not None and i >= vol_lookback:
                        # i는 dates 인덱스가 아니므로 df 자체에서 lookback 계산
                        loc = df.index.get_loc(today)
                        if loc >= vol_lookback:
                            window = df["close"].iloc[loc - vol_lookback:loc]
                            rets = window.pct_change().dropna()
                            if len(rets) >= 5 and rets.std() > 0:
                                vol = float(rets.std())
                                scale = max(0.2, min(2.0, vol_target_daily / vol))
                                qty = max(1, round(qty * scale))

                    px_with_cost = today_open * (1 + cost_per_side)
                    affordable = int(cash // px_with_cost)
                    actual = min(qty, affordable)
                    if actual > 0:
                        cost_total = actual * today_open * (1 + cost_per_side)
                        old_cost = avg_entry * pos
                        new_pos = pos + actual
                        new_avg = (old_cost + actual * today_open * (1 + cost_per_side)) / new_pos
                        state[sym]["cash"] = cash - cost_total
                        state[sym]["position"] = new_pos
                        state[sym]["avg_entry"] = new_avg
                        # 첫 진입 (이전 pos == 0) 시 peak_close 초기화 = 오늘 종가
                        if pos == 0:
                            state[sym]["peak_close"] = today_close
                        strat.apply(sym, sig, actual)
                elif sig == Signal.SELL and qty > 0:
                    actual = min(qty, pos)
                    if actual > 0:
                        proceeds = actual * today_open * (1 - cost_per_side)
                        state[sym]["cash"] = cash + proceeds
                        state[sym]["position"] = pos - actual
                        if state[sym]["position"] == 0:
                            state[sym]["avg_entry"] = 0.0
                            state[sym]["peak_close"] = 0.0
                        strat.apply(sym, sig, actual)

            # peak_close 갱신 (보유 중) + trailing/stop armed 결정
            if state[sym]["position"] > 0:
                state[sym]["peak_close"] = max(
                    state[sym]["peak_close"], today_close
                )

                # stop_armed (avg_entry 기준 절대 손절)
                if stop_loss_pct is not None and state[sym]["avg_entry"] > 0:
                    stop_price = state[sym]["avg_entry"] * (1 - stop_loss_pct)
                    state[sym]["stop_armed"] = today_close < stop_price
                else:
                    state[sym]["stop_armed"] = False

                # trailing_armed (peak_close 기준 추적 손절)
                if trailing_stop_pct is not None and state[sym]["peak_close"] > 0:
                    trail_price = state[sym]["peak_close"] * (1 - trailing_stop_pct)
                    state[sym]["trailing_armed"] = today_close < trail_price
                else:
                    state[sym]["trailing_armed"] = False
            else:
                state[sym]["stop_armed"] = False
                state[sym]["trailing_armed"] = False
                state[sym]["peak_close"] = 0.0

            val = state[sym]["cash"] + state[sym]["position"] * today_close
            history_rows.append({
                "date": today, "symbol": sym,
                "cash": state[sym]["cash"],
                "position": state[sym]["position"],
                "value": val,
            })

        # MDD cap 체크 — 오늘 종가 기준 portfolio total
        if mdd_cap is not None:
            today_total = sum(
                state[s]["cash"]
                + state[s]["position"]
                * float(sym_data[s].loc[today, "close"])
                for s in sym_data if today in sym_data[s].index
            )
            # peak 추적 (history_rows 기반)
            if history_rows:
                hist_df = pd.DataFrame(history_rows)
                day_totals = hist_df.groupby("date")["value"].sum()
                peak = float(day_totals.max())
                if peak > 0:
                    dd = (today_total - peak) / peak
                    if dd <= -mdd_cap:
                        mdd_armed = True

        if mdd_cooldown_remaining > 0:
            mdd_cooldown_remaining -= 1

        strat.advance_day()

    history = pd.DataFrame(history_rows)
    if history.empty:
        return PortfolioBacktestResult(history, pd.DataFrame(), {})

    # 종목별 summary
    sym_summary = []
    for sym in sym_data:
        sub = history[history["symbol"] == sym].sort_values("date")
        if sub.empty:
            continue
        final = float(sub["value"].iloc[-1])
        ret = (final - quota) / quota
        peak = sub["value"].cummax()
        dd = ((sub["value"] - peak) / peak).min()
        sym_summary.append({
            "symbol": sym,
            "start_value": quota,
            "final_value": final,
            "return": ret,
            "max_drawdown": float(dd),
        })
    summary = pd.DataFrame(sym_summary)

    # Overall (5종 합)
    by_date = history.groupby("date")["value"].sum()
    overall_final = float(by_date.iloc[-1])
    overall_ret = (overall_final - initial_cash) / initial_cash
    peak = by_date.cummax()
    overall_dd = float(((by_date - peak) / peak).min())

    overall = {
        "initial_cash": initial_cash,
        "final_value": overall_final,
        "return": overall_ret,
        "max_drawdown": overall_dd,
        "n_days": len(by_date),
    }
    return PortfolioBacktestResult(history, summary, overall)


def run_buyhold_portfolio(
    cfg: ForeignFlowBacktestConfig,
    initial_cash: float = 10_000_000.0,
) -> PortfolioBacktestResult:
    """매수후보유 portfolio baseline — 첫날 시가에 quota 만큼 매수, 끝까지 보유.

    동일 cost_bps, 동일 자본 quota 로 비교 가능한 baseline 산출.
    """
    cost_per_side = cfg.cost_bps / 10_000.0 / 2.0

    sym_data: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df
    if not sym_data:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    n_sym = len(sym_data)
    quota = initial_cash / n_sym
    rng = _common_date_range(sym_data)
    if rng is None:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})
    common_start, common_end = rng
    all_dates = sorted([
        d for d in set().union(*[set(df.index) for df in sym_data.values()])
        if common_start <= d <= common_end
    ])

    # 첫날 매수 한 번
    state: dict[str, dict] = {}
    for sym, df in sym_data.items():
        first_date = df.index[0]
        first_open = float(df.loc[first_date, "open"])
        if first_open <= 0:
            state[sym] = {"cash": quota, "position": 0}
            continue
        px_with_cost = first_open * (1 + cost_per_side)
        qty = int(quota // px_with_cost)
        spent = qty * first_open * (1 + cost_per_side)
        state[sym] = {"cash": quota - spent, "position": qty}

    history_rows: list[dict] = []
    for today in all_dates:
        for sym, df in sym_data.items():
            if today not in df.index:
                continue
            close = float(df.loc[today, "close"])
            val = state[sym]["cash"] + state[sym]["position"] * close
            history_rows.append({
                "date": today, "symbol": sym,
                "cash": state[sym]["cash"],
                "position": state[sym]["position"],
                "value": val,
            })

    history = pd.DataFrame(history_rows)
    sym_summary = []
    for sym in sym_data:
        sub = history[history["symbol"] == sym].sort_values("date")
        if sub.empty:
            continue
        final = float(sub["value"].iloc[-1])
        ret = (final - quota) / quota
        peak = sub["value"].cummax()
        dd = ((sub["value"] - peak) / peak).min()
        sym_summary.append({
            "symbol": sym,
            "start_value": quota,
            "final_value": final,
            "return": ret,
            "max_drawdown": float(dd),
        })
    summary = pd.DataFrame(sym_summary)

    by_date = history.groupby("date")["value"].sum()
    overall_final = float(by_date.iloc[-1])
    overall_ret = (overall_final - initial_cash) / initial_cash
    peak = by_date.cummax()
    overall_dd = float(((by_date - peak) / peak).min())

    overall = {
        "initial_cash": initial_cash,
        "final_value": overall_final,
        "return": overall_ret,
        "max_drawdown": overall_dd,
        "n_days": len(by_date),
    }
    return PortfolioBacktestResult(history, summary, overall)


def run_dual_dynamic_backtest(
    cfg: ForeignFlowBacktestConfig,
    initial_cash: float = 10_000_000.0,
    enter_threshold: float = 0.05,
    sizing_factor: float = 200.0,
    cost_bps: float | None = None,
    max_concurrent: int = 10,
    mdd_cap: float | None = None,
    cooldown_days: int = 10,
) -> PortfolioBacktestResult:
    """동적 universe + dual confirmation 백테스트.

    매일 universe 전체 종목 스캔:
      - 매수 후보: 어제 외인 ratio > +threshold AND 기관 ratio > +threshold
      - 매도 후보: 보유 + 어제 둘 다 < -threshold
    고정 universe 가 아니라 매일 신호 종목만 자동 진입.

    Args:
        max_concurrent: 동시 보유 가능 최대 종목 수 (자본 분산 한도)
        mdd_cap:        Portfolio drawdown 안전망 (예: 0.30). 도달 시 전 종목
                        강제 청산 + cooldown_days 동안 신규 진입 차단.
        cooldown_days:  MDD cap 발동 후 cooldown 일수.

    이 모드는 영웅문 화면 사용 방식과 일치한다.
    """
    cost_per_side = (cost_bps if cost_bps is not None else cfg.cost_bps) / 10_000.0 / 2.0

    sym_data: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df
    if not sym_data:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    rng = _common_date_range(sym_data)
    if rng is None:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})
    common_start, common_end = rng
    all_dates = sorted([
        d for d in set().union(*[set(df.index) for df in sym_data.values()])
        if common_start <= d <= common_end
    ])

    # 종목별 cash/position 상태 (자본 균등 분산 — 진입 시점에 가용 cash / 빈 슬롯 수)
    positions: dict[str, dict] = {}     # 보유 중인 종목만
    cash: float = initial_cash
    portfolio_peak: float = initial_cash
    cooldown_remaining: int = 0

    history_rows: list[dict] = []

    for i, today in enumerate(all_dates):
        if i == 0:
            history_rows.append({
                "date": today, "symbol": "_TOTAL_",
                "cash": cash, "position": 0,
                "value": cash,
            })
            continue

        prev = all_dates[i - 1]

        # ===== MDD Cap 체크 (전일 종가 기준 portfolio total 이 -mdd_cap 도달했나?) =====
        if mdd_cap is not None and not positions:
            pass  # 보유 0 이면 청산할 것 없음
        if mdd_cap is not None and positions:
            prev_total = cash
            for sym, pos in positions.items():
                df = sym_data[sym]
                if prev in df.index:
                    prev_total += pos["qty"] * float(df.loc[prev, "close"])
            if prev_total > 0 and portfolio_peak > 0:
                dd = (prev_total - portfolio_peak) / portfolio_peak
                if dd <= -mdd_cap:
                    # 오늘 시가에 모두 청산
                    for sym in list(positions.keys()):
                        df = sym_data[sym]
                        if today in df.index:
                            today_open = float(df.loc[today, "open"])
                            if today_open > 0:
                                cash += positions[sym]["qty"] * today_open * (1 - cost_per_side)
                        del positions[sym]
                    cooldown_remaining = cooldown_days

        # 1. 보유 종목 매도 신호 체크 (어제 dual 매도 신호?)
        sells_today = []
        for sym in list(positions.keys()):
            df = sym_data.get(sym)
            if df is None or prev not in df.index or today not in df.index:
                continue
            prev_row = df.loc[prev]
            flow = float(prev_row["flow_ratio"])
            inst = float(prev_row["inst_ratio"])
            if flow < -enter_threshold and inst < -enter_threshold:
                sells_today.append(sym)

        # 2. 매도 실행 (오늘 시가)
        for sym in sells_today:
            df = sym_data[sym]
            today_open = float(df.loc[today, "open"])
            if today_open <= 0:
                continue
            qty = positions[sym]["qty"]
            proceeds = qty * today_open * (1 - cost_per_side)
            cash += proceeds
            del positions[sym]

        # 3. 매수 후보 식별 (cooldown 중이면 skip)
        buy_candidates = []
        if cooldown_remaining <= 0:
            for sym, df in sym_data.items():
                if sym in positions:
                    continue   # 이미 보유 중이면 추가 매수 안 함 (단순화)
                if prev not in df.index or today not in df.index:
                    continue
                prev_row = df.loc[prev]
                flow = float(prev_row["flow_ratio"])
                inst = float(prev_row["inst_ratio"])
                if flow > enter_threshold and inst > enter_threshold:
                    strength = min(flow, inst)
                    buy_candidates.append((sym, strength, flow, inst))

        # 4. 매수 실행 — 여유 슬롯 만큼만 (강도 순)
        free_slots = max_concurrent - len(positions)
        buy_candidates.sort(key=lambda x: -x[1])     # 강도 내림차순
        for sym, strength, flow, inst in buy_candidates[:free_slots]:
            df = sym_data[sym]
            today_open = float(df.loc[today, "open"])
            if today_open <= 0:
                continue
            slot_cash = cash / max(1, free_slots)
            px_with_cost = today_open * (1 + cost_per_side)
            qty = int(slot_cash // px_with_cost)
            if qty <= 0:
                continue
            cost_total = qty * today_open * (1 + cost_per_side)
            cash -= cost_total
            positions[sym] = {"qty": qty, "avg_entry": today_open, "entry_date": today}

        # 5. portfolio value 마크 (오늘 종가 기준)
        total_value = cash
        for sym, pos in positions.items():
            df = sym_data[sym]
            if today in df.index:
                total_value += pos["qty"] * float(df.loc[today, "close"])

        # peak 업데이트 + cooldown 감소
        if total_value > portfolio_peak:
            portfolio_peak = total_value
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        history_rows.append({
            "date": today, "symbol": "_TOTAL_",
            "cash": cash, "position": len(positions),
            "value": total_value,
        })

    history = pd.DataFrame(history_rows)
    if history.empty:
        return PortfolioBacktestResult(history, pd.DataFrame(), {})

    by_date = history.set_index("date")["value"]
    final = float(by_date.iloc[-1])
    ret = (final - initial_cash) / initial_cash
    peak = by_date.cummax()
    mdd = float(((by_date - peak) / peak).min())

    overall = {
        "initial_cash": initial_cash,
        "final_value": final,
        "return": ret,
        "max_drawdown": mdd,
        "n_days": len(by_date),
    }
    return PortfolioBacktestResult(history, pd.DataFrame(), overall)


def universe_index(
    cfg: ForeignFlowBacktestConfig,
) -> pd.Series:
    """Universe 등가중 인덱스 — 각 종목 close 를 t=0 정규화 후 평균.

    하락장 / 강세장 구간 식별의 기준 시계열로 사용.
    """
    sym_data = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df
    if not sym_data:
        return pd.Series(dtype=float)

    # 각 종목 close 를 t=0 정규화
    norm = {}
    for sym, df in sym_data.items():
        first = float(df["close"].iloc[0])
        if first > 0:
            norm[sym] = df["close"] / first
    if not norm:
        return pd.Series(dtype=float)

    idx_df = pd.DataFrame(norm)
    return idx_df.mean(axis=1)


def detect_regimes(
    universe_idx: pd.Series,
    drawdown_threshold: float = 0.05,
) -> pd.Series:
    """Universe 인덱스 기반 강세/약세 regime 식별.

    drawdown <= -threshold 인 날짜 = "bear" (하락장).
    그 외 = "bull" (강세장).

    Returns: index=date, value="bull"|"bear"
    """
    peak = universe_idx.cummax()
    dd = (universe_idx - peak) / peak
    return pd.Series(
        ["bear" if d <= -drawdown_threshold else "bull" for d in dd],
        index=universe_idx.index,
    )


def analyze_regime_performance(
    history: pd.DataFrame,
    regimes: pd.Series,
) -> dict:
    """Strategy 의 portfolio history 를 regime 별로 분리해서 일별 수익률 분석.

    Returns:
        bull / bear 각 regime 의:
          - n_days, mean_daily_return, std_daily_return,
            cumulative_return, sharpe_daily
    """
    by_date = history.groupby("date")["value"].sum().sort_index()
    daily_ret = by_date.pct_change().dropna()
    # 인덱스 timestamp normalize (시간 정보 제거) 후 정확히 alignment
    daily_ret.index = pd.to_datetime(daily_ret.index).normalize()
    reg_aligned = regimes.copy()
    reg_aligned.index = pd.to_datetime(reg_aligned.index).normalize()
    common = daily_ret.index.intersection(reg_aligned.index)
    daily_ret = daily_ret.loc[common]
    reg_aligned = reg_aligned.loc[common]
    aligned = pd.DataFrame({"ret": daily_ret, "regime": reg_aligned}).dropna()
    out = {}
    for label in ["bull", "bear"]:
        sub = aligned[aligned["regime"] == label]["ret"]
        if len(sub) == 0:
            out[label] = {"n_days": 0}
            continue
        cum = float((1 + sub).prod() - 1)
        std = float(sub.std())
        mean = float(sub.mean())
        sharpe = mean / std * (252 ** 0.5) if std > 0 else float("nan")
        out[label] = {
            "n_days": int(len(sub)),
            "mean_daily_return": mean,
            "std_daily_return": std,
            "cumulative_return": cum,
            "sharpe_annualized": sharpe,
        }
    return out


def run_dual_dynamic_backtest_v2(
    cfg: ForeignFlowBacktestConfig,
    initial_cash: float = 10_000_000.0,
    enter_threshold: float = 0.05,
    cost_bps: float | None = None,
    max_concurrent: int = 10,
    mdd_cap: float | None = None,
    cooldown_days: int = 0,
    # ===== sell rule v2 =====
    # trailing_stop_pct: 양전(avg_entry 위) 상태에서 peak_close 대비 -X% 매도.
    # weak_sell_when_loss: 음전 상태에서 외인 OR 기관 한 쪽이라도 < -threshold 시 매도.
    # stop_loss_pct: 진입가 대비 -X% (absolute) 매도. 양/음전 무관.
    # trailing_lock_days: 진입 후 N일 동안은 trailing 무시 (whipsaw 보호).
    # time_exit_days: 진입 후 N일 지나면 무조건 청산.
    # sell_threshold_override: dual sell rule threshold 만 별도 설정 (asymmetric).
    #   None=enter_threshold 대칭. 예: 0.03 → -3% 만으로도 매도 (민감), 0.10 → -10% 강한
    trailing_stop_pct: float | None = None,
    weak_sell_when_loss: bool = False,
    stop_loss_pct: float | None = None,
    trailing_lock_days: int = 0,
    time_exit_days: int | None = None,
    sell_threshold_override: float | None = None,
    # ===== momentum confluence (buy 게이트) =====
    # momentum_lookback: N일 수익률을 모멘텀으로 사용. None=비활성(기존 동작).
    # momentum_min: 진입하려면 prev 시점 N일 모멘텀 > 이 값이어야 함 (기본 0 = 양의 추세).
    momentum_lookback: int | None = None,
    momentum_min: float = 0.0,
    # ===== replacement rule v2 (P&L 기반 — "횡보·저수익" 포지션만 교체 대상) =====
    # replacement_pnl_max: 보유 종목의 unrealized P&L 이 이 값 미만이면 교체 후보로 분류.
    #   None = 비활성. 예: 0.01 → +1% 미만(즉 손실·횡보)만 교체 대상.
    # replacement_min_hold: 진입 후 이 일수 미만은 교체 불가 (갓 산 포지션 보호).
    # v1 (strength 기준) 은 승자를 자르는 buggy 동작이라 폐기.
    replacement_pnl_max: float | None = None,
    replacement_min_hold: int = 3,
) -> PortfolioBacktestResult:
    """run_dual_dynamic_backtest 의 sell 룰 확장판.

    매수 룰은 동일 (외인 AND 기관 둘 다 +enter_threshold).
    매도 룰은 다음 중 어느 하나라도 충족 시 청산:
      (a) 기존 강한 dual: 외인 AND 기관 둘 다 < -enter_threshold
      (b) 양전 trailing: 현재 보유 종목이 진입가 위에 있을 때, peak_close 대비
          -trailing_stop_pct 이상 하락
      (c) 음전 weak: 진입가 아래일 때, 외인 OR 기관 한 쪽이라도
          < -enter_threshold

    가설 (사용자 2026-05-17):
      "양전이면 익절 보호용 trailing -10%, 음전이면 더 빠른 손절 (둘 다 가
      필요한 dual 보다 한 쪽만 필요한 OR)" — 5/12 식 급락에 sell signal 이
      늦게 도착하는 문제를 보완.
    """
    cost_per_side = (cost_bps if cost_bps is not None else cfg.cost_bps) / 10_000.0 / 2.0

    sym_data: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df
    if not sym_data:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    rng = _common_date_range(sym_data)
    if rng is None:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})
    common_start, common_end = rng
    all_dates = sorted([
        d for d in set().union(*[set(df.index) for df in sym_data.values()])
        if common_start <= d <= common_end
    ])

    # 모멘텀 confluence 게이트: 각 종목 close 의 N일 수익률 사전계산.
    # NaN(초기 N일)은 진입 조건에서 자동 탈락 → lookahead 없음.
    if momentum_lookback is not None:
        for _df in sym_data.values():
            _df["_mom"] = _df["close"].pct_change(momentum_lookback)

    positions: dict[str, dict] = {}
    cash: float = initial_cash
    portfolio_peak: float = initial_cash
    cooldown_remaining: int = 0
    history_rows: list[dict] = []
    sell_reasons: list[dict] = []

    for i, today in enumerate(all_dates):
        if i == 0:
            history_rows.append({
                "date": today, "symbol": "_TOTAL_",
                "cash": cash, "position": 0, "value": cash,
            })
            continue
        prev = all_dates[i - 1]

        # MDD Cap 체크 (전일 종가 기준)
        if mdd_cap is not None and positions:
            prev_total = cash
            for sym, pos in positions.items():
                df = sym_data[sym]
                if prev in df.index:
                    prev_total += pos["qty"] * float(df.loc[prev, "close"])
            if prev_total > 0 and portfolio_peak > 0:
                dd = (prev_total - portfolio_peak) / portfolio_peak
                if dd <= -mdd_cap:
                    for sym in list(positions.keys()):
                        df = sym_data[sym]
                        if today in df.index:
                            today_open = float(df.loc[today, "open"])
                            if today_open > 0:
                                cash += positions[sym]["qty"] * today_open * (1 - cost_per_side)
                        del positions[sym]
                    cooldown_remaining = cooldown_days
                    # peak 리셋 — 운영 코드와 동일 (무한 재발동 방지)
                    portfolio_peak = cash

        # ===== Sell rule v2 =====
        sells_today: list[tuple[str, str]] = []
        for sym in list(positions.keys()):
            df = sym_data.get(sym)
            if df is None or prev not in df.index or today not in df.index:
                continue
            prev_row = df.loc[prev]
            flow = float(prev_row["flow_ratio"])
            inst = float(prev_row["inst_ratio"])
            prev_close = float(prev_row["close"])
            pos = positions[sym]
            avg_entry = pos["avg_entry"]
            # peak_close 갱신 (전일 종가까지)
            pos["peak_close"] = max(pos.get("peak_close", avg_entry), prev_close)

            reason = None
            days_held = (prev - pos["entry_date"]).days
            sell_th = sell_threshold_override if sell_threshold_override is not None else enter_threshold
            # (a) 강한 dual (sell_th 가 매수와 다를 수 있음 — asymmetric)
            if flow < -sell_th and inst < -sell_th:
                reason = "dual_strong"
            # (b) absolute stop loss (진입가 대비 -X%) — 양/음전 무관, 최우선 보호
            elif (stop_loss_pct is not None and avg_entry > 0):
                pnl_from_entry = (prev_close - avg_entry) / avg_entry
                if pnl_from_entry <= -stop_loss_pct:
                    reason = "stop_loss"
            # (c) 양전 trailing — lock days 지나야 활성화
            if (reason is None
                  and trailing_stop_pct is not None
                  and prev_close > avg_entry
                  and pos["peak_close"] > 0
                  and days_held >= trailing_lock_days):
                dd_from_peak = (prev_close - pos["peak_close"]) / pos["peak_close"]
                if dd_from_peak <= -trailing_stop_pct:
                    reason = "trailing"
            # (d) 음전 weak
            if reason is None and weak_sell_when_loss and prev_close <= avg_entry:
                if flow < -enter_threshold or inst < -enter_threshold:
                    reason = "weak_when_loss"
            # (e) time exit
            if (reason is None and time_exit_days is not None
                  and days_held >= time_exit_days):
                reason = "time_exit"

            if reason is not None:
                sells_today.append((sym, reason))

        # 매도 실행 (오늘 시가)
        for sym, reason in sells_today:
            df = sym_data[sym]
            today_open = float(df.loc[today, "open"])
            if today_open <= 0:
                continue
            qty = positions[sym]["qty"]
            avg_entry = positions[sym]["avg_entry"]
            proceeds = qty * today_open * (1 - cost_per_side)
            cash += proceeds
            pnl = (today_open - avg_entry) / avg_entry if avg_entry > 0 else 0.0
            sell_reasons.append({
                "date": today, "symbol": sym, "reason": reason,
                "pnl": pnl, "avg_entry": avg_entry, "exit_price": today_open,
            })
            del positions[sym]

        # 매수 후보 (cooldown skip)
        buy_candidates = []
        if cooldown_remaining <= 0:
            for sym, df in sym_data.items():
                if sym in positions:
                    continue
                if prev not in df.index or today not in df.index:
                    continue
                prev_row = df.loc[prev]
                flow = float(prev_row["flow_ratio"])
                inst = float(prev_row["inst_ratio"])
                if flow > enter_threshold and inst > enter_threshold:
                    # 모멘텀 confluence: 추세도 같은 방향(상승)일 때만 진입.
                    # NaN(초기 구간) 은 `not (mom > min)` 이 True → 자동 탈락.
                    if momentum_lookback is not None:
                        mom = float(prev_row["_mom"])
                        if not (mom > momentum_min):
                            continue
                    strength = min(flow, inst)
                    buy_candidates.append((sym, strength, flow, inst))

        free_slots = max_concurrent - len(positions)
        buy_candidates.sort(key=lambda x: -x[1])

        # === Replacement rule v2: 슬롯이 차도 강한 후보가 있으면
        # "수익률이 낮은 횡보 포지션"만 골라 교체. 승자는 자동 보호 (pnl > max 면 비적격),
        # 갓 진입한 포지션도 보호 (held < min_hold). 가격은 prev close 로 marking
        # (signal-time, no lookahead) — 실제 체결은 today open.
        if replacement_pnl_max is not None and positions and len(buy_candidates) > free_slots:
            eligible: dict[str, float] = {}    # sym -> 현재 unrealized pnl (prev close 기준)
            for sym_h in list(positions.keys()):
                df_h = sym_data.get(sym_h)
                if df_h is None or prev not in df_h.index:
                    continue
                pos_h = positions[sym_h]
                if (prev - pos_h["entry_date"]).days < replacement_min_hold:
                    continue
                avg_entry = pos_h["avg_entry"]
                if avg_entry <= 0:
                    continue
                pnl_now = (float(df_h.loc[prev, "close"]) - avg_entry) / avg_entry
                if pnl_now < replacement_pnl_max:
                    eligible[sym_h] = pnl_now
            for cand_sym, _cand_strength, _, _ in buy_candidates[free_slots:]:
                if not eligible:
                    break
                weakest_sym = min(eligible, key=eligible.get)
                df_w = sym_data[weakest_sym]
                if today not in df_w.index:
                    del eligible[weakest_sym]
                    continue
                today_open_w = float(df_w.loc[today, "open"])
                if today_open_w <= 0:
                    del eligible[weakest_sym]
                    continue
                qty_w = positions[weakest_sym]["qty"]
                avg_entry_w = positions[weakest_sym]["avg_entry"]
                cash += qty_w * today_open_w * (1 - cost_per_side)
                pnl_exit = (today_open_w - avg_entry_w) / avg_entry_w if avg_entry_w > 0 else 0.0
                sell_reasons.append({
                    "date": today, "symbol": weakest_sym, "reason": "replacement",
                    "pnl": pnl_exit, "avg_entry": avg_entry_w, "exit_price": today_open_w,
                })
                del positions[weakest_sym]
                del eligible[weakest_sym]
                free_slots += 1

        for sym, strength, flow, inst in buy_candidates[:free_slots]:
            df = sym_data[sym]
            today_open = float(df.loc[today, "open"])
            if today_open <= 0:
                continue
            slot_cash = cash / max(1, free_slots)
            px_with_cost = today_open * (1 + cost_per_side)
            qty = int(slot_cash // px_with_cost)
            if qty <= 0:
                continue
            cost_total = qty * today_open * (1 + cost_per_side)
            cash -= cost_total
            positions[sym] = {
                "qty": qty, "avg_entry": today_open,
                "entry_date": today, "peak_close": today_open,
            }

        # portfolio mark
        total_value = cash
        for sym, pos in positions.items():
            df = sym_data[sym]
            if today in df.index:
                total_value += pos["qty"] * float(df.loc[today, "close"])
        if total_value > portfolio_peak:
            portfolio_peak = total_value
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        history_rows.append({
            "date": today, "symbol": "_TOTAL_",
            "cash": cash, "position": len(positions), "value": total_value,
        })

    history = pd.DataFrame(history_rows)
    if history.empty:
        return PortfolioBacktestResult(history, pd.DataFrame(), {})
    by_date = history.set_index("date")["value"]
    final = float(by_date.iloc[-1])
    ret = (final - initial_cash) / initial_cash
    peak = by_date.cummax()
    mdd = float(((by_date - peak) / peak).min())
    overall = {
        "initial_cash": initial_cash,
        "final_value": final, "return": ret,
        "max_drawdown": mdd, "n_days": len(by_date),
        "sell_reasons": sell_reasons,
    }
    return PortfolioBacktestResult(history, pd.DataFrame(), overall)


__all__ = [
    "ForeignFlowBacktestConfig",
    "run_foreign_flow_backtest",
    "buy_and_hold_baseline",
    "summarize_trades",
    "PortfolioBacktestResult",
    "run_portfolio_backtest",
    "run_buyhold_portfolio",
    "run_dual_dynamic_backtest",
    "run_dual_dynamic_backtest_v2",
    "universe_index",
    "detect_regimes",
    "analyze_regime_performance",
]
