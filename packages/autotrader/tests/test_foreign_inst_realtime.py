"""KIS realtime foreign+institution wrapper 단위 테스트 (mock response)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from autotrader.market.foreign_inst_realtime import (
    parse_foreign_institution_response,
    compute_flow_ratios,
    PersistentPayloadCache,
    RealtimeFlowCache,
)


# 실제 KIS 응답 형식 (2026-04-29 라이브 호출로 검증된 컬럼명)
SAMPLE_PAYLOAD = {
    "rt_cd": "0",
    "msg_cd": "MCA00000",
    "msg1": "정상처리 되었습니다.",
    "output": [
        {
            "hts_kor_isnm": "삼성전자",
            "mksc_shrn_iscd": "005930",
            "stck_prpr": "75000",
            "acml_vol": "10000000",
            "ntby_qty": "1469134",       # 외국인+기관 합계
            "frgn_ntby_qty": "1234567",
            "orgn_ntby_qty": "234567",
            "frgn_ntby_tr_pbmn": "92000",
            "orgn_ntby_tr_pbmn": "17500",
            "prdy_ctrt": "2.03",
        },
        {
            "hts_kor_isnm": "SK하이닉스",
            "mksc_shrn_iscd": "000660",
            "stck_prpr": "280000",
            "acml_vol": "4000000",
            "ntby_qty": "-580245",
            "frgn_ntby_qty": "-456789",
            "orgn_ntby_qty": "-123456",
            "frgn_ntby_tr_pbmn": "-127000",
            "orgn_ntby_tr_pbmn": "-34500",
            "prdy_ctrt": "-1.50",
        },
        {
            "hts_kor_isnm": "NAVER",
            "mksc_shrn_iscd": "035420",
            "stck_prpr": "180000",
            "acml_vol": "2000000",
            "ntby_qty": "0",
            "frgn_ntby_qty": "-",          # 빈 케이스
            "orgn_ntby_qty": "0",
            "frgn_ntby_tr_pbmn": "",
            "orgn_ntby_tr_pbmn": "0",
            "prdy_ctrt": "0",
        },
    ],
}


def test_parse_response_basic():
    df = parse_foreign_institution_response(SAMPLE_PAYLOAD)
    assert len(df) == 3
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["foreign_net_qty"] == 1234567
    assert df.iloc[1]["foreign_net_qty"] == -456789
    # NAVER row: '-' / '' 가 0 으로 안전하게 처리되어야
    assert df.iloc[2]["foreign_net_qty"] == 0
    assert df.iloc[2]["inst_net_qty"] == 0


def test_parse_response_empty():
    df = parse_foreign_institution_response({"output": []})
    assert df.empty


def test_parse_response_malformed():
    df = parse_foreign_institution_response({})
    assert df.empty


def test_compute_flow_ratios_with_volumes():
    df = parse_foreign_institution_response(SAMPLE_PAYLOAD)
    volumes = {
        "005930": 10_000_000,    # 거래량 1천만
        "000660": 4_000_000,     # 4백만
        "035420": 2_000_000,     # 2백만
    }
    out = compute_flow_ratios(df, daily_volumes=volumes)
    # 005930: 1,234,567 / 10,000,000 = 0.1235 (양수 매수 강세)
    assert abs(out.iloc[0]["flow_ratio"] - 0.1234567) < 1e-4
    assert out.iloc[0]["inst_ratio"] > 0
    # 000660: -456,789 / 4,000,000 = -0.114 (매도)
    assert out.iloc[1]["flow_ratio"] < 0
    assert out.iloc[1]["inst_ratio"] < 0
    # 035420: 0 / 2,000,000 = 0
    assert out.iloc[2]["flow_ratio"] == 0
    assert out.iloc[2]["inst_ratio"] == 0


def test_compute_flow_ratios_uses_response_volume_when_no_override():
    """daily_volumes=None 이면 응답의 acml_vol(volume) 사용."""
    df = parse_foreign_institution_response(SAMPLE_PAYLOAD)
    out = compute_flow_ratios(df, daily_volumes=None)
    # 005930: 1234567 / 10_000_000 = 0.1234567
    assert abs(out.iloc[0]["flow_ratio"] - 0.1234567) < 1e-4
    assert out.iloc[0]["inst_ratio"] > 0
    # 000660: -456789 / 4_000_000 = -0.1142
    assert out.iloc[1]["flow_ratio"] < -0.1
    # 035420: foreign_net_qty='-' → 0
    assert out.iloc[2]["flow_ratio"] == 0


def test_compute_flow_ratios_zero_volume():
    df = parse_foreign_institution_response(SAMPLE_PAYLOAD)
    volumes = {"005930": 0, "000660": 0, "035420": 0}
    out = compute_flow_ratios(df, daily_volumes=volumes)
    # division by zero 방어
    assert (out["flow_ratio"] == 0).all()


class MockKISClient:
    """KIS API 호출을 mock — foreign_institution_total 만 stub."""
    def __init__(self, payload):
        self._payload = payload
        self.call_count = 0

    def foreign_institution_total(self, **kwargs):
        self.call_count += 1
        return self._payload


def test_realtime_cache_first_call_fetches():
    cache = RealtimeFlowCache(update_interval=60.0)
    mock = MockKISClient(SAMPLE_PAYLOAD)
    df = cache.get(mock)
    assert len(df) == 3
    assert mock.call_count == 1


def test_realtime_cache_within_interval_uses_cache():
    cache = RealtimeFlowCache(update_interval=60.0)
    mock = MockKISClient(SAMPLE_PAYLOAD)
    cache.get(mock)
    cache.get(mock)
    # 두 번째 호출은 캐시에서 — API 호출 1번만
    assert mock.call_count == 1


def test_realtime_cache_force_bypasses():
    cache = RealtimeFlowCache(update_interval=60.0)
    mock = MockKISClient(SAMPLE_PAYLOAD)
    cache.get(mock)
    cache.get(mock, force=True)
    assert mock.call_count == 2


def test_realtime_cache_expired_refetches():
    cache = RealtimeFlowCache(update_interval=0.01)   # 10ms
    mock = MockKISClient(SAMPLE_PAYLOAD)
    cache.get(mock)
    time.sleep(0.05)
    cache.get(mock)
    assert mock.call_count == 2


# ===== PersistentPayloadCache (시초 데이터 공백 fallback) =====


def test_persistent_cache_save_then_load(tmp_path):
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    cache.save("0001", SAMPLE_PAYLOAD, SAMPLE_PAYLOAD)

    result = cache.load("0001")
    assert result is not None
    buy, sell, age_h = result
    assert buy == SAMPLE_PAYLOAD
    assert sell == SAMPLE_PAYLOAD
    assert age_h < 1.0   # 방금 저장 → 거의 0


def test_persistent_cache_save_skips_both_none(tmp_path):
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    cache.save("0001", None, None)
    assert cache.load("0001") is None


def test_persistent_cache_save_accepts_partial(tmp_path):
    """매수상위만 성공하고 매도상위는 None 이어도 저장 — 부분 fallback 허용."""
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    cache.save("0001", SAMPLE_PAYLOAD, None)
    result = cache.load("0001")
    assert result is not None
    buy, sell, _ = result
    assert buy == SAMPLE_PAYLOAD
    assert sell == {}


def test_persistent_cache_load_missing_file(tmp_path):
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    assert cache.load("0001") is None


def test_persistent_cache_stale_rejected(tmp_path):
    """max_age_hours 초과 stale 은 None — 주말 4일짜리 stale 차단 시나리오."""
    cache = PersistentPayloadCache(cache_dir=tmp_path, max_age_hours=48.0)
    record = {
        "saved_at": (datetime.now() - timedelta(hours=72)).isoformat(),
        "market_code": "0001",
        "payload_buy": SAMPLE_PAYLOAD,
        "payload_sell": SAMPLE_PAYLOAD,
    }
    (tmp_path / "dual_last_good_payload_0001.json").write_text(
        json.dumps(record), encoding="utf-8",
    )
    assert cache.load("0001") is None


def test_persistent_cache_market_codes_isolated(tmp_path):
    """KOSPI(0001) 와 KOSDAQ(1001) 은 별도 파일."""
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    other_payload = {"rt_cd": "0", "output": []}
    cache.save("0001", SAMPLE_PAYLOAD, SAMPLE_PAYLOAD)
    cache.save("1001", other_payload, other_payload)

    r0001 = cache.load("0001")
    r1001 = cache.load("1001")
    assert r0001 is not None and r0001[0] == SAMPLE_PAYLOAD
    assert r1001 is not None and r1001[0] == other_payload


def test_persistent_cache_corrupted_file(tmp_path):
    """손상된 JSON → None (예외 안 던짐)."""
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    (tmp_path / "dual_last_good_payload_0001.json").write_text(
        "{not valid json", encoding="utf-8",
    )
    assert cache.load("0001") is None


def test_persistent_cache_missing_saved_at(tmp_path):
    """saved_at 필드 누락 → None."""
    cache = PersistentPayloadCache(cache_dir=tmp_path)
    record = {"market_code": "0001", "payload_buy": SAMPLE_PAYLOAD, "payload_sell": {}}
    (tmp_path / "dual_last_good_payload_0001.json").write_text(
        json.dumps(record), encoding="utf-8",
    )
    assert cache.load("0001") is None


def test_persistent_cache_creates_dir(tmp_path):
    """cache_dir 가 없으면 자동 생성."""
    nested = tmp_path / "subdir" / "cache"
    cache = PersistentPayloadCache(cache_dir=nested)
    cache.save("0001", SAMPLE_PAYLOAD, SAMPLE_PAYLOAD)
    assert nested.exists()
    assert cache.load("0001") is not None


# ===== fetch_dual_candidates 통합 (fallback wiring) =====


def _load_runner():
    """scripts/run_dual_paper_trading.py 를 모듈로 로드."""
    import importlib.util
    runner_path = (Path(__file__).resolve().parents[3]
                   / "scripts" / "run_dual_paper_trading.py")
    spec = importlib.util.spec_from_file_location("rdpt_test", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MockClientEmpty:
    """가집계 API 가 빈 응답 — 시초 09:00 KIS 모의서버 패턴."""
    def foreign_institution_total(self, **kwargs):
        return {"rt_cd": "0", "output": []}


class MockClientGood:
    def __init__(self, payload):
        self._payload = payload
    def foreign_institution_total(self, **kwargs):
        return self._payload


def test_fetch_dual_saves_on_success(tmp_path, monkeypatch):
    """정상 응답 → cache 에 저장."""
    rdpt = _load_runner()
    monkeypatch.setattr(rdpt.time, "sleep", lambda *_: None)

    client = MockClientGood(SAMPLE_PAYLOAD)
    buy, sell, fb = rdpt.fetch_dual_candidates(client, cache_dir=tmp_path)
    assert fb is None
    assert (tmp_path / "dual_last_good_payload_0001.json").exists()


def test_fetch_dual_uses_fallback_on_empty(tmp_path, monkeypatch):
    """빈 응답 + cache 존재 → fallback 발동, 임계값 강화."""
    rdpt = _load_runner()
    monkeypatch.setattr(rdpt.time, "sleep", lambda *_: None)

    # 1) 사전 cache 저장 (어제 정상 응답)
    PersistentPayloadCache(cache_dir=tmp_path).save(
        "0001", SAMPLE_PAYLOAD, SAMPLE_PAYLOAD,
    )

    # 2) 오늘 시초 빈 응답
    client = MockClientEmpty()
    buy, sell, fb = rdpt.fetch_dual_candidates(client, cache_dir=tmp_path)

    assert fb is not None, "fallback 발동 안 됨"
    # 임계값이 1.5배로 강화돼야 함
    expected = rdpt.ENTER_THRESHOLD * rdpt.FALLBACK_THRESHOLD_MULT
    assert abs(fb["threshold"] - expected) < 1e-9
    assert fb["age_hours"] < 1.0


def test_fetch_dual_no_fallback_without_cache_dir(tmp_path, monkeypatch):
    """cache_dir=None → fallback 비활성, 빈 결과 그대로 반환 (backward compat)."""
    rdpt = _load_runner()
    monkeypatch.setattr(rdpt.time, "sleep", lambda *_: None)

    client = MockClientEmpty()
    buy, sell, fb = rdpt.fetch_dual_candidates(client, cache_dir=None)
    assert fb is None
    assert buy.empty
    assert sell.empty


def test_fetch_dual_no_fallback_when_cache_empty(tmp_path, monkeypatch):
    """cache_dir 지정했지만 cache 파일 없음 → fallback 못함, 빈 결과."""
    rdpt = _load_runner()
    monkeypatch.setattr(rdpt.time, "sleep", lambda *_: None)

    client = MockClientEmpty()
    buy, sell, fb = rdpt.fetch_dual_candidates(client, cache_dir=tmp_path)
    assert fb is None
    assert buy.empty


# ===== compute_effective_cap (강한 신호 시 cap 확장 #3) =====


def _cands_df(rows):
    """flow_ratio/inst_ratio 만 가진 buy_cands DataFrame 생성 helper."""
    return pd.DataFrame(rows)


def test_cap_no_extension_normal_signals():
    """평범한 신호 (모두 임계 통과지만 strong 미만) → cap 그대로."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": "A", "flow_ratio": 0.06, "inst_ratio": 0.06},   # strength 0.06 < 0.075
        {"symbol": "B", "flow_ratio": 0.055, "inst_ratio": 0.055},
    ])
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions={}, cfg_max_concurrent=7, fallback_info=None,
    )
    assert cap_eff == 7
    assert strong == 0
    assert ext == 0


def test_cap_extends_when_strong_overflow():
    """cap 도달 + strong 후보 다수 → cap 확장."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": f"S{i}", "flow_ratio": 0.10, "inst_ratio": 0.10}   # 모두 strong
        for i in range(5)
    ])
    positions = {f"X{i}": None for i in range(7)}   # cap 7 도달
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions=positions, cfg_max_concurrent=7, fallback_info=None,
    )
    # strong 5 - free_slots 0 = 5, 상한 3 → ext = 3
    assert ext == 3
    assert cap_eff == 10
    assert strong == 5


def test_cap_extends_partial_overflow():
    """cap 도달 + strong 후보 2종 → ext=2."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": "S1", "flow_ratio": 0.10, "inst_ratio": 0.10},
        {"symbol": "S2", "flow_ratio": 0.08, "inst_ratio": 0.08},
    ])
    positions = {f"X{i}": None for i in range(7)}
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions=positions, cfg_max_concurrent=7, fallback_info=None,
    )
    assert ext == 2
    assert cap_eff == 9


def test_cap_no_extension_with_free_slots():
    """free slots 충분 → strong 후보 다수여도 확장 안 함."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": f"S{i}", "flow_ratio": 0.10, "inst_ratio": 0.10}
        for i in range(5)
    ])
    positions = {"X1": None, "X2": None}   # free_slots = 5
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions=positions, cfg_max_concurrent=7, fallback_info=None,
    )
    assert ext == 0
    assert cap_eff == 7
    assert strong == 5   # strong 카운트는 정확히


def test_cap_no_extension_during_fallback():
    """fallback 사용 중이면 cap 확장 금지 (stale 데이터 over-trade 방지)."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": f"S{i}", "flow_ratio": 0.15, "inst_ratio": 0.15}
        for i in range(5)
    ])
    positions = {f"X{i}": None for i in range(7)}
    fallback_info = {"age_hours": 18.0, "threshold": 0.075}
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions=positions,
        cfg_max_concurrent=7, fallback_info=fallback_info,
    )
    assert ext == 0
    assert cap_eff == 7


def test_cap_extension_excludes_already_held():
    """이미 보유 중인 종목은 strong 카운트에서 제외."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": "X1", "flow_ratio": 0.10, "inst_ratio": 0.10},   # 이미 보유
        {"symbol": "S1", "flow_ratio": 0.10, "inst_ratio": 0.10},   # 신규
    ])
    positions = {f"X{i}": None for i in range(7)}   # X1 포함 7개 보유
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions=positions, cfg_max_concurrent=7, fallback_info=None,
    )
    # 신규 strong = S1 한 개만 → ext = 1
    assert strong == 1
    assert ext == 1
    assert cap_eff == 8


def test_cap_extension_respects_max_limit():
    """MAX_CAP_EXTENSION (3) 이상으로는 확장 안 함."""
    rdpt = _load_runner()
    cands = _cands_df([
        {"symbol": f"S{i}", "flow_ratio": 0.20, "inst_ratio": 0.20}
        for i in range(20)   # strong 후보 20종
    ])
    positions = {f"X{i}": None for i in range(7)}
    cap_eff, _, ext = rdpt.compute_effective_cap(
        cands, positions=positions, cfg_max_concurrent=7, fallback_info=None,
    )
    assert ext == rdpt.MAX_CAP_EXTENSION
    assert cap_eff == 7 + rdpt.MAX_CAP_EXTENSION


def test_cap_no_extension_empty_candidates():
    """매수 후보 없으면 확장 없음 (분기 안전성)."""
    rdpt = _load_runner()
    cands = pd.DataFrame(columns=["symbol", "flow_ratio", "inst_ratio"])
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions={}, cfg_max_concurrent=7, fallback_info=None,
    )
    assert ext == 0
    assert cap_eff == 7
    assert strong == 0


def test_cap_strong_threshold_uses_weaker_side():
    """strength = min(flow, inst). 한쪽만 강하면 strong 아님."""
    rdpt = _load_runner()
    cands = _cands_df([
        # flow 강해도 inst 약하면 strong 아님 (min = 0.06)
        {"symbol": "A", "flow_ratio": 0.20, "inst_ratio": 0.06},
        # 둘 다 strong (min = 0.08)
        {"symbol": "B", "flow_ratio": 0.08, "inst_ratio": 0.08},
    ])
    positions = {f"X{i}": None for i in range(7)}
    cap_eff, strong, ext = rdpt.compute_effective_cap(
        cands, positions=positions, cfg_max_concurrent=7, fallback_info=None,
    )
    assert strong == 1   # B 만
    assert ext == 1
    assert cap_eff == 8


# ===== should_force_sell_underperform (#5 코스피 대비 강제 청산) =====


from autotrader.paper.dual_state import Position


def _pos(avg_entry=10000.0, kospi=2500.0, entry_date="2026-05-20"):
    return Position(
        qty=10, avg_entry=avg_entry,
        entry_date=entry_date, kospi_at_entry=kospi,
    )


def test_underperform_sells_when_below_kospi_by_threshold():
    """종목 -3%, 코스피 +5% → relative -8% < -5% → 청산."""
    rdpt = _load_runner()
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-20")
    today = datetime(2026, 5, 26)
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=9700, current_kospi=2625, today=today,
    )
    assert should is True
    # stock -3%, kospi +5%, relative -8%
    assert abs(rel - (-0.08)) < 1e-6


def test_underperform_holds_when_relative_within_threshold():
    """relative -3% → 임계 -5% 미만 → 보유."""
    rdpt = _load_runner()
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-20")
    today = datetime(2026, 5, 26)
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=9700, current_kospi=2500, today=today,
    )
    assert should is False
    assert abs(rel - (-0.03)) < 1e-6


def test_underperform_holds_when_outperforming():
    """종목 +5%, 코스피 -2% → relative +7% → 보유."""
    rdpt = _load_runner()
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-20")
    today = datetime(2026, 5, 26)
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=10500, current_kospi=2450, today=today,
    )
    assert should is False
    assert rel > 0


def test_underperform_skips_when_kospi_at_entry_missing():
    """legacy 보유 종목 (kospi_at_entry=None) → skip (backward compat)."""
    rdpt = _load_runner()
    pos = Position(
        qty=10, avg_entry=10000.0, entry_date="2026-05-20",
        kospi_at_entry=None,
    )
    today = datetime(2026, 5, 26)
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=8000, current_kospi=2700, today=today,
    )
    assert should is False
    assert rel is None


def test_underperform_skips_when_held_too_briefly():
    """매수 후 1일만 지남 (min_hold_days=2 미만) → skip."""
    rdpt = _load_runner()
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-25")
    today = datetime(2026, 5, 26)   # 1 day
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=8000, current_kospi=2700, today=today,
    )
    assert should is False


def test_underperform_unknown_entry_date_skipped():
    """entry_date='unknown' (KIS sync backfill) → skip."""
    rdpt = _load_runner()
    pos = Position(
        qty=10, avg_entry=10000.0, entry_date="unknown",
        kospi_at_entry=2500.0,
    )
    today = datetime(2026, 5, 26)
    should, _ = rdpt.should_force_sell_underperform(
        pos, current_price=8000, current_kospi=2700, today=today,
    )
    assert should is False


def test_underperform_zero_kospi_skipped():
    """current_kospi=0 (조회 실패) → skip."""
    rdpt = _load_runner()
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-20")
    today = datetime(2026, 5, 26)
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=9000, current_kospi=0, today=today,
    )
    assert should is False
    assert rel is None


def test_underperform_boundary_exactly_at_threshold():
    """relative -5% 정확히 (임계와 같음) → 청산 안 함 (strict less than)."""
    rdpt = _load_runner()
    # stock 0%, kospi +5% → relative -5%
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-20")
    today = datetime(2026, 5, 26)
    should, rel = rdpt.should_force_sell_underperform(
        pos, current_price=10000, current_kospi=2625, today=today,
    )
    assert should is False    # 정확히 -5% 면 아직 보유
    assert abs(rel - (-0.05)) < 1e-6


def test_parse_kospi_quote_basic():
    """KIS index_price 응답 파싱."""
    rdpt = _load_runner()
    payload = {
        "rt_cd": "0",
        "output": {
            "bstp_nmix_prpr": "2,650.45",
            "bstp_nmix_prdy_vrss": "78.30",
            "bstp_nmix_prdy_ctrt": "3.05",
        },
    }
    assert abs(rdpt._parse_kospi_quote(payload) - 2650.45) < 1e-6


def test_parse_kospi_quote_malformed():
    rdpt = _load_runner()
    assert rdpt._parse_kospi_quote({}) == 0.0
    assert rdpt._parse_kospi_quote({"output": None}) == 0.0
    assert rdpt._parse_kospi_quote({"output": {"bstp_nmix_prpr": "not-a-number"}}) == 0.0


def test_days_between_correctly_counts():
    rdpt = _load_runner()
    assert rdpt._days_between("2026-05-20", datetime(2026, 5, 26)) == 6
    assert rdpt._days_between("2026-05-26", datetime(2026, 5, 26)) == 0
    assert rdpt._days_between("unknown", datetime(2026, 5, 26)) == 0
    assert rdpt._days_between("", datetime(2026, 5, 26)) == 0


def test_underperform_with_custom_threshold():
    """threshold override 작동 확인."""
    rdpt = _load_runner()
    pos = _pos(avg_entry=10000, kospi=2500, entry_date="2026-05-20")
    today = datetime(2026, 5, 26)
    # relative -3%, threshold -2% → 청산
    should, _ = rdpt.should_force_sell_underperform(
        pos, current_price=9700, current_kospi=2500,
        today=today, threshold=0.02,
    )
    assert should is True
