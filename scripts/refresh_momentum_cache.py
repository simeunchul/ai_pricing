"""모멘텀 캐시(data/krx_cache/*_combined.parquet) 일괄 갱신 — ①번 주력.

run_dual_paper_trading 의 _get_momentum() 은 이 캐시(종목별 일봉)를 읽어 60일/10일
가격 추세를 판정한다. 캐시에 없거나 오래된 종목은 매수/매도 룰에서 "데이터없음"으로
떨어져 사실상 차단된다. 이 스크립트가 평소 캐시를 넓게·최신으로 채우는 주력이고,
봇 안의 on-demand fallback(②)은 이 스크립트가 못 메운 종목만 그 자리에서 1회 보강한다.

갱신 대상 유니버스 =
  (1) 이미 캐시에 있는 종목  (계속 최신 유지)
  (2) 현재 보유 종목         (data/dual_state.json)
  (3) SEED 대형주/주요 종목  (아래 SEED_UNIVERSE)
  (4) --with-kis 시 KIS 가집계 매수/매도 상위 60종 (오늘 봇이 실제로 볼 후보)

데이터 출처는 네이버 금융 스크래핑(krx_investor.ensure_combined). 공개 데이터라
종목 제한은 없으며, "종목 부족"은 데이터 비공개가 아니라 캐시 미적재 문제였다.

사용 예:
  python scripts/refresh_momentum_cache.py                 # stale/미적재만 갱신
  python scripts/refresh_momentum_cache.py --with-kis      # + 오늘 KIS 후보까지
  python scripts/refresh_momentum_cache.py --force         # 전종목 강제 재수집
  python scripts/refresh_momentum_cache.py --limit 5       # 앞 5종목만 (스모크 테스트)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

from autotrader.data.krx_investor import (  # noqa: E402
    InvestorFlowCache, ensure_combined, combined_last_date,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("refresh_cache")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 거래대금 상위 대형주 + 주요 종목 seed (KOSPI 위주 + KOSDAQ 일부).
# 봇 후보가 대부분 여기서 나오므로 캐시 backbone 역할. 중복은 union 으로 흡수.
SEED_UNIVERSE: list[str] = [
    "005930", "000660", "373220", "207940", "005380", "000270", "005490",
    "035420", "035720", "051910", "006400", "068270", "105560", "055550",
    "086790", "316140", "138930", "024110", "003550", "012330", "028260",
    "066570", "015760", "017670", "030200", "033780", "096770", "034730",
    "010130", "011200", "009150", "032830", "000810", "323410", "003670",
    "010950", "011170", "018260", "090430", "251270", "036570", "247540",
    "086520", "196170", "000100", "128940", "302440", "326030",
]


def _load_position_symbols(state_path: Path) -> list[str]:
    if not state_path.exists():
        return []
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return list((data.get("positions") or {}).keys())
    except Exception as e:
        log.warning(f"state 읽기 실패 ({state_path}): {e}")
        return []


def _load_cached_symbols(cache: InvestorFlowCache) -> list[str]:
    return sorted({
        p.name.replace("_combined.parquet", "")
        for p in cache.cache_dir.glob("*_combined.parquet")
    })


def _harvest_kis_symbols(market_code: str) -> list[str]:
    """KIS 가집계 매수/매도 상위에서 종목코드 수집 (오늘 봇이 보는 후보)."""
    try:
        from autotrader.broker.kis_client import KISClient, KISConfig
        from autotrader.market.foreign_inst_realtime import (
            parse_foreign_institution_response,
        )
    except Exception as e:
        log.warning(f"KIS 모듈 import 실패 — --with-kis skip: {e}")
        return []
    try:
        client = KISClient(KISConfig.from_env())
        syms: set[str] = set()
        for rank_sort in ("0", "1"):  # 0=매수상위, 1=매도상위
            payload = client.foreign_institution_total(
                market_code=market_code, rank_sort=rank_sort,
            )
            df = parse_foreign_institution_response(payload or {})
            if not df.empty and "symbol" in df.columns:
                syms.update(df["symbol"].astype(str).tolist())
            time.sleep(1.0)
        log.info(f"KIS 가집계 후보 {len(syms)}종 수집")
        return sorted(syms)
    except Exception as e:
        log.warning(f"KIS 가집계 수집 실패 — seed/캐시만 사용: {e}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="모멘텀 캐시 일괄 갱신")
    ap.add_argument("--max-pages", type=int, default=60,
                    help="네이버 페이지네이션 깊이 (≈10영업일/페이지, 60=~2.4년)")
    ap.add_argument("--max-age-days", type=int, default=0,
                    help="캐시 마지막 거래일이 오늘 기준 이 일수 이내면 skip "
                         "(0=오늘 데이터 없으면 갱신)")
    ap.add_argument("--min-rows", type=int, default=250,
                    help="이 행 수 미만 캐시는 신선해도 재수집(얇은 on-demand 캐시 보강)")
    ap.add_argument("--force", action="store_true",
                    help="신선도 무관 전종목 강제 재수집")
    ap.add_argument("--with-kis", action="store_true",
                    help="KIS 가집계 매수/매도 상위 60종도 유니버스에 포함")
    ap.add_argument("--market-code", default="0001")
    ap.add_argument("--request-sleep", type=float, default=0.3,
                    help="네이버 페이지 간 sleep(초) — 서버 부담 회피")
    ap.add_argument("--limit", type=int, default=0,
                    help="유니버스 앞 N종목만 처리 (0=전체, 스모크 테스트용)")
    ap.add_argument("--state", default=str(ROOT / "data" / "dual_state.json"))
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cache = InvestorFlowCache(
        cache_dir=Path(args.cache_dir)
    ) if args.cache_dir else InvestorFlowCache()

    # ── 유니버스 조립 ──────────────────────────────────────────────
    cached = _load_cached_symbols(cache)
    positions = _load_position_symbols(Path(args.state))
    kis_syms = _harvest_kis_symbols(args.market_code) if args.with_kis else []

    universe = sorted(set(cached) | set(positions) | set(SEED_UNIVERSE) | set(kis_syms))
    if args.limit > 0:
        universe = universe[:args.limit]

    log.info(
        f"유니버스 {len(universe)}종 "
        f"(기존캐시 {len(cached)}, 보유 {len(positions)}, "
        f"seed {len(SEED_UNIVERSE)}, KIS {len(kis_syms)})"
    )
    max_age = -1 if args.force else args.max_age_days

    counts = {"fresh": 0, "refetched": 0, "failed": 0}
    failed_syms: list[str] = []
    t0 = time.time()
    for i, sym in enumerate(universe, 1):
        try:
            _, action = ensure_combined(
                sym,
                max_age_days=max_age,
                min_rows=args.min_rows,
                max_pages=args.max_pages,
                cache=cache,
                request_sleep=args.request_sleep,
            )
        except Exception as e:
            action = "failed"
            log.warning(f"  [{i}/{len(universe)}] {sym} 예외: {type(e).__name__}: {e}")
        counts[action] = counts.get(action, 0) + 1
        if action == "failed":
            failed_syms.append(sym)
        if action != "fresh":  # 재수집/실패만 로그 (skip 은 조용히)
            last = combined_last_date(sym, cache)
            log.info(
                f"  [{i}/{len(universe)}] {sym} {action}"
                + (f" → 최신 {last.date()}" if last is not None else "")
            )

    dt = time.time() - t0
    log.info(
        f"완료 ({dt:.0f}s): fresh(skip) {counts['fresh']}, "
        f"refetched {counts['refetched']}, failed {counts['failed']}"
    )
    if failed_syms:
        log.warning(f"실패 종목 {len(failed_syms)}종: {', '.join(failed_syms[:20])}"
                    + (" …" if len(failed_syms) > 20 else ""))
    return 0 if counts["refetched"] or counts["fresh"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
