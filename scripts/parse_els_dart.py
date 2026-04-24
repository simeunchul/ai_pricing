"""Parse ELS spec from DART document.xml (일괄신고추가서류).

Usage:
    python scripts/parse_els_dart.py data/els_samples/dart_*.xml
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path


def xml_to_text(xml_path: str) -> str:
    """Strip XML tags and normalize whitespace."""
    raw = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = re.sub(r"&amp;", "&", txt)
    txt = re.sub(r"&nbsp;", " ", txt)
    txt = re.sub(r"&lt;", "<", txt)
    txt = re.sub(r"&gt;", ">", txt)
    txt = re.sub(r"&#x?\d+;", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt


def parse_els_spec(text: str) -> dict:
    out = {
        "issue_no": None,
        "issue_date": None,
        "maturity_date": None,
        "maturity_years": None,
        "notional_krw": None,
        "issue_price_krw": None,
        "underlyings": [],
        "coupon_rate_pct_per_year": None,
        "ki_barrier_pct": None,
        "autocall_barriers_pct": [],
        "observation_period_months": None,
    }

    # 제 XXXX 호
    for pat in [r"제\s*(\d{3,5})\s*호", r"스마트ELS\s*제\s*(\d+)\s*호",
                r"제(\d+)회", r"회차\s*[:：]?\s*(\d+)"]:
        m = re.search(pat, text)
        if m:
            out["issue_no"] = int(m.group(1)); break

    # 발행일 2026년 04월 24일
    m = re.search(r"발\s*행\s*일[자]?\s*[:：]?\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if m:
        out["issue_date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 만기일
    m = re.search(r"만\s*기\s*일?\s*[:：]?\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if m:
        out["maturity_date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 발행금액 총액 (원)
    m = re.search(r"(?:발행총액|모집\s*총\s*액|총\s*발행\s*금액)\s*[:：]?\s*([\d,]{7,})\s*원", text)
    if m:
        out["notional_krw"] = int(m.group(1).replace(",", ""))

    # 발행가액 10,000원 등
    m = re.search(r"(?:발행가액|1증권당|1좌당|단위당|액면가액)\s*[:：]?\s*([\d,]+)\s*원", text)
    if m:
        out["issue_price_krw"] = int(m.group(1).replace(",", ""))

    # 쿠폰율 — "연 X.XX%" or "수익률 X.XX%" or "X.XX% (연)"
    for pat in [
        r"연\s*환산?\s*수익률\s*[:：]?\s*(\d+\.?\d*)\s*%",
        r"연\s*수익률\s*[:：]?\s*(\d+\.?\d*)\s*%",
        r"연\s*(\d+\.?\d{1,3})\s*%",
        r"수익률\s*[:：]?\s*(\d+\.?\d*)\s*%\s*\(세전[,\s]*연[\)]",
    ]:
        m = re.search(pat, text)
        if m:
            rate = float(m.group(1))
            if 0.5 < rate < 30:  # sanity guard
                out["coupon_rate_pct_per_year"] = rate; break

    # KI barrier
    for pat in [
        r"원금\s*손실\s*조건\s*[:：]?\s*(\d{2,3})\s*%",
        r"Knock[\-\s]*In\s*(?:수준|Barrier|Level)?\s*[:：]?\s*(\d{2,3})\s*%",
        r"(?:하락한도|손실한계)\s*(\d{2,3})\s*%",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out["ki_barrier_pct"] = int(m.group(1)); break

    # Observation / autocall schedule — look for barrier sequence
    # E.g. "95%-90%-90%-85%-85%-80%" or "1차 95% 2차 90% ..."
    seqs = re.findall(r"((?:\d{2,3}\s*%[\s\-\,/]*){4,})", text)
    for s in seqs:
        nums = [int(n) for n in re.findall(r"\d{2,3}", s) if 50 <= int(n) <= 100]
        if 4 <= len(nums) <= 8 and all(nums[i] >= nums[i+1] - 2 for i in range(len(nums)-1)):
            # non-increasing (or nearly) sequence — typical step-down
            out["autocall_barriers_pct"] = nums[:6]
            break

    # Underlyings
    underlyings_map = {
        "KOSPI200": ["KOSPI200", "KOSPI 200", "코스피200"],
        "HSCEI": ["HSCEI", "항셍중국기업"],
        "S&P500": ["S&P500", "S&P 500"],
        "SX5E": ["EuroStoxx50", "EuroStoxx 50", "SX5E", "유로스톡스"],
        "NIKKEI225": ["NIKKEI225", "NIKKEI 225", "닛케이"],
        "SPX": ["SPX"],
        "삼성전자": ["삼성전자"],
        "SK하이닉스": ["SK하이닉스"],
        "NAVER": ["NAVER", "네이버"],
    }
    for canon, variants in underlyings_map.items():
        if any(v in text for v in variants):
            out["underlyings"].append(canon)

    return out


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("data/els_samples/dart_*.xml")
    results = []
    for p in paths:
        if not Path(p).exists():
            continue
        print(f"\n=== {Path(p).name} ===")
        txt = xml_to_text(p)
        spec = parse_els_spec(txt)
        spec["_file"] = Path(p).name
        results.append(spec)
        for k, v in spec.items():
            if k.startswith("_"): continue
            print(f"  {k}: {v}")

    out = Path("data/els_samples/parsed_dart.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ wrote {out}")


if __name__ == "__main__":
    main()
