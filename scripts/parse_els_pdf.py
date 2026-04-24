"""Extract ELS spec from Hanwha 증권발행실적보고서 PDF.

Usage:
    python scripts/parse_els_pdf.py
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pdfplumber


def extract_text(pdf_path: str) -> str:
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def parse_els(text: str) -> dict:
    """Best-effort parse of common fields from a Hanwha 증권발행실적보고서."""
    out = {
        "issue_no": None,
        "issue_date": None,
        "maturity_years": None,
        "notional_krw": None,
        "underlyings": [],
        "coupon_rate_pct": None,
        "ki_barrier_pct": None,
        "autocall_barriers_pct": [],
        "issue_price_krw": None,
    }

    # 제 XXX 호
    m = re.search(r"제\s*(\d+)\s*호", text)
    if m:
        out["issue_no"] = int(m.group(1))

    # 발행일
    m = re.search(r"발행일[자]?\s*[:：]?\s*(\d{4})[년\.\-/]\s*(\d{1,2})[월\.\-/]\s*(\d{1,2})", text)
    if m:
        out["issue_date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 만기 (예: "만기 3년" or "3년")
    m = re.search(r"(?:만기|만기일|상환일).{0,20}?(\d+)\s*년", text)
    if m:
        out["maturity_years"] = int(m.group(1))

    # 발행금액 (예: "발행가액 10,000,000,000원" or "총 발행금액")
    m = re.search(r"(?:발행가?액|발행총액|총발행금액)\s*[:：]?\s*([\d,]+)", text)
    if m:
        out["notional_krw"] = int(m.group(1).replace(",", ""))

    # 쿠폰율 (예: "수익률 6.00%", "쿠폰 3.00%", "연 6.00%")
    m = re.search(r"(?:연수익률|수익률|쿠폰|coupon).{0,15}?(\d+\.?\d*)\s*%", text, re.IGNORECASE)
    if m:
        out["coupon_rate_pct"] = float(m.group(1))

    # KI 배리어 (예: "Knock-In 50%" or "원금손실조건 50%")
    m = re.search(r"(?:Knock-?In|KI|원금손실조건).{0,15}?(\d{2,3})\s*%", text, re.IGNORECASE)
    if m:
        out["ki_barrier_pct"] = int(m.group(1))

    # 자동조기상환 barrier (예: "95-90-90-85-80-75%")
    m = re.search(r"(\d{2,3}(?:[\-\,]\s*\d{2,3}){3,})", text)
    if m:
        nums = re.findall(r"\d{2,3}", m.group(1))
        out["autocall_barriers_pct"] = [int(n) for n in nums if 50 <= int(n) <= 100]

    # 기초자산
    for keyword in ["KOSPI200", "KOSPI 200", "HSCEI", "S&P500", "S&P 500",
                    "EuroStoxx50", "EuroStoxx 50", "NIKKEI225", "NIKKEI 225",
                    "SX5E", "SPX"]:
        if keyword in text:
            out["underlyings"].append(keyword.replace(" ", ""))

    # 개별 종목 (삼성전자 등)
    for stock in ["삼성전자", "SK하이닉스", "현대차", "NAVER", "카카오", "LG화학"]:
        if stock in text:
            out["underlyings"].append(stock)

    out["underlyings"] = list(dict.fromkeys(out["underlyings"]))  # dedup, keep order

    # 발행가 (단위별 발행가)
    m = re.search(r"(?:발행가|1좌당|단위당|액면가).{0,15}?([\d,]+)\s*원", text)
    if m:
        out["issue_price_krw"] = int(m.group(1).replace(",", ""))

    return out


def main():
    pdfs = sorted(glob.glob("data/els_samples/*.pdf"))
    results = []
    for p in pdfs:
        print(f"\n=== {Path(p).name} ===")
        try:
            text = extract_text(p)
            spec = parse_els(text)
            spec["_file"] = Path(p).name
            spec["_text_preview"] = text[:600].replace("\n", " / ")
            results.append(spec)
            for k, v in spec.items():
                if k.startswith("_"):
                    continue
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  [parse error] {e}")

    out_path = Path("data/els_samples/parsed.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
