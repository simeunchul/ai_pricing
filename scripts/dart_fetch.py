"""DART 전자공시 Open API 호출 유틸.

- 한화투자증권 corp_code 조회
- 최근 ELB/ELS 공시 목록
- 특정 rcept_no 의 보고서 PDF 링크

Usage:
    export DART_API_KEY=...    # or .env
    python scripts/dart_fetch.py corpcode
    python scripts/dart_fetch.py list --days 30
    python scripts/dart_fetch.py spec --rcept 20260424000XXX
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import requests


def get_api_key() -> str:
    key = os.environ.get("DART_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DART_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("DART_API_KEY not set (export or .env)")


def fetch_corp_codes(cache: Path = Path("data/dart_corp_codes.xml")) -> ET.Element:
    """Download the master corp_code XML (zipped). Cache locally."""
    if cache.exists() and (datetime.now().timestamp() - cache.stat().st_mtime) < 86400 * 7:
        return ET.parse(cache).getroot()

    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    r = requests.get(url, params={"crtfc_key": get_api_key()}, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    xml = z.read("CORPCODE.xml").decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(xml, encoding="utf-8")
    return ET.fromstring(xml)


def find_corp(root: ET.Element, name_contains: str) -> list[dict]:
    hits = []
    for node in root.findall("list"):
        nm = (node.findtext("corp_name") or "").strip()
        if name_contains in nm:
            hits.append({
                "corp_code": node.findtext("corp_code"),
                "corp_name": nm,
                "stock_code": (node.findtext("stock_code") or "").strip(),
                "modify_date": node.findtext("modify_date"),
            })
    # Prefer listed companies (with stock_code) and most recently modified first
    hits.sort(key=lambda h: (0 if h["stock_code"] else 1, -(int(h["modify_date"] or 0))))
    return hits


def list_disclosures(corp_code: str, days: int = 30, page_count: int = 100) -> list[dict]:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    url = "https://opendart.fss.or.kr/api/list.json"
    r = requests.get(url, params={
        "crtfc_key": get_api_key(),
        "corp_code": corp_code,
        "bgn_de": start,
        "end_de": end,
        "page_no": 1,
        "page_count": page_count,
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "000":
        print(f"[dart] status={data.get('status')} {data.get('message')}", file=sys.stderr)
        return []
    return data.get("list", [])


def disclosure_pdf_url(rcept_no: str) -> str:
    return f"https://dart.fss.or.kr/pdf/download/main.do?rcp_no={rcept_no}"


def cmd_corpcode(args):
    root = fetch_corp_codes()
    hits = find_corp(root, args.name)
    print(f"Found {len(hits)} matches for '{args.name}':")
    for h in hits[:20]:
        print(f"  corp_code={h['corp_code']}  stock={h['stock_code']:<7}  name={h['corp_name']}  ({h['modify_date']})")
    if hits:
        print(f"\nExample first match corp_code = {hits[0]['corp_code']}")


def cmd_list(args):
    root = fetch_corp_codes()
    hits = find_corp(root, args.name)
    if not hits:
        print(f"Corp not found: {args.name}", file=sys.stderr); return
    corp = hits[0]
    print(f"Querying disclosures for {corp['corp_name']} (corp_code={corp['corp_code']})...")
    items = list_disclosures(corp["corp_code"], days=args.days)

    kw = args.filter
    filt = [x for x in items if (not kw) or kw.lower() in x.get("report_nm", "").lower()]
    print(f"Got {len(items)} total, {len(filt)} after filter '{kw}'\n")

    for it in filt[:args.limit]:
        print(f"  {it['rcept_dt']}  rcept_no={it['rcept_no']}  {it['report_nm']}")
        print(f"     pdf: {disclosure_pdf_url(it['rcept_no'])}")


def fetch_document_xml(rcept_no: str) -> bytes:
    """DART API document.xml: returns the filing body as ZIP with XML inside."""
    url = "https://opendart.fss.or.kr/api/document.xml"
    r = requests.get(url, params={"crtfc_key": get_api_key(), "rcept_no": rcept_no}, timeout=30)
    r.raise_for_status()
    return r.content


def cmd_spec(args):
    content = fetch_document_xml(args.rcept)
    out = Path(args.out) if args.out else Path(f"data/els_samples/dart_{args.rcept}.xml")
    out.parent.mkdir(parents=True, exist_ok=True)

    # The response is a ZIP containing one or more XML files
    if content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(content))
        names = z.namelist()
        print(f"ZIP contains: {names}")
        for nm in names:
            target = out.parent / f"dart_{args.rcept}_{nm}"
            target.write_bytes(z.read(nm))
            print(f"  → {target} ({len(z.read(nm))//1024} KB)")
    else:
        out.write_bytes(content)
        print(f"Wrote raw: {out} ({len(content)//1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpcode", help="Find corp_code by company name")
    p.add_argument("--name", default="한화투자증권")
    p.set_defaults(func=cmd_corpcode)

    p = sub.add_parser("list", help="List recent disclosures")
    p.add_argument("--name", default="한화투자증권")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--filter", default="",
                   help="Substring filter on report_nm (예: 'ELB', 'ELS', '투자설명서')")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("spec", help="Download a specific disclosure PDF by rcept_no")
    p.add_argument("--rcept", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_spec)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
