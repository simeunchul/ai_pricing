"""한화투자증권 ELS 공시 추가 표본 다운로드 (1 → 다수).

이미 받아둔 8286호 외에 최근 3개월 한화투자증권 ELS 공시 1~2건을 추가 다운로드.
Universe 를 늘려서 ±3% 검증의 표본 N 강화.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dart_fetch import get_api_key  # noqa: E402


HANWHA_INVESTMENT_CORP_CODE = "00148610"  # 한화투자증권 (003530)


def list_hanwha_els(days: int = 90, limit: int = 100) -> list[dict]:
    """한화투자증권 최근 N일 공시 목록 (전체)."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    r = requests.get("https://opendart.fss.or.kr/api/list.json", params={
        "crtfc_key": get_api_key(),
        "corp_code": HANWHA_INVESTMENT_CORP_CODE,
        "bgn_de": start, "end_de": end,
        "page_no": 1, "page_count": limit,
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "000":
        print(f"[dart] error: {data.get('status')} {data.get('message')}")
        return []
    return data.get("list", [])


def fetch_document_xml(rcept_no: str, save_path: Path) -> bool:
    """문서 XML 다운로드 — ZIP 으로 오면 자동 해제."""
    import io
    import zipfile

    r = requests.get("https://opendart.fss.or.kr/api/document.xml", params={
        "crtfc_key": get_api_key(),
        "rcept_no": rcept_no,
    }, timeout=30)
    if r.status_code != 200 or len(r.content) < 1000:
        return False
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if r.content[:2] == b"PK":
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            inner = [n for n in z.namelist() if n.endswith(".xml")]
            if inner:
                save_path.write_bytes(z.read(inner[0]))
                return True
        except Exception as e:
            print(f"    unzip fail: {e}")
            return False
    save_path.write_bytes(r.content)
    return True


def main():
    print("=" * 70)
    print("  한화투자증권 ELS 공시 추가 표본 다운로드")
    print("=" * 70)

    print("\n[1] 한화투자증권 최근 90일 공시 목록 ...")
    discs = list_hanwha_els(days=90, limit=100)
    print(f"    {len(discs)}건")

    els_discs = [d for d in discs if "ELS" in d.get("report_nm", "") or "증권발행실적" in d.get("report_nm", "") or "ELB" in d.get("report_nm", "")]
    print(f"    ELS/ELB/발행실적 관련: {len(els_discs)}건")

    seen_rcept = {"20260424000266"}
    candidates = [d for d in els_discs if d.get("rcept_no") not in seen_rcept][:8]

    print(f"\n[2] 후보 {len(candidates)}건 (이미 보유한 8286 제외):")
    for d in candidates:
        print(f"    rcept={d.get('rcept_no')}  date={d.get('rcept_dt')}  {d.get('report_nm')[:60]}")

    # 다운로드 — 최대 3건 시도
    out_dir = ROOT / "data" / "els_samples"
    downloaded = []
    for d in candidates[:3]:
        rcept = d.get("rcept_no")
        save = out_dir / f"dart_{rcept}.xml"
        if save.exists():
            print(f"  [{rcept}] already cached, skip download")
        else:
            print(f"  [{rcept}] downloading ...")
            ok = fetch_document_xml(rcept, save)
            if not ok:
                print(f"    FAIL")
                continue
            print(f"    OK ({save.stat().st_size:,} bytes)")
        downloaded.append({
            "rcept_no": rcept,
            "rcept_dt": d.get("rcept_dt"),
            "report_nm": d.get("report_nm"),
            "xml_path": str(save.relative_to(ROOT)),
        })

    print(f"\n[3] {len(downloaded)}건 확보. 메타데이터 저장 ...")
    meta = ROOT / "data" / "els_samples" / "additional_samples.json"
    meta.write_text(json.dumps(downloaded, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"    → {meta}")

    return downloaded


if __name__ == "__main__":
    main()
