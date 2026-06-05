"""봇 watchdog — heartbeat 가 N초 이상 stale 이면 봇 자동 restart.

Windows 작업 스케줄러로 1분 또는 5분 간격 호출 권장.
  schtasks /Create /TN "Dual_Watchdog" /TR "<this script>" /SC MINUTE /MO 5
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Windows cp949 stdout 환경에서 ⚠ / → 같은 유니코드 문자 로깅 시 UnicodeEncodeError
# 발생 (2026-06-01 봇 restart 경로 진입 시 처음으로 관찰됨). 다른 스크립트들과
# 동일한 reconfigure 패턴 적용.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT = ROOT / "data" / "dual_heartbeat.txt"
LOG = ROOT / "data" / "watchdog.log"
PYTHON = r"C:\Users\user\anaconda3\python.exe"
STREAMLIT = r"C:\Users\user\anaconda3\Scripts\streamlit.exe"
RUNNER = ROOT / "scripts" / "run_dual_paper_trading.py"
DASHBOARD = ROOT / "scripts" / "dual_dashboard.py"
STALE_THRESHOLD = 600   # 10분 (interval 120s * 5 = 정상 갱신 주기의 5배)
DASHBOARD_PORT = 8501

# pythonw 부모에서 wmic/taskkill 같은 콘솔 자식을 띄우면 5분마다 콘솔 창이
# 깜빡인다. CREATE_NO_WINDOW 로 자식 콘솔 할당을 막는다.
CREATE_NO_WINDOW = 0x08000000


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _is_bot_running() -> bool:
    """tasklist 로 run_dual_paper_trading.py 실행 중인지 확인."""
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline,processid"],
            stderr=subprocess.DEVNULL, text=True,
            creationflags=CREATE_NO_WINDOW,
        )
        return "run_dual_paper_trading" in out
    except Exception:
        return False


def _start_bot() -> None:
    _log("봇 재시작 시도...")
    try:
        creationflags = 0x00000200 | 0x08000000  # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        subprocess.Popen(
            [PYTHON, str(RUNNER), "--watch", "--interval", "120", "--off-hours-skip"],
            cwd=str(ROOT),
            stdout=open(ROOT / "data" / "dual_daemon_stdout.log", "ab"),
            stderr=open(ROOT / "data" / "dual_daemon_stderr.log", "ab"),
            creationflags=creationflags,
            close_fds=True,
        )
        _log("봇 재시작 명령 발행")
    except Exception as e:
        _log(f"봇 재시작 실패: {type(e).__name__}: {e}")


def _is_dashboard_running() -> bool:
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            stderr=subprocess.DEVNULL, text=True,
            creationflags=CREATE_NO_WINDOW,
        )
        return "dual_dashboard.py" in out or "streamlit" in out
    except Exception:
        return False


def _dashboard_health() -> bool:
    """localhost:8501 health endpoint 확인."""
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://127.0.0.1:{DASHBOARD_PORT}/_stcore/health",
                                    timeout=3)
        return r.status == 200
    except Exception:
        return False


def _kill_existing_dashboards() -> int:
    """기존 dashboard 프로세스(streamlit/dual_dashboard) 모두 종료.

    중복 spawn 방지: health 실패 후 _start_dashboard 호출 시 old 프로세스를
    먼저 정리해야 메모리 누수 + 포트 경합이 안 생긴다. 2026-06-04 관찰된
    streamlit 2개 동시 실행 사고 대응.
    """
    killed = 0
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get",
             "commandline,processid", "/format:csv"],
            stderr=subprocess.DEVNULL, text=True,
            creationflags=CREATE_NO_WINDOW,
        )
        for line in out.splitlines():
            if "dual_dashboard.py" not in line and "streamlit" not in line:
                continue
            parts = line.strip().split(",")
            if parts and parts[-1].isdigit():
                pid = parts[-1]
                try:
                    subprocess.run(
                        ["taskkill", "/PID", pid, "/F"],
                        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        creationflags=CREATE_NO_WINDOW,
                    )
                    killed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return killed


def _start_dashboard() -> None:
    # spawn 전 기존 dashboard 정리 — 중복 실행으로 인한 메모리 누수/포트 경합 차단
    n_killed = _kill_existing_dashboards()
    if n_killed > 0:
        _log(f"기존 dashboard 프로세스 {n_killed}개 정리")
        time.sleep(1.0)  # 포트 release 대기

    _log("dashboard 재시작 시도...")
    try:
        creationflags = 0x00000200 | 0x08000000
        subprocess.Popen(
            [PYTHON, STREAMLIT, "run", str(DASHBOARD),
             "--server.headless", "true", "--server.port", str(DASHBOARD_PORT),
             "--server.runOnSave", "true"],
            cwd=str(ROOT),
            stdout=open(ROOT / "data" / "dual_dashboard_stdout.log", "ab"),
            stderr=open(ROOT / "data" / "dual_dashboard_stderr.log", "ab"),
            creationflags=creationflags,
            close_fds=True,
        )
        _log("dashboard 재시작 명령 발행")
    except Exception as e:
        _log(f"dashboard 재시작 실패: {type(e).__name__}: {e}")


def main():
    # ===== Dashboard 점검 =====
    if not _dashboard_health():
        if not _is_dashboard_running():
            _log("⚠ dashboard 프로세스 없음 → 시작")
        else:
            _log("⚠ dashboard 프로세스는 있는데 health 실패 → restart 안 함 (다음 watchdog 에서 재평가)")
        _start_dashboard()
    else:
        _log("dashboard OK")

    # ===== 봇 점검 =====
    if not HEARTBEAT.exists():
        _log("heartbeat 파일 없음 — 봇 첫 실행 여부 확인")
        if not _is_bot_running():
            _log("봇 프로세스도 없음 → 시작")
            _start_bot()
        return

    age = time.time() - HEARTBEAT.stat().st_mtime
    running = _is_bot_running()

    if age < STALE_THRESHOLD and running:
        _log(f"OK · heartbeat {int(age)}s 전 · 봇 running")
        return

    if not running:
        _log(f"⚠ 봇 프로세스 없음 (heartbeat {int(age)}s 전) → restart")
        _start_bot()
        return

    if age >= STALE_THRESHOLD:
        _log(f"⚠ heartbeat stale {int(age)}s (> {STALE_THRESHOLD}s) — 봇 hang 의심 → 죽이고 restart")
        # 기존 프로세스 강제 종료
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where", "name='python.exe'", "get", "commandline,processid", "/format:csv"],
                stderr=subprocess.DEVNULL, text=True,
                creationflags=CREATE_NO_WINDOW,
            )
            for line in out.splitlines():
                if "run_dual_paper_trading" in line:
                    parts = line.strip().split(",")
                    if parts and parts[-1].isdigit():
                        pid = parts[-1]
                        subprocess.run(["taskkill", "/PID", pid, "/F"],
                                       stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL,
                                       creationflags=CREATE_NO_WINDOW)
                        _log(f"  killed PID {pid}")
        except Exception as e:
            _log(f"  kill 시도 실패: {e}")
        time.sleep(3)
        _start_bot()


if __name__ == "__main__":
    main()
