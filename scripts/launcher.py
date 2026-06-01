"""자동매매 시스템 — 통합 Launcher (Tkinter GUI).

원클릭 시작/정지 + 종목 설정 + API 키 wizard.
이 파일을 PyInstaller 로 빌드하면 start.exe 생성 가능.

Usage:
  python scripts/launcher.py
  또는 (build 후)
  start.exe (더블클릭)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
PYTHON = sys.executable    # 현재 Python (또는 portable 의 python.exe)


PROGRAMS = {
    "kis_runner": {
        "label": "주식 Dual Runner (외인+기관 신호)",
        "cmd": [PYTHON, str(ROOT / "scripts" / "run_dual_paper_trading.py"),
                "--watch", "--interval", "300", "--off-hours-skip"],
        "cwd": str(ROOT),
        "process": None,
    },
    "dashboard": {
        "label": "Dashboard (http://localhost:8501)",
        "cmd": [PYTHON, "-m", "streamlit", "run",
                str(ROOT / "scripts" / "dual_dashboard.py"),
                "--server.headless", "true", "--server.port", "8501"],
        "cwd": str(ROOT),
        "process": None,
    },
    "crypto_runner": {
        "label": "Crypto Runner (Binance Testnet)",
        "cmd": [PYTHON, str(ROOT / "scripts" / "run_crypto_per_symbol.py"),
                "--duration-hours", "24", "--poll-min", "30"],
        "cwd": str(ROOT),
        "process": None,
    },
}


def is_running(prog_key: str) -> bool:
    p = PROGRAMS[prog_key]["process"]
    return p is not None and p.poll() is None


def start_program(prog_key: str) -> bool:
    if is_running(prog_key):
        return True
    p = PROGRAMS[prog_key]
    try:
        # CREATE_NO_WINDOW (Windows) 로 콘솔 창 안 뜨게
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(
            p["cmd"], cwd=p["cwd"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        p["process"] = proc
        return True
    except Exception as e:
        messagebox.showerror(f"{p['label']} 시작 실패", str(e))
        return False


def stop_program(prog_key: str):
    p = PROGRAMS[prog_key]
    proc = p["process"]
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass
    p["process"] = None


def get_current_mode() -> dict:
    """현재 .env 의 모드 읽기."""
    if not ENV_PATH.exists():
        return {"kis_env": None, "binance_env": None, "kis_dry": True, "bnc_dry": True}
    env = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return {
        "kis_env": env.get("KIS_ENV", "vts"),
        "binance_env": env.get("BINANCE_ENV", "testnet"),
        "kis_dry": env.get("KIS_DRY_RUN", "false").lower() == "true",
        "bnc_dry": env.get("BINANCE_DRY_RUN", "false").lower() == "true",
    }


def main():
    root = tk.Tk()
    root.title("자동매매 시스템")
    root.geometry("560x600")
    root.resizable(False, False)

    mode = get_current_mode()
    has_live = (mode["kis_env"] == "prod") or (mode["binance_env"] == "mainnet")
    header_color = "#c0392b" if has_live else "#2c5aa0"

    # Header
    header = tk.Frame(root, bg=header_color, pady=14)
    header.pack(fill="x")
    tk.Label(header, text="📊 자동매매 시스템",
              bg=header_color, fg="white",
              font=("맑은 고딕", 16, "bold")).pack()
    if has_live:
        tk.Label(header, text="🔴 LIVE 실거래 모드 (진짜 돈)",
                  bg=header_color, fg="#fff3cd",
                  font=("맑은 고딕", 10, "bold")).pack()
    else:
        tk.Label(header, text="🟢 모의투자 모드 (가짜 자금)",
                  bg=header_color, fg="#cce0ff",
                  font=("맑은 고딕", 9)).pack()

    # 모드 상세 표시
    mode_frame = tk.Frame(root, bg="#f4f4f4", pady=8)
    mode_frame.pack(fill="x")

    def _mode_label(name, env, dry):
        if env is None:
            return f"{name}: 미설정"
        is_live = env in ("prod", "mainnet")
        emoji = "🔴" if is_live else "🟢"
        env_str = {"vts": "모의", "prod": "실거래",
                    "testnet": "Testnet", "mainnet": "Mainnet"}.get(env, env)
        dry_str = " [DRY]" if dry else ""
        return f"{emoji} {name}: {env_str}{dry_str}"

    tk.Label(mode_frame,
              text=f"  {_mode_label('KIS 주식', mode['kis_env'], mode['kis_dry'])}    "
                   f"|    {_mode_label('Binance 코인', mode['binance_env'], mode['bnc_dry'])}",
              bg="#f4f4f4", font=("맑은 고딕", 9)).pack()

    # API key check
    if not ENV_PATH.exists():
        warn = tk.Frame(root, bg="#fff3cd", pady=8)
        warn.pack(fill="x")
        tk.Label(warn, text="⚠️ .env 없음 — 먼저 API 키 설정 필요",
                 bg="#fff3cd", fg="#856404",
                 font=("맑은 고딕", 9, "bold")).pack()

    # Status panel
    status_frame = tk.LabelFrame(root, text="실행 상태", padx=15, pady=10,
                                   font=("맑은 고딕", 10))
    status_frame.pack(fill="x", padx=20, pady=15)

    status_labels = {}
    for key, info in PROGRAMS.items():
        row = tk.Frame(status_frame)
        row.pack(fill="x", pady=4)
        emoji_label = tk.Label(row, text="⚪", font=("맑은 고딕", 12))
        emoji_label.pack(side="left", padx=(0, 8))
        tk.Label(row, text=info["label"],
                 font=("맑은 고딕", 9), anchor="w").pack(side="left", fill="x", expand=True)
        status_labels[key] = emoji_label

    def update_status():
        while True:
            for key, lbl in status_labels.items():
                lbl.config(text="🟢" if is_running(key) else "⚪")
            time.sleep(1)

    threading.Thread(target=update_status, daemon=True).start()

    # Action buttons
    btn_frame = tk.Frame(root, pady=10)
    btn_frame.pack()

    def start_all():
        if not ENV_PATH.exists():
            if messagebox.askyesno("API 키 설정 필요",
                ".env 파일이 없습니다.\nAPI 키 설정 wizard 를 먼저 실행할까요?"):
                run_wizard()
            return

        # 실거래 모드면 강력 confirm
        cur_mode = get_current_mode()
        live_warnings = []
        if cur_mode["kis_env"] == "prod" and not cur_mode["kis_dry"]:
            live_warnings.append("• KIS 실거래 — 한국 주식 진짜 돈 매매")
        if cur_mode["binance_env"] == "mainnet" and not cur_mode["bnc_dry"]:
            live_warnings.append("• Binance Mainnet — 코인 진짜 돈 매매")

        if live_warnings:
            confirm = messagebox.askyesno(
                "⚠️ 실거래 시작 확인",
                "다음 모드로 자동매매가 시작됩니다:\n\n"
                + "\n".join(live_warnings)
                + "\n\n• 손실 시 사용자 본인 책임\n"
                  "• 매일 점검 권장\n"
                  "• Stop-loss 자동 X (signal exit 만)\n\n"
                  "정말 시작하시겠습니까?",
                icon="warning",
            )
            if not confirm:
                return
            # 한 번 더
            confirm2 = messagebox.askyesno(
                "최종 확인",
                "실거래 자동매매를 정말 시작합니까?\n\n"
                "[예] = 즉시 매매 시작\n"
                "[아니오] = 취소",
                icon="warning",
            )
            if not confirm2:
                return

        for key in PROGRAMS:
            start_program(key)
        time.sleep(2)
        webbrowser.open("http://localhost:8501")

    def stop_all():
        for key in PROGRAMS:
            stop_program(key)

    def run_wizard():
        wizard_path = ROOT / "scripts" / "setup_wizard.py"
        try:
            subprocess.run([PYTHON, str(wizard_path)], cwd=str(ROOT))
        except Exception as e:
            messagebox.showerror("Wizard 실행 실패", str(e))

    def open_dashboard():
        if not is_running("dashboard"):
            if messagebox.askyesno("Dashboard 미실행",
                "Dashboard 가 실행 중이 아닙니다. 시작할까요?"):
                start_program("dashboard")
                time.sleep(3)
        webbrowser.open("http://localhost:8501")

    def open_config_folder():
        cfg_dir = ROOT / "config"
        cfg_dir.mkdir(exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(cfg_dir))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(cfg_dir)])
        else:
            subprocess.run(["xdg-open", str(cfg_dir)])

    btn_style = {"font": ("맑은 고딕", 10, "bold"), "width": 18, "pady": 8}

    tk.Button(btn_frame, text="▶  모두 시작",
              bg="#0b8043", fg="white",
              command=start_all, **btn_style).grid(row=0, column=0, padx=5, pady=4)
    tk.Button(btn_frame, text="■  모두 정지",
              bg="#c0392b", fg="white",
              command=stop_all, **btn_style).grid(row=0, column=1, padx=5, pady=4)
    tk.Button(btn_frame, text="📊  대시보드 열기",
              bg="#2c5aa0", fg="white",
              command=open_dashboard, **btn_style).grid(row=1, column=0, padx=5, pady=4)
    tk.Button(btn_frame, text="🔑  API 키 설정",
              bg="#bf8700", fg="white",
              command=run_wizard, **btn_style).grid(row=1, column=1, padx=5, pady=4)
    tk.Button(btn_frame, text="⚙  종목 설정 폴더",
              bg="#6c3483", fg="white",
              command=open_config_folder, **btn_style).grid(row=2, column=0, padx=5, pady=4)
    tk.Button(btn_frame, text="❌  종료",
              bg="#888", fg="white",
              command=lambda: (stop_all(), root.destroy()),
              **btn_style).grid(row=2, column=1, padx=5, pady=4)

    # Footer
    footer = tk.Frame(root, bg="#f4f4f4", pady=8)
    footer.pack(fill="x", side="bottom")
    tk.Label(footer, text="첫 실행: API 키 설정 → 모두 시작",
              bg="#f4f4f4", fg="#666",
              font=("맑은 고딕", 8)).pack()
    tk.Label(footer, text="대시보드에서 종목 화이트/블랙리스트 편집 가능",
              bg="#f4f4f4", fg="#666",
              font=("맑은 고딕", 8)).pack()

    # Cleanup on close
    def on_close():
        if any(is_running(k) for k in PROGRAMS):
            if messagebox.askyesno("종료 확인",
                "실행 중인 프로그램이 있습니다. 모두 정지 후 종료할까요?"):
                stop_all()
                root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
