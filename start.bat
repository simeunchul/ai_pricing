@echo off
REM ===================================================
REM   자동매매 시스템 — Quick Start
REM ===================================================
REM 더블클릭 한 번으로 모든 처리:
REM   1. 의존성 체크 + 자동 설치
REM   2. .env 없으면 wizard 실행
REM   3. Launcher GUI 시작
REM ===================================================
chcp 65001 > nul
cd /d %~dp0

REM Python 체크
where python > nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 미설치
    echo.
    echo 다음 중 하나 설치하세요:
    echo   - Anaconda: https://www.anaconda.com/download
    echo   - Python:   https://www.python.org/downloads
    echo.
    echo 설치 후 이 .bat 다시 실행.
    pause
    exit /b 1
)

REM 의존성 빠른 체크 (streamlit 있으면 skip)
python -c "import streamlit, binance, pandas, plotly" 2>nul
if errorlevel 1 (
    echo.
    echo === 의존성 설치 (한 번만) ===
    python -m pip install -q -r release\requirements.txt
    if errorlevel 1 (
        echo [ERROR] 의존성 설치 실패. release\install.bat 수동 실행.
        pause
        exit /b 1
    )
)

REM .env 체크 — 없으면 wizard
if not exist .env (
    echo.
    echo === 처음 실행 — API 키 설정 wizard ===
    python scripts\setup_wizard.py
    if not exist .env (
        echo [INFO] .env 미생성 — 종료
        exit /b 0
    )
)

REM Launcher GUI 실행
python scripts\launcher.py
