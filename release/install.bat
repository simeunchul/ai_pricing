@echo off
REM 자동매매 시스템 — 의존성 설치
chcp 65001 > nul
cd /d %~dp0\..

echo === Python 버전 확인 ===
python --version
if errorlevel 1 (
    echo [ERROR] Python 미설치 또는 PATH 등록 안 됨.
    echo Anaconda 또는 Python 3.10+ 설치 후 재실행.
    pause
    exit /b 1
)

echo.
echo === 의존성 설치 ===
python -m pip install --upgrade pip
python -m pip install -r release\requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] 의존성 설치 실패.
    pause
    exit /b 1
)

echo.
echo === 패키지 import 검증 ===
python -c "import pandas, numpy, requests, plotly, streamlit, binance, ccxt, yfinance; print('✓ 핵심 패키지 OK')"

echo.
echo === 설치 완료 ===
echo 다음 단계:
echo   1. setup_env.bat 실행 (또는 .env.example 을 .env 로 복사 후 키 입력)
echo   2. start_all.bat 실행
echo.
pause
