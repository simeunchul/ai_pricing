@echo off
REM 모두 시작 — 3개 프로그램 (KIS runner + Dashboard + Crypto runner)
chcp 65001 > nul
cd /d %~dp0\..

echo ===================================================
echo   자동매매 시스템 — 전체 시작
echo ===================================================
echo.
echo 시작될 프로그램:
echo   [1] KIS Dual Runner       (주식 — 외국인/기관 신호)
echo   [2] KIS Dashboard         (http://localhost:8501)
echo   [3] Binance Crypto Runner (코인 — 종목별 strategy)
echo.

REM .env 존재 확인
if not exist .env (
    echo [ERROR] .env 파일 없음.
    echo setup_env.bat 먼저 실행하세요.
    pause
    exit /b 1
)

echo === [1/3] KIS Dual Runner 시작 ===
start "KIS Dual Runner" cmd /k "cd /d %~dp0\.. && release\start_kis_runner.bat"
timeout /t 3 /nobreak > nul

echo === [2/3] KIS Dashboard 시작 ===
start "KIS Dashboard" cmd /k "cd /d %~dp0\.. && release\start_kis_dashboard.bat"
timeout /t 3 /nobreak > nul

echo === [3/3] Crypto Runner 시작 ===
start "Crypto Runner" cmd /k "cd /d %~dp0\.. && release\start_crypto_runner.bat"

echo.
echo ===================================================
echo   ✓ 3개 프로그램 모두 시작됨
echo ===================================================
echo.
echo 각 프로그램 별도 창에서 실행 중.
echo Dashboard: http://localhost:8501
echo.
echo 정지: stop_all.bat
echo.
pause
