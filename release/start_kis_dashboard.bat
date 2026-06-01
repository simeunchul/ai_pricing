@echo off
REM 주식 자동매매 — 실시간 Dashboard
chcp 65001 > nul
cd /d %~dp0\..

echo === KIS Dual Confirmation Dashboard ===
echo URL: http://localhost:8501
echo 자동 새로고침 30초
echo.
echo 종료: Ctrl+C
echo.

REM 브라우저 자동 열기 (3초 후)
start "" /B cmd /c "timeout /t 3 /nobreak > nul && start http://localhost:8501"

streamlit run scripts\dual_dashboard.py --server.headless true --server.port 8501

pause
