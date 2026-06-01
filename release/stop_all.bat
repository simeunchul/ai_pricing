@echo off
REM 모든 자동매매 프로그램 정지
chcp 65001 > nul

echo === 자동매매 시스템 정지 ===
echo.
echo 정지 대상:
echo   - run_dual_paper_trading.py  (KIS runner)
echo   - run_crypto_per_symbol.py   (Crypto runner)
echo   - streamlit (Dashboard)
echo.

powershell -NoProfile -Command "Get-WmiObject Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_dual_paper_trading|run_crypto_per_symbol|streamlit.*dashboard' } | ForEach-Object { Write-Host ('  Stopping PID=' + $_.ProcessId + ' ' + ($_.CommandLine.Substring(0, [Math]::Min(80, $_.CommandLine.Length)))); Stop-Process -Id $_.ProcessId -Force }"

echo.
echo === 정지 완료 ===
echo.
echo 다시 시작: start_all.bat
echo.
pause
