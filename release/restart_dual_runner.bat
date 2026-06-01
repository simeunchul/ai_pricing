@echo off
cd /d d:\simeunchul\ai_pricing
echo. >> data\dual_restart.log
echo ====================================================== >> data\dual_restart.log
echo [%date% %time%] daily restart at 09:00 >> data\dual_restart.log
echo ====================================================== >> data\dual_restart.log

REM 기존 dual runner 정지
powershell -NoProfile -Command "Get-WmiObject Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_dual_paper_trading' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >> data\dual_restart.log 2>&1

timeout /t 3 /nobreak > nul

REM 재시작
echo Starting fresh runner @ %time% >> data\dual_restart.log
start "" /B "C:\Users\user\anaconda3\python.exe" scripts\run_dual_paper_trading.py --watch --interval 120 --off-hours-skip

exit /b 0
