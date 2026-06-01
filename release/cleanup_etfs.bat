@echo off
REM ETF 일괄 청산 — 보유 ETF 모두 시장가 매도
REM 장중 (09:00~15:30 KST) 에만 동작
chcp 65001 > nul
cd /d %~dp0\..

echo === ETF 일괄 청산 ===
echo.
echo [1] Dry-run (대상 확인만, 매도 X)
echo [2] 실제 청산 (시장가 매도)
echo [3] 취소
echo.

choice /C 123 /N /M "선택: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto APPLY
if errorlevel 1 goto DRYRUN

:DRYRUN
python scripts\cleanup_etfs.py --dry-run
pause
exit /b 0

:APPLY
python scripts\cleanup_etfs.py
echo.
echo Sync state 권장: release\sync_state.bat
pause
