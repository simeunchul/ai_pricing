@echo off
REM state ↔ KIS sync 유틸리티
REM Drift 발생 시 (orphan/ghost) 강제 동기화
chcp 65001 > nul
cd /d %~dp0\..

echo === State ↔ KIS Holdings Sync ===
echo.
echo [1] Dry-run (비교만, 변경 X)
echo [2] 실제 sync (state 정정)
echo [3] 취소
echo.

choice /C 123 /N /M "선택: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto APPLY
if errorlevel 1 goto DRYRUN

:DRYRUN
python scripts\sync_state_with_kis.py --dry-run
pause
exit /b 0

:APPLY
python scripts\sync_state_with_kis.py
echo.
echo Runner 재시작 권장 (state 갱신 반영).
pause
