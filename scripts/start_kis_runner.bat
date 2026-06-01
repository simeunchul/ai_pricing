@echo off
REM === DISABLED 2026-05-07 ===
REM ETF iNAV arb 봇은 비활성화. dual_paper_trading 만 운영.
REM 작업 스케줄러는 그대로지만 이 .bat 가 inert 상태라 매매 발생 X.
REM
REM 다시 활성화하려면: 이 파일을 git history (이전 버전) 에서 복원
REM 또는 작업 스케줄러 자체를 admin 권한으로:
REM   schtasks /Change /TN "KIS_Paper_Trading_Daily" /DISABLE
REM
REM 원래 명령 (참고용 — 실행 안 됨):
REM   python scripts\run_kis_paper_trading.py --enter-bps 10 --exit-bps 1 ^
REM       --max-position 200 --qty-per-step 50 ^
REM       --symbols 069500,102110,152100,278530,105190,229200,091160,305720,266420
chcp 65001 > nul
cd /d d:\simeunchul\ai_pricing
echo [%date% %time%] start_kis_runner.bat is DISABLED (ETF arb halted) >> data\runner_stdout.log
exit /b 0
