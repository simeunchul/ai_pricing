@echo off
REM 주식 자동매매 — Dual Confirmation Runner
REM 외국인+기관 동방향 신호 → 매수/매도
chcp 65001 > nul
cd /d %~dp0\..

echo === KIS Dual Paper Trading Runner ===
echo 전략: 외국인+기관 둘 다 ±5%% 이상 → 매수/매도
echo Polling: 5분마다, 장중 (09:00~15:30 KST)
echo MDD cap 30%% / max 7종 / cooldown 10일
echo.
echo 종료: Ctrl+C
echo 로그: data\dual_trading.log
echo.

python scripts\run_dual_paper_trading.py --watch --interval 300 --off-hours-skip

pause
