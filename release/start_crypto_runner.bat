@echo off
REM 코인 자동매매 — Per-Symbol Strategy + Trailing Stop
REM Binance Futures Testnet (가짜 USDT)
chcp 65001 > nul
cd /d %~dp0\..

echo === Binance Crypto Per-Symbol Runner ===
echo 종목별 Strategy:
echo   BTCUSDT  → B_trend           + Trail2%%
echo   ETHUSDT  → C_trend_long_only + Trail2%%
echo   SOLUSDT  → D_trend+funding   + Trail2%%
echo   AVAXUSDT → B_trend           + Trail3%%
echo   BNBUSDT  → D_trend+funding   + Trail2%%
echo.
echo Polling: 30분마다 / 24h 운영
echo Multi-asset margin: USDT + USDC pool
echo.
echo 종료: Ctrl+C
echo.

python scripts\run_crypto_per_symbol.py --duration-hours 24 --poll-min 30

pause
