@echo off
REM .env 파일 셋업 — .env.example 복사
chcp 65001 > nul
cd /d %~dp0\..

if exist .env (
    echo [INFO] .env 이미 존재.
    echo 직접 편집하려면: notepad .env
    echo.
    choice /M "기존 .env 덮어쓸까요"
    if errorlevel 2 (
        echo 취소됨.
        pause
        exit /b 0
    )
)

copy /Y release\.env.example .env > nul
echo ✓ .env 생성됨 (release\.env.example 복사)
echo.
echo 이제 .env 파일을 편집하여 API 키 입력:
echo   notepad .env
echo.
echo 필수:
echo   KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT
echo   BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET
echo.
pause

notepad .env
