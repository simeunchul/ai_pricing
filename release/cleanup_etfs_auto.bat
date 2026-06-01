@echo off
cd /d d:\simeunchul\ai_pricing
echo. >> data\cleanup_etfs.log
echo ====================================================== >> data\cleanup_etfs.log
echo [%date% %time%] cleanup_etfs auto run >> data\cleanup_etfs.log
echo ====================================================== >> data\cleanup_etfs.log
"C:\Users\user\anaconda3\python.exe" scripts\cleanup_etfs.py >> data\cleanup_etfs.log 2>&1
"C:\Users\user\anaconda3\python.exe" scripts\sync_state_with_kis.py >> data\cleanup_etfs.log 2>&1
exit /b 0
