@echo off
cd /d C:\work\drone_lab
set CKPT=C:\IsaacLab\logs\rsl_rl\drone_racing_direct\2026-11-25_20-39-46\model_799.pt
call C:\IsaacLab\_isaac_sim\python.bat -u play_racing.py --checkpoint %CKPT% --steps 220 --out C:\work\drone_lab\out\videos --enable_cameras --headless > C:\temp\jobs\play.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\play.log