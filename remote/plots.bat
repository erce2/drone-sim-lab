@echo off
cd /d C:\work\drone_lab
call C:\IsaacLab\_isaac_sim\python.bat -u make_plots.py --logdir C:\IsaacLab\logs\rsl_rl\drone_racing_direct\2026-11-25_20-39-46 --out C:\work\drone_lab\out > C:\temp\jobs\plots.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\plots.log