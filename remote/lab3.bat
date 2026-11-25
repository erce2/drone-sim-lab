@echo off
cd /d C:\work\drone_lab
echo CWD=%CD% > C:\temp\jobs\lab3.log
call C:\IsaacLab\_isaac_sim\python.bat -u lab3_racing_smoke.py --num_envs 8 --steps 500 --headless >> C:\temp\jobs\lab3.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\lab3.log