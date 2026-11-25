@echo off
cd /d C:\work\drone_lab
echo CWD=%CD% > C:\temp\jobs\dbg.log
call C:\IsaacLab\_isaac_sim\python.bat -u lab3_debug.py --num_envs 2 --headless >> C:\temp\jobs\dbg.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\dbg.log