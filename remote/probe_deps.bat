@echo off
cd /d C:\work\drone_lab
call C:\IsaacLab\_isaac_sim\python.bat -u probe_deps.py > C:\temp\jobs\deps.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\deps.log