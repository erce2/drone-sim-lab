@echo off
REM Lab 1: 冒烟测试. 用法: lab1.bat [num_envs]
set NUMENV=%1
if "%NUMENV%"=="" set NUMENV=64
cd /d C:\work\drone_lab
call C:\IsaacLab\_isaac_sim\python.bat -u lab1_smoke.py --num_envs %NUMENV% --headless > C:\temp\jobs\lab1.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\lab1.log