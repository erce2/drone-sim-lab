@echo off
cd /d C:\IsaacLab
echo START_TRAIN >> C:\temp\jobs\train.log
call C:\IsaacLab\_isaac_sim\python.bat -u C:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-DroneRacing-Direct-v0 --headless --num_envs 4096 --max_iterations 300 --logger tensorboard >> C:\temp\jobs\train.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\train.log