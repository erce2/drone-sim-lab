@echo off
cd /d C:\IsaacLab
echo START_V2 >> C:\temp\jobs\train3.log
call C:\IsaacLab\_isaac_sim\python.bat -u C:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-DroneRacing-Direct-v0 --headless --num_envs 4096 --max_iterations 800 --logger tensorboard >> C:\temp\jobs\train3.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\train3.log