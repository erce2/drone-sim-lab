@echo off
cd /d C:\IsaacLab
echo START_RESUME >> C:\temp\jobs\train2.log
call C:\IsaacLab\_isaac_sim\python.bat -u C:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-DroneRacing-Direct-v0 --headless --num_envs 4096 --max_iterations 1200 --resume --load_run 2026-11-25_20-29-47 --checkpoint model_299.pt --logger tensorboard >> C:\temp\jobs\train2.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\train2.log