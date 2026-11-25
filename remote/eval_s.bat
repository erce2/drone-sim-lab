@echo off
cd /d C:\work\drone_lab
call C:\IsaacLab\_isaac_sim\python.bat -u eval_racing.py --controller scripted --num_envs 32 --steps 800 --out C:\work\drone_lab\out > C:\temp\jobs\evals.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\evals.log