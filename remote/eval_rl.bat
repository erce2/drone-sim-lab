@echo off
cd /d C:\work\drone_lab
set CKPT=C:\IsaacLab\logs\rsl_rl\drone_racing_direct\2026-11-25_20-39-46\model_799.pt
call C:\IsaacLab\_isaac_sim\python.bat -u eval_racing.py --controller rl --num_envs 32 --steps 800 --checkpoint %CKPT% --out C:\work\drone_lab\out > C:\temp\jobs\evalrl.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\evalrl.log