@echo off
cd /d C:\work\drone_lab
set CKPT=C:\IsaacLab\logs\rsl_rl\drone_racing_direct\2026-11-25_20-39-46\model_799.pt
echo EVAL_RL_START >> C:\temp\jobs\eval.log
call C:\IsaacLab\_isaac_sim\python.bat -u eval_racing.py --controller rl --num_envs 32 --steps 1200 --checkpoint %CKPT% --out C:\work\drone_lab\out >> C:\temp\jobs\eval.log 2>&1
echo EVAL_SCRIPTED_START >> C:\temp\jobs\eval.log
call C:\IsaacLab\_isaac_sim\python.bat -u eval_racing.py --controller scripted --num_envs 32 --steps 1200 --out C:\work\drone_lab\out >> C:\temp\jobs\eval.log 2>&1
echo DONE_ALL >> C:\temp\jobs\eval.log
echo EXITCODE=%ERRORLEVEL% >> C:\temp\jobs\eval.log