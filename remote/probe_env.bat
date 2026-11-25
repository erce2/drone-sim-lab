@echo off
set PY=C:\IsaacLab\_isaac_sim\python.bat
echo === PYVER ===
call %PY% -c "import sys; print(sys.version)"
echo === VERSIONS ===
call %PY% -c "import isaaclab, isaaclab_tasks; print('isaaclab', isaaclab.__version__); print('pkg', isaaclab.__file__)"
echo === QUADCOPTER TREE ===
dir /s /b C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter
echo === QUADCOPTER REGISTRATION ===
type C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadcopter\__init__.py
echo === ENV HINTS ===
set | findstr /i "ISAAC_ ASSET CARB_ LOG"
echo === FREE SPACE ===
dir C:\ | findstr /i "bytes free"
echo DONE_PROBE