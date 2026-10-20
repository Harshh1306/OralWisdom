@echo off
setlocal
if not defined ORALWISDOM_PYTHON set "ORALWISDOM_PYTHON=C:\OralWisdom_venv\Scripts\python.exe"
set "ORALWISDOM_RUNTIME_ROOT=%~dp0restoration_runtime"
set "ORALWISDOM_MODEL_PATH=%~dp0improved_site\models\best.pt"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
"%ORALWISDOM_PYTHON%" -B "%~dp0verify_release.py" --environment
if errorlevel 1 goto failed
"%ORALWISDOM_PYTHON%" -B "%~dp0improved_site\serve.py"
if errorlevel 1 goto failed
exit /b 0
:failed
echo Startup failed. Read the integrity or environment error above.
pause
exit /b 1
