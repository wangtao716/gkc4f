@echo off
setlocal
cd /d "%~dp0"
set "GAME_PYTHON=D:\ProgramData\Anaconda3\envs\gkc3C\python.exe"
if exist "%GAME_PYTHON%" (
    "%GAME_PYTHON%" "%~dp0main.py" %*
) else (
    call conda run --no-capture-output -n gkc3C python "%~dp0main.py" %*
)
if errorlevel 1 pause
