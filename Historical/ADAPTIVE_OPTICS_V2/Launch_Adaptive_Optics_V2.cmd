@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHONW_EXE=C:\Users\arthu\anaconda3\envs\ao311\pythonw.exe"
set "PYTHON_EXE=C:\Users\arthu\anaconda3\envs\ao311\python.exe"

cd /d "%ROOT%"

if exist "%PYTHONW_EXE%" (
    start "" "%PYTHONW_EXE%" "%ROOT%ao_v2_launcher.py"
    goto :eof
)

if exist "%PYTHON_EXE%" (
    start "" "%PYTHON_EXE%" "%ROOT%ao_v2_launcher.py"
    goto :eof
)

echo Unable to find the ao311 Python interpreter.
pause
