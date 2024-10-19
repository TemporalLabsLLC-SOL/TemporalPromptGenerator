@echo off
REM Navigate to the CogVideo directory
cd /d "%~dp0"

REM Activate the CogVx virtual environment and run TemporalCog-5b.py
powershell -NoExit -Command "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\CogVx\Scripts\Activate.ps1; python TemporalCog-5b.py"
