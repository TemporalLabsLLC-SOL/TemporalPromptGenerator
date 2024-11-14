@echo off
REM Upgrade pip for Python 3.10
py -3.10 -m pip install --upgrade pip

REM Install pywin32 for Python 3.10
py -3.10 -m pip install pywin32

REM Execute the PRE-FLIGHT-CHECK.sh script using Git Bash
"C:\Program Files\Git\bin\bash.exe" -c "./PRE-FLIGHT-CHECK.sh"
pause
