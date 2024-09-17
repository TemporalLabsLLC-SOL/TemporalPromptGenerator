@echo off
SET VENV_DIR=venv

:: Activate the virtual environment and run the script
call %VENV_DIR%\Scripts\activate
echo Launching the main Python script...
python TemporalPromptEngine.py

pause
