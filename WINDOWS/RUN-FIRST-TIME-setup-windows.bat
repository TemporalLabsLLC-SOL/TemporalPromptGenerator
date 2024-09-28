@echo off
SET VENV_DIR=TemporalPromptEngine

:: Check if virtual environment exists
IF EXIST %VENV_DIR% (
    echo Activating the virtual environment...
    call %VENV_DIR%\Scripts\activate
) ELSE (
    echo Creating a new virtual environment...
    python -m venv venv
    call %VENV_DIR%\Scripts\activate
)

:: Upgrade pip to the latest version
echo Upgrading pip...
pip install --upgrade pip

:: Install required packages from requirements.txt
echo Installing required packages from requirements.txt...
pip install -r requirements.txt

:: Check for common packages and manually install if necessary
echo Checking and installing additional required packages...
pip show python-dotenv >nul 2>&1 || pip install python-dotenv
pip show openai >nul 2>&1 || pip install openai
pip show moviepy >nul 2>&1 || pip install moviepy
pip show pydub >nul 2>&1 || pip install pydub
pip show Pillow >nul 2>&1 || pip install Pillow
pip show requests >nul 2>&1 || pip install requests
pip show pyperclip >nul 2>&1 || pip install pyperclip


:: Confirm completion
echo Setup complete. Would you like to launch the script now? (y/n)
set /p LAUNCH_NOW=
IF /I "%LAUNCH_NOW%"=="y" (
    echo Launching the main Python script...
    python TemporalPromptEngine.py
) ELSE (
    echo Setup completed. You can run the script later with launch.bat.
)

pause
