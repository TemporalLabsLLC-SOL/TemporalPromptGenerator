@echo off
REM Navigate to the directory of the batch file
cd /d "%~dp0"

REM Define the path to the virtual environment
set VENV_PATH=.\CogVideo\inference\gradio_composite_demo\CogVx

REM Check if the virtual environment exists
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo Virtual environment not found. Creating a new virtual environment...
    python -m venv "%VENV_PATH%"
)

REM Activate the virtual environment
call "%VENV_PATH%\Scripts\activate.bat"

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required packages
echo Installing required packages...
python -m pip install torch diffusers transformers tk tqdm numpy tiktoken sentencepiece accelerate safetensors scipy pillow

REM Run TemporalCog-5b-FLUX.py
echo Running TemporalCog-5b-FLUX.py...
python TemporalCog-5b-FLUX.py

REM Deactivate the virtual environment
deactivate
