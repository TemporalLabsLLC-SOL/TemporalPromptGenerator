@echo off
REM Navigate to the CogVideo directory
cd /d "%~dp0"

REM Activate the CogVx virtual environment and run TemporalCog-5b.py
.\CogVideo\inference\gradio_composite_demo\CogVx\Scripts\activate
python WatermarkVideos.py
