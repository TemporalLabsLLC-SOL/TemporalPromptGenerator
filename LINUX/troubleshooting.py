
import sys
import os
import importlib

# List of required packages for the application
required_packages = [
    "dotenv", "openai", "tkinter", "moviepy", "pydub", "Pillow", "requests", "pyperclip", "elevenlabs"
]

def check_python_version():
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("Warning: Python version is lower than 3.7, please update.")

def check_env_variables():
    print("Checking environment variables...")
    env_vars = ['ELEVENLABS_API_KEY', 'OPENAI_API_KEY', 'OUTPUT_DIRECTORY', 'COMFYUI_PROMPTS_FOLDER']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"{var}: Set")
        else:
            print(f"{var}: NOT SET")

def check_installed_packages():
    print("Checking installed packages...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"{package}: Installed")
        except ImportError:
            print(f"{package}: NOT INSTALLED")

if __name__ == "__main__":
    check_python_version()
    check_env_variables()
    check_installed_packages()
