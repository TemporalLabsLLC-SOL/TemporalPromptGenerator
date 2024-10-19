#!/bin/bash

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Windows 10 or later
if [[ $(uname -s) != *"NT"* ]]; then
    echo "Error: This script must be run in a Windows environment (using Git Bash or WSL)."
    exit 1
fi

# Activate TemporalPromptEngineEnv environment
if [ -d "$HOME/TemporalPromptEngineEnv" ]; then
    echo "Activating TemporalPromptEngineEnv..."
    source "$HOME/TemporalPromptEngineEnv/bin/activate"
else
    echo "Error: TemporalPromptEngineEnv not found. Please create the environment before proceeding."
    exit 1
fi

# Check Python installations
check_python_version() {
    version=$1
    expected=$2
    py_version=$(py -$version --version 2>&1)
    if [[ "$py_version" == *"Python $expected"* ]]; then
        echo "Python $expected is correctly installed."
    else
        echo "Error: Python $expected not found. Please install Python $expected."
    fi
}

echo "Checking Python installations..."
check_python_version 3.10 3.10.9
check_python_version 3.12 3.12.4

# Check Git installation
echo "Checking Git installation..."
if command_exists git; then
    echo "Git is installed."
else
    echo "Error: Git is not installed. Please install Git for Windows."
fi

# Check NVIDIA GPU and CUDA
if command_exists nvidia-smi; then
    echo "NVIDIA GPU detected."
    nvcc_version=$(nvcc --version 2>&1)
    if [[ "$nvcc_version" == *"release 11.8"* ]]; then
        echo "CUDA Toolkit 11.8 is correctly installed."
    else
        echo "Error: CUDA Toolkit 11.8 not found or incorrectly installed."
    fi
else
    echo "Error: No NVIDIA GPU detected or NVIDIA drivers not installed."
fi

# Check FFmpeg installation
echo "Checking FFmpeg installation..."
if command_exists ffmpeg; then
    echo "FFmpeg is installed."
else
    echo "Error: FFmpeg is not installed or not in PATH."
fi

# Check Ollama installation
echo "Checking Ollama installation..."
if command_exists ollama; then
    echo "Ollama is installed."
else
    echo "Error: Ollama is not installed. Please download from Ollama Setup."
fi

# Check HuggingFace API key setup
echo "Checking HuggingFace API key setup..."
if [ -f ~/.huggingface/token ]; then
    echo "HuggingFace API key is set up."
else
    echo "Error: HuggingFace API key not found. Please generate and add your API key."
fi

echo "
Setup verification complete. Please address any errors above to proceed."