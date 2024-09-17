#!/bin/bash

# Activate the virtual environment if it exists, otherwise create it
if [ -d "venv" ]; then
    echo "Activating the virtual environment..."
    source venv/bin/activate
else
    echo "Creating a new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
echo "Installing required packages from requirements.txt..."
if ! pip install -r requirements.txt; then
    echo "Some packages failed to install. Attempting manual installation..."
    
    # Install individual packages to handle errors
    packages=("python-dotenv" "openai" "moviepy" "pydub" "Pillow" "requests" "pyperclip")
    
    for pkg in "${packages[@]}"; do
        if ! pip show "$pkg" > /dev/null 2>&1; then
            echo "Attempting to install $pkg..."
            pip install "$pkg"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install $pkg. Please install it manually."
            else
                echo "$pkg installed successfully."
            fi
        else
            echo "$pkg is already installed."
        fi
    done
fi

# Check for OS-specific dependencies (tkinter for Ubuntu)
if ! dpkg -l | grep -q python3-tk; then
    echo "Installing tkinter for Ubuntu..."
    sudo apt-get update
    sudo apt-get install -y python3-tk
fi

# Confirm completion
echo "Setup complete. Would you like to launch the script now? (y/n)"
read launch_now
if [ "$launch_now" == "y" ]; then
    echo "Launching the main Python script..."
    python TemporalPromptEngine.py
else
    echo "Setup completed. You can run the script later with './launch.sh'."
fi
