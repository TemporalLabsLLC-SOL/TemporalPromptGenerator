#!/usr/bin/env python3
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import json
import platform
import re

def run_command(command, capture_output=False, cwd=None):
    """
    Run a system command with enhanced logging.
    """
    print(f"\nRunning command: {command}")
    try:
        if capture_output:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd
            )
            print(f"Command output:\n{result.stdout}")
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=True, cwd=cwd)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error Output: {e.stderr}")
        return None

def get_venv_executables(venv_dir):
    """
    Get paths to the Python and pip executables in the virtual environment.
    """
    python_executable = venv_dir / "bin" / "python"
    pip_executable = venv_dir / "bin" / "pip"
    return python_executable, pip_executable

def check_root():
    """
    Check if the script is run as root.
    """
    if os.geteuid() != 0:
        print("\nThis setup script must be run with root privileges. Please run using sudo:")
        print("sudo python3 setup.py")
        sys.exit(1)

def check_pip(cwd=None):
    """
    Check if pip is installed.
    """
    print("\nChecking for pip installation...")
    try:
        output = run_command("pip --version", capture_output=True, cwd=cwd)
        if output:
            print(f"pip is installed: {output}")
            return True
    except:
        pass
    print("pip is not installed.")
    return False

def install_pip(script_dir):
    """
    Install pip using ensurepip.
    """
    print("Attempting to install pip...")
    result = run_command(f'"{sys.executable}" -m ensurepip --upgrade', cwd=script_dir)
    if result is None:
        print("pip installed successfully.")
        return True
    else:
        print("Failed to install pip.")
        return False

def install_xvfb():
    """
    Install Xvfb and set up the virtual display environment.
    """
    print("\nInstalling Xvfb...")
    try:
        run_command("sudo apt-get install -y xvfb")
        print("Xvfb installed successfully.")
    except Exception as e:
        print(f"Failed to install Xvfb. Error: {e}")
        sys.exit(1)

def start_xvfb():
    """
    Start Xvfb on display :1 and set the DISPLAY environment variable.
    """
    print("\nStarting Xvfb on display :1...")
    try:
        run_command("Xvfb :1 -screen 0 1024x768x24 &")
        os.environ["DISPLAY"] = ":1"
        print("Xvfb started successfully with DISPLAY=:1")
    except Exception as e:
        print(f"Failed to start Xvfb. Error: {e}")
        sys.exit(1)

def install_system_dependencies_apt(script_dir):
    """
    Install necessary system dependencies on Linux using APT.
    """
    print("\nInstalling necessary system dependencies...")
    try:
        run_command("apt-get update")
        run_command("apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libfreetype6-dev liblcms2-dev libopenjp2-7-dev libwebp-dev")
        run_command("apt-get install -y xclip")
        run_command("apt-get install -y ffmpeg")
        run_command("apt-get install -y libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good")
        print("Installed system dependencies successfully.")
        return True
    except Exception as e:
        print(f"Failed to install system dependencies. Error: {e}")
        return False

def create_virtualenv(venv_dir):
    """
    Create a virtual environment.
    """
    print(f"\nCreating a virtual environment in '{venv_dir}'...")
    result = run_command(f'"{sys.executable}" -m venv "{venv_dir}"')
    if result is None:
        print("Virtual environment created successfully.")
        return True
    else:
        print("Failed to create virtual environment.")
        return False

def upgrade_pip(venv_dir):
    """
    Upgrade pip, setuptools, and wheel within the virtual environment.
    """
    print("\nUpgrading pip, setuptools, and wheel in the virtual environment...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False
    result = run_command(f'"{python_executable}" -m pip install --upgrade pip setuptools wheel', cwd=venv_dir)
    if result is None:
        print("pip, setuptools, and wheel upgraded successfully.")
        return True
    else:
        print("Failed to upgrade pip, setuptools, and wheel.")
        return False

def install_requirements(venv_dir, script_dir):
    """
    Install packages from requirements.txt using pip.
    """
    print("\nInstalling packages from requirements.txt...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    requirements_file = script_dir / "requirements.txt"
    if not requirements_file.exists():
        print(f"requirements.txt not found at {requirements_file}. Please create it with the necessary packages.")
        return False
    try:
        run_command(f'"{pip_executable}" install -r "{requirements_file}"', cwd=venv_dir)
        print("All packages installed successfully from requirements.txt.")
        return True
    except Exception as e:
        print(f"Failed to install packages from requirements.txt. Error: {e}")
        return False

def prompt_launch_script(venv_dir, main_script_path):
    """
    Prompt the user to launch the main Python script.
    """
    choice = input("\nSetup complete. Would you like to launch the script now? (y/n): ").strip().lower()
    if choice == 'y':
        print("\nLaunching the main Python script...")
        try:
            python_executable, _ = get_venv_executables(venv_dir)
            if not python_executable.exists():
                print(f"Virtual environment's Python executable not found at {python_executable}.")
                return
            subprocess.run([str(python_executable), str(main_script_path)], check=True)
            print("\nTemporalPromptEngine.py launched successfully.")
        except subprocess.CalledProcessError as e:
            print("\nFailed to launch TemporalPromptEngine.py. Please check the setup and try running the script manually.")
            print(f"Error: {e}")
    else:
        print("\nSetup completed. You can run the script later using the virtual environment's Python.")

def main():
    print("============================================")
    print("     Temporal Prompt Engine Setup Script    ")
    print("============================================\n")

    # Step 0: Ensure the script is run as root
    check_root()

    # Define the main script path relative to setup.py
    script_dir = Path(__file__).parent.resolve()
    main_script = "TemporalPromptEngine.py"
    main_script_path = script_dir / main_script

    if not main_script_path.exists():
        print(f"\nMain script '{main_script}' not found at {main_script_path}. Please ensure it exists.")
        sys.exit(1)

    # Step 1: Install system-level dependencies
    if not install_system_dependencies_apt(script_dir):
        print("\nSystem dependencies installation failed. Exiting setup.")
        sys.exit(1)

    # Step 2: Install Xvfb and set the virtual display
    install_xvfb()
    start_xvfb()

    # Step 3: Set up virtual environment
    venv_dir = script_dir / "TemporalPromptEngineEnv"
    if not venv_dir.exists():
        if not create_virtualenv(venv_dir):
            print("\nFailed to create virtual environment. Exiting setup.")
            sys.exit(1)

    # Step 4: Upgrade pip, setuptools, and wheel
    if not upgrade_pip(venv_dir):
        print("\nFailed to upgrade pip, setuptools, and wheel. Exiting setup.")
        sys.exit(1)

    # Step 5: Install requirements
    if not install_requirements(venv_dir, script_dir):
        print("\nFailed to install required packages. Exiting setup.")
        sys.exit(1)

    # Step 6: Prompt to launch the main script
    prompt_launch_script(venv_dir, main_script_path)

    print("\n============================================")
    print("         Setup Completed Successfully       ")
    print("============================================")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
