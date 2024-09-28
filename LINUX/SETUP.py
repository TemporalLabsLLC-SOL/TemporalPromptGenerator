import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import json
import platform

def run_command(command, capture_output=False):
    """
    Run a system command.
    """
    try:
        if capture_output:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(command, check=True)
            return None
    except subprocess.CalledProcessError as e:
        if capture_output and e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error Output: {e.stderr}")
        return None


def check_pip():
    """
    Check if pip is installed.
    """
    try:
        output = run_command([sys.executable, "-m", "pip", "--version"], capture_output=True)
        if output:
            print(f"pip is installed: {output}")
            return True
    except:
        pass
    print("pip is not installed.")
    return False


def install_pip():
    """
    Install pip using ensurepip.
    """
    print("Attempting to install pip...")
    result = run_command([sys.executable, "-m", "ensurepip", "--upgrade"])
    if result is None:
        print("pip installed successfully.")
        return True
    else:
        print("Failed to install pip.")
        return False


def check_ollama():
    """
    Check if Ollama is installed.
    """
    try:
        output = run_command("ollama --version", capture_output=True)
        if output:
            print(f"Ollama is installed: {output}")
            return True
    except:
        pass
    print("Ollama is not installed.")
    return False

def prompt_install_ollama(download_url):
    """
    Prompt the user to install Ollama.
    """
    print("\nOllama is not installed. Please install it from the following link:")
    print(download_url)
    webbrowser.open(download_url)
    input("Press Enter after you have installed Ollama to continue...")
    # Recheck after installation
    return check_ollama()

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

def activate_virtualenv(venv_dir):
    """
    Activate the virtual environment.
    Note: Activation is handled when invoking the virtual environment's Python directly.
    This function only checks for the existence of the activation script.
    """
    activate_script = Path(venv_dir) / "bin" / "activate"
    if not activate_script.exists():
        print(f"Activation script not found at {activate_script}.")
        return False
    print("Virtual environment activation script found.")
    return True

def upgrade_pip(venv_dir):
    """
    Upgrade pip within the virtual environment.
    """
    print("\nUpgrading pip in the virtual environment...")
    pip_executable = Path(venv_dir) / "bin" / "pip3"
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False
    result = run_command(f'"{pip_executable}" install --upgrade pip')
    if result is None:
        print("pip upgraded successfully.")
        return True
    else:
        print("Failed to upgrade pip.")
        return False

def install_requirements(venv_dir, requirements_file):
    """
    Install packages from requirements.txt.
    """
    print(f"\nInstalling required packages from '{requirements_file}'...")
    pip_executable = Path(venv_dir) / "bin" / "pip"
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False
    if not Path(requirements_file).exists():
        print(f"\n'{requirements_file}' not found. Creating a default requirements file...")
        default_requirements = [
            "python-dotenv>=0.19.0",
            "openai>=0.27.0",
            "moviepy>=1.0.3",
            "pydub>=0.25.1",
            "Pillow>=9.0.0",
            "requests>=2.25.1",
            "pyperclip>=1.8.2",
            "torch>=1.13.1",
            "diffusers>=0.21.1",
            "transformers>=4.31.0",
            "accelerate>=0.21.0",  # Adding accelerate to handle memory
            "scipy>=1.7.0",
            "tk",  # For tkinter
            "audioldm2 @ git+https://github.com/haoheliu/AudioLDM2.git"
        ]
        with open(requirements_file, 'w') as f:
            for pkg in default_requirements:
                f.write(f"{pkg}\n")
        print(f"Default '{requirements_file}' created. Please review and modify it as needed.")
        input("Press Enter to continue...")
    result = run_command([str(pip_executable), "install", "-r", str(requirements_file)])
    if result is None:
        print("Required packages installed successfully.")
        return True
    else:
        print("Failed to install required packages.")
        return False


def install_torch(venv_dir):
    """
    Install torch if not already installed.
    """
    print("\nChecking for torch installation...")
    pip_executable = Path(venv_dir) / "bin" / "pip"
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False
    try:
        output = run_command([str(pip_executable), "show", "torch"], capture_output=True)
        if output:
            print("torch is already installed.")
            return True
    except:
        pass
    # Install torch
    print("torch is not installed. Installing torch...")
    # Determine the appropriate torch installation command based on CUDA availability
    try:
        cuda_available = False
        cuda_version = ""
        # Check if CUDA is available on the system
        result = run_command(["nvidia-smi"], capture_output=True)
        if result:
            cuda_available = True
            # Extract CUDA version
            for line in result.split('\n'):
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip()
                    break
    except:
        # If nvidia-smi is not found or fails, assume CPU-only installation
        cuda_available = False

    if cuda_available:
        # Install torch with CUDA support
        install_command = [str(pip_executable), "install", "torch", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu118"]
    else:
        # Install CPU-only version
        install_command = [str(pip_executable), "install", "torch", "torchvision", "torchaudio"]

    result = run_command(install_command)
    if result is None:
        print("torch installed successfully.")
        return True
    else:
        print("Failed to install torch.")
        print("Please install it manually from the following link:")
        print("https://pytorch.org/get-started/locally/")
        return False


def install_diffusers_from_github(venv_dir):
    """
    Install the latest diffusers from GitHub to ensure compatibility.
    """
    print("\nInstalling diffusers from GitHub to ensure latest features and compatibility...")
    pip_executable = Path(venv_dir) / "bin" / "pip"
    install_command = [str(pip_executable), "install", "git+https://github.com/huggingface/diffusers.git@main#egg=diffusers"]
    try:
        run_command(install_command)
        print("diffusers installed successfully from GitHub.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install diffusers from GitHub: {e}")
        print("Please install it manually using the following command:")
        print(' '.join(install_command))
        return False
    except Exception as e:
        print(f"An unexpected error occurred while installing diffusers: {e}")
        print("Please install it manually using the following command:")
        print(' '.join(install_command))
        return False


def install_transformers_from_github(venv_dir):
    """
    Install the latest transformers from GitHub to ensure compatibility and access to VitsModel.
    """
    print("\nInstalling transformers from GitHub to ensure latest features and compatibility...")
    pip_executable = Path(venv_dir) / "bin" / "pip"
    install_command = [str(pip_executable), "install", "git+https://github.com/huggingface/transformers.git@main#egg=transformers"]
    try:
        run_command(install_command)
        print("transformers installed successfully from GitHub.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install transformers from GitHub: {e}")
        print("Please install it manually using the following command:")
        print(' '.join(install_command))
        return False
    except Exception as e:
        print(f"An unexpected error occurred while installing transformers: {e}")
        print("Please install it manually using the following command:")
        print(' '.join(install_command))
        return False


def create_env_file():
    """
    Create a default .env file if it doesn't exist.
    """
    env_path = Path('.env')
    if env_path.exists():
        print("\n.env file already exists. Skipping creation.")
        return
    print("\nCreating a default .env file...")
    default_env = {
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "COMFYUI_PROMPTS_FOLDER": "path_to_comfyui_prompts_folder",
        "LAST_USED_DIRECTORY": "path_to_last_used_directory"
    }
    with open(env_path, 'w') as f:
        for key, value in default_env.items():
            f.write(f"{key}={value}\n")
    print(".env file created with placeholder values. Please edit it to include your actual configuration.")

def create_default_json_files():
    """
    Create default JSON configuration files if they don't exist or are empty.
    """
    json_files = {
        "settings.json": {
            "setting1": "default_value1",
            "setting2": "default_value2",
            "setting3": 123
        },
        "video_options.json": {
            "option1": "value1",
            "option2": "value2",
            "option3": True
        }
    }

    for file_name, default_content in json_files.items():
        file_path = Path(file_name)
        if not file_path.exists() or file_path.stat().st_size == 0:
            print(f"\nCreating default '{file_name}'...")
            with open(file_path, 'w') as f:
                json.dump(default_content, f, indent=4)
            print(f"'{file_name}' created with default content.")
        else:
            # Optionally, validate JSON structure here
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                print(f"'{file_name}' exists and contains valid JSON.")
            except json.JSONDecodeError:
                print(f"\n'{file_name}' is corrupted or contains invalid JSON. Recreating with default content...")
                with open(file_path, 'w') as f:
                    json.dump(default_content, f, indent=4)
                print(f"'{file_name}' has been recreated with default content.")

def install_system_dependencies():
    """
    Install necessary system dependencies using the package manager.
    """
    print("\nInstalling system dependencies...")
    package_manager = ""
    install_command = ""

    # Detect package manager
    if os.path.exists('/usr/bin/apt'):
        package_manager = 'apt'
    elif os.path.exists('/usr/bin/dnf'):
        package_manager = 'dnf'
    elif os.path.exists('/usr/bin/yum'):
        package_manager = 'yum'
    elif os.path.exists('/usr/bin/pacman'):
        package_manager = 'pacman'
    else:
        print("Unsupported package manager. Please install the following dependencies manually:")
        print("ffmpeg, xclip (or xsel), python3-tk")
        return

    if package_manager == 'apt':
        install_command = 'sudo apt update && sudo apt install -y ffmpeg xclip python3-tk'
    elif package_manager == 'dnf':
        install_command = 'sudo dnf install -y ffmpeg xclip python3-tkinter'
    elif package_manager == 'yum':
        install_command = 'sudo yum install -y ffmpeg xclip python3-tkinter'
    elif package_manager == 'pacman':
        install_command = 'sudo pacman -Sy --noconfirm ffmpeg xclip tk'

    try:
        run_command(install_command)
        print("System dependencies installed successfully.")
    except Exception as e:
        print(f"Failed to install system dependencies: {e}")
        print("Please install the following packages manually:")
        print("ffmpeg, xclip (or xsel), python3-tk")

def prompt_launch_script(venv_dir, main_script_path):
    """
    Prompt the user to launch the main Python script.
    """
    choice = input("\nSetup complete. Would you like to launch the script now? (y/n): ").strip().lower()
    if choice == 'y':
        print("\nLaunching the main Python script...")
        try:
            # Use the virtual environment's Python executable to run the main script
            python_executable = Path(venv_dir) / "bin" / "python3"
            if not python_executable.exists():
                print(f"Virtual environment's Python executable not found at {python_executable}.")
                print("Please ensure the virtual environment is set up correctly.")
                return
            subprocess.run([str(python_executable), str(main_script_path)], check=True)
            print("\nTemporalPromptEngine.py launched successfully.")
        except subprocess.CalledProcessError:
            print("\nFailed to launch TemporalPromptEngine.py. Please check the setup and try running the script manually.")
    else:
        print("\nSetup completed. You can run the script later using the virtual environment's Python interpreter.")

def main():
    print("============================================")
    print("     Temporal Prompt Engine Setup Script    ")
    print("============================================\n")

    # Define the main script path relative to setup.py
    setup_dir = Path(__file__).parent.resolve()
    main_script = "TemporalPromptEngine.py"
    main_script_path = setup_dir / main_script

    if not main_script_path.exists():
        print(f"\nMain script '{main_script}' not found at {main_script_path}. Please ensure it exists.")
        sys.exit(1)

    # Step 1: Install system dependencies
    install_system_dependencies()

    # Step 2: Check for pip
    if not check_pip():
        if not install_pip():
            print("\npip installation failed. Exiting setup.")
            sys.exit(1)

    # Step 3: Check for Ollama
    if not check_ollama():
        if not prompt_install_ollama("https://ollama.ai/downloads"):
            print("\nOllama installation not completed. Exiting setup.")
            sys.exit(1)

    # Step 4: Set up virtual environment
    venv_dir = "TemporalPromptEngineEnv"  # Updated virtual environment name
    if not Path(venv_dir).exists():
        if not create_virtualenv(venv_dir):
            print("\nFailed to create virtual environment. Exiting setup.")
            sys.exit(1)

    if not activate_virtualenv(venv_dir):
        print("\nFailed to locate virtual environment's activation script. Exiting setup.")
        sys.exit(1)

    # Step 5: Upgrade pip
    if not upgrade_pip(venv_dir):
        print("\nFailed to upgrade pip. Exiting setup.")
        sys.exit(1)

    # Step 6: Install required packages
    requirements_file = "requirements.txt"
    if not install_requirements(venv_dir, requirements_file):
        print("\nFailed to install required packages. Exiting setup.")
        sys.exit(1)

    # Step 7: Install diffusers and transformers from GitHub to ensure latest features and compatibility
    if not install_diffusers_from_github(venv_dir):
        print("\ndiffusers installation failed. Exiting setup.")
        sys.exit(1)

    if not install_transformers_from_github(venv_dir):
        print("\ntransformers installation failed. Exiting setup.")
        sys.exit(1)

    # Step 8: Install torch
    if not install_torch(venv_dir):
        print("\ntorch installation failed. Exiting setup.")
        sys.exit(1)

    # Step 9: Create .env file
    create_env_file()

    # Step 10: Create or validate JSON configuration files
    create_default_json_files()

    # Step 11: Prompt to launch the main script
    prompt_launch_script(venv_dir, main_script_path)

    print("\n============================================")
    print("         Setup Completed Successfully       ")
    print("============================================")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
