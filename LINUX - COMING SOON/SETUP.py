import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import json
import platform
import shutil
import time
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def print_banner():
    banner = f"""
{Fore.CYAN}
============================================
     ✨ Temporal Lab: Time Portal Setup ✨     
============================================
"""
    print(banner)

def print_status(message):
    print(f"{Fore.GREEN}✔️ {message}{Style.RESET_ALL}")

def print_warning(message):
    print(f"{Fore.YELLOW}⚠️ {message}{Style.RESET_ALL}")

def print_error(message):
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def print_info(message):
    print(f"{Fore.BLUE}ℹ️ {message}{Style.RESET_ALL}")

def print_action(message):
    print(f"{Fore.MAGENTA}🔧 {message}{Style.RESET_ALL}")

def run_command(command, capture_output=False, cwd=None):
    """
    Run a system command with enhanced logging.
    """
    command_str = ' '.join(command) if isinstance(command, list) else command
    print_action(f"Executing: {command_str}")
    try:
        if capture_output:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd
            )
            output = result.stdout.strip()
            if output:
                print_info(f"Output: {output}")
            return output
        else:
            subprocess.run(command, check=True, cwd=cwd)
            return None
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with error code {e.returncode}")
        if e.stdout:
            print_error(f"Output: {e.stdout.strip()}")
        if e.stderr:
            print_error(f"Error Output: {e.stderr.strip()}")
        return None

def get_venv_executables(venv_dir):
    """
    Get paths to the Python and pip executables in the virtual environment.
    """
    if platform.system() == "Windows":
        python_executable = venv_dir / "Scripts" / "python.exe"
        pip_executable = venv_dir / "Scripts" / "pip.exe"
    else:
        python_executable = venv_dir / "bin" / "python3"
        pip_executable = venv_dir / "bin" / "pip3"
    return python_executable, pip_executable

def check_pip(cwd=None):
    """
    Check if pip is installed.
    """
    print_info("Initiating Pip Verification...")
    try:
        output = run_command(["pip3", "--version"], capture_output=True, cwd=cwd)
        if output:
            print_status(f"Pip is ready: {output}")
            return True
    except:
        pass
    print_warning("Pip is not detected.")
    return False

def install_pip(script_dir):
    """
    Install pip using ensurepip or system package manager.
    """
    print_action("Installing Pip...")
    try:
        # Try using ensurepip
        result = run_command([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=script_dir)
        if result is None:
            print_status("Pip has been successfully installed using ensurepip.")
            return True
    except:
        pass
    # If ensurepip fails, instruct the user to install pip via package manager
    print_warning("Failed to install pip using ensurepip.")
    if platform.system() == "Linux":
        print_info("Attempting to install pip using system package manager...")
        try:
            run_command(["sudo", "apt", "update"], cwd=script_dir)
            run_command(["sudo", "apt", "install", "-y", "python3-pip"], cwd=script_dir)
            print_status("Pip has been successfully installed via apt.")
            return True
        except:
            pass
    print_error("Failed to install pip automatically. Please install pip manually and rerun the setup.")
    return False

def check_ollama(cwd=None):
    """
    Check if Ollama is installed.
    """
    print_info("Verifying Ollama Connectivity...")
    try:
        output = run_command(["ollama", "--version"], capture_output=True, cwd=cwd)
        if output and "ollama version" in output.lower():
            print_status(f"Ollama is operational: {output}")
            return True
    except:
        pass
    print_warning("Ollama connection not established.")
    return False

def prompt_install_ollama(download_url, script_dir):
    """
    Prompt the user to install Ollama.
    """
    print_warning("\nOllama is essential for bridging our Time Portal.")
    print_info("Please install Ollama from the following gateway:")
    print(f"{download_url}")
    webbrowser.open(download_url)
    input("\n🕰️ Once Ollama is installed, press Enter to synchronize and continue...")
    # Recheck after installation
    return check_ollama(cwd=script_dir)

def create_virtualenv(venv_dir, python_command=None):
    """
    Create a virtual environment using the specified Python command.
    """
    print_action(f"Constructing Virtual Chamber at '{venv_dir}'...")
    if python_command:
        command = [python_command, "-m", "venv", str(venv_dir)]
    else:
        command = [sys.executable, "-m", "venv", str(venv_dir)]

    result = run_command(command)
    if result is None:
        print_status("Virtual Chamber established successfully.")
        return True
    else:
        print_error("Failed to construct the Virtual Chamber.")
        return False

def upgrade_pip(venv_dir):
    """
    Upgrade pip within the virtual environment.
    """
    print_action("Enhancing Pip within the Virtual Chamber...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print_warning(f"Pip executable not found at {pip_executable}.")
        return False
    result = run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"], cwd=venv_dir)
    if result is None:
        print_status("Pip has been upgraded to the latest version.")
        return True
    else:
        print_error("Pip upgrade encountered issues.")
        return False

def install_and_verify_package(package_name, version, venv_dir, install_command=None, import_name=None):
    """
    Install a package using pip from the virtual environment and verify its installation.
    """
    print_action(f"Integrating '{package_name}=={version}' into the Temporal Engine...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print_warning(f"Pip executable not found at {pip_executable}.")
        return False

    # Use custom install command if provided
    if install_command is None:
        install_cmd = [str(pip_executable), "install", f"{package_name}=={version}"]
    else:
        install_cmd = install_command

    # Install the package
    try:
        run_command(install_cmd, cwd=venv_dir)
        print_status(f"'{package_name}=={version}' integrated successfully.")
    except Exception as e:
        print_error(f"Failed to integrate '{package_name}=={version}'. Error: {e}")
        return False

    # Skip verification for specific packages
    skip_verification = [
        "wheel",
        "torch",
        "torchvision",
        "torchaudio",
        "huggingface-hub",
        "audioldm2",
    ]
    if package_name in skip_verification:
        print_info(f"Verification skipped for '{package_name}'.")
        return True

    # Verify installation by attempting to import the package
    try:
        # Use custom import name if provided
        if import_name is None:
            package_import_name = package_name.replace('-', '_')
            # Handle special cases
            special_cases = {
                'python_dotenv': 'dotenv',
                'pyyaml': 'yaml',
                'moviepy': 'moviepy.editor',
                'Pillow': 'PIL',
                'tk': 'tkinter',
                'audioldm2': 'audioldm2',
                'pydub': 'pydub'
            }
            package_import_name = special_cases.get(package_import_name, package_import_name)
        else:
            package_import_name = import_name

        command = [str(python_executable), "-c", f"import {package_import_name}"]
        run_command(command, cwd=venv_dir)
        print_status(f"'{package_name}' verified and operational.")
        return True
    except Exception as e:
        print_error(f"Verification failed for '{package_name}'. Error: {e}")
        return False

def clone_cogvideo_repo(cogvideo_repo_path):
    """
    Clone the CogVideo repository into the specified path.
    If the repository already exists, it will be removed and re-cloned.
    """
    if cogvideo_repo_path.exists():
        print_warning("\nCogVideo repository detected. Resetting to ensure a pristine environment...")
        shutil.rmtree(cogvideo_repo_path)
    print_info("\nInitiating CogVideo Repository Cloning...")
    run_command(["git", "clone", "https://github.com/THUDM/CogVideo", str(cogvideo_repo_path)])
    print_status("CogVideo repository cloned successfully.")

def copy_custom_scripts(custom_scripts_dir, gradio_demo_path):
    """
    Copy custom scripts from the user's directory to the gradio_composite_demo directory.
    """
    print_info("\nTransferring Custom Scripts to CogVideo Interface...")
    for script_name in ["TemporalCog-5b.py", "TemporalCog-2b.py", "PromptList2MP4Utility.py"]:
        source_file = custom_scripts_dir / script_name
        target_file = gradio_demo_path / script_name
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print_status(f"'{script_name}' successfully transferred.")
        else:
            print_warning(f"Custom script '{script_name}' not found in '{custom_scripts_dir}'. Skipping.")

def install_dependencies_for_cogvideo(cogvideo_gradio_demo_path, cogvx_venv_dir):
    """
    Install dependencies for CogVideo into the CogVx virtual environment.
    """
    print_action(f"Installing CogVideo Dependencies in '{cogvx_venv_dir}'...")
    python_executable, pip_executable = get_venv_executables(cogvx_venv_dir)

    # Install requirements
    requirements_path = cogvideo_gradio_demo_path / "requirements.txt"
    install_commands = [
        [str(pip_executable), "install", "-r", str(requirements_path)],
        [str(pip_executable), "install", "torch==1.13.1+cu117", "torchvision==0.14.1+cu117", "torchaudio==0.13.1+cu117", "--index-url", "https://download.pytorch.org/whl/cu117"],
        [str(pip_executable), "install", "moviepy==2.0.0.dev2"]
    ]

    for command in install_commands:
        try:
            run_command(command, cwd=cogvideo_gradio_demo_path)
            print_status(f"Dependency Command Executed: {' '.join(command)}")
        except Exception as e:
            print_error(f"Failed to execute command: {' '.join(command)}. Error: {e}")
            sys.exit(1)

    print_status("All CogVideo dependencies installed successfully.")

def setup_cogvx(cogvideo_gradio_demo_path, cogvx_venv_dir):
    """
    Create and set up the virtual environment for CogVideo using Python 3.9 if not already present.
    """
    if not cogvx_venv_dir.exists():
        print_info("\nCogVx Virtual Environment not detected. Creating with Python 3.9...")
        # Use 'python3.9' to create the virtual environment
        python_command = "python3.9"
        # Check if python3.9 exists
        if shutil.which(python_command) is None:
            print_error("Python 3.9 is not installed. Please install Python 3.9 to proceed.")
            print_info("You can install Python 3.9 using the following commands:")
            print("sudo apt update")
            print("sudo apt install python3.9 python3.9-venv")
            sys.exit(1)
        command = [python_command, "-m", "venv", str(cogvx_venv_dir)]
        result = run_command(command)
        if result is None:
            print_status("CogVx Virtual Environment established successfully.")
        else:
            print_error("Failed to establish CogVx Virtual Environment with Python 3.9. Halting setup.")
            sys.exit(1)
    else:
        print_status("CogVx Virtual Environment already operational.")

    # Upgrade pip within the CogVx environment
    upgrade_pip(cogvx_venv_dir)

    # Install CogVideo dependencies
    install_dependencies_for_cogvideo(cogvideo_gradio_demo_path, cogvx_venv_dir)

def create_env_file(script_dir):
    """
    Create a .env file with user-provided values if it doesn't exist.
    """
    env_path = script_dir / '.env'
    if env_path.exists():
        print_info("\nConfiguration File '.env' already exists. Skipping creation.")
        return

    print_info("\nSetting Up Configuration Parameters...")
    print_action("Creating '.env' configuration file.")

    # Ask the user for the COMFYUI_PROMPTS_FOLDER
    comfyui_prompts_folder = input("🌟 Please enter the full path to your ComfyUI prompts folder: ").strip()

    # Ask the user for the LAST_USED_DIRECTORY
    last_used_directory = input("🌟 Please enter the full path to your last used directory: ").strip()

    default_env = {
        "COMFYUI_PROMPTS_FOLDER": comfyui_prompts_folder,
        "LAST_USED_DIRECTORY": last_used_directory
    }

    with open(env_path, 'w') as f:
        for key, value in default_env.items():
            f.write(f"{key}={value}\n")

    print_status("Configuration File '.env' created successfully.")

def create_default_json_files(script_dir):
    """
    Create default JSON configuration files if they don't exist or are empty.
    """
    json_files = {
        "settings.json": {
            "resolution": "1920x1080",
            "frame_rate": 30,
            "bitrate": "5000k",
            "output_folder": str(script_dir / "outputs")
        },
        "video_options.json": {
            "codec": "h264",
            "format": "mp4",
            "audio_codec": "aac",
            "preset": "medium"
        }
    }

    for file_name, default_content in json_files.items():
        file_path = script_dir / file_name
        if not file_path.exists() or file_path.stat().st_size == 0:
            print_action(f"Generating default '{file_name}' configuration...")
            with open(file_path, 'w') as f:
                json.dump(default_content, f, indent=4)
            print_status(f"'{file_name}' has been initialized with default settings.")
        else:
            # Validate JSON structure
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                print_info(f"'{file_name}' is already configured and validated.")
            except json.JSONDecodeError:
                print_warning(f"'{file_name}' contains invalid data. Reinitializing with default settings...")
                with open(file_path, 'w') as f:
                    json.dump(default_content, f, indent=4)
                print_status(f"'{file_name}' has been refreshed with default configurations.")

def check_ffmpeg(cwd=None):
    """
    Check if ffmpeg is installed.
    """
    print_info("\nScanning for ffmpeg Installation...")
    try:
        result = run_command(["ffmpeg", "-version"], capture_output=True, cwd=cwd)
        if result and "ffmpeg version" in result.lower():
            print_status("ffmpeg is active and ready.")
            return True
    except:
        pass
    print_warning("ffmpeg is not detected.")
    return False

def prompt_install_ffmpeg(script_dir):
    """
    Prompt the user to install ffmpeg.
    """
    print_warning("\nffmpeg is crucial for our video processing capabilities.")
    print_info("Attempting to install ffmpeg via system package manager...")
    try:
        run_command(["sudo", "apt", "update"], cwd=script_dir)
        run_command(["sudo", "apt", "install", "-y", "ffmpeg"], cwd=script_dir)
        print_status("ffmpeg has been successfully installed via apt.")
        return True
    except:
        print_warning("Automatic installation of ffmpeg failed.")
    print_info("Please install ffmpeg manually using your system package manager.")
    return False

def create_desktop_shortcut(venv_dir, main_script_path):
    """
    Create a desktop shortcut to run the main script using the virtual environment.
    """
    print_info("\nSetting Up Desktop Shortcut for Quick Access...")
    # Only proceed if on Linux
    if platform.system() != "Linux":
        print_warning("Desktop shortcut creation is currently only supported on Linux.")
        return

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop):
        print_warning("Desktop directory not found. Shortcut creation skipped.")
        return

    shortcut_path = os.path.join(desktop, 'Temporal Prompt Engine.desktop')
    target = str(venv_dir / 'bin' / 'python3')
    working_directory = str(main_script_path.parent)
    arguments = f'"{main_script_path}"'

    # Create the .desktop file content
    desktop_entry = f"""[Desktop Entry]
Type=Application
Name=Temporal Prompt Engine
Exec={target} {arguments}
Path={working_directory}
Icon=utilities-terminal
Terminal=true
"""

    # Write the .desktop file
    with open(shortcut_path, 'w') as f:
        f.write(desktop_entry)

    # Make the shortcut executable
    os.chmod(shortcut_path, 0o755)

    print_status(f"Desktop shortcut created at {shortcut_path}")

def check_cuda_toolkit(cwd=None):
    """
    Check if CUDA toolkit is installed.
    """
    print_info("\nAssessing CUDA Toolkit Integration...")
    try:
        cuda_version_output = run_command(["nvcc", "--version"], capture_output=True, cwd=cwd)
        if cuda_version_output:
            # Parse the CUDA version from the output
            import re
            match = re.search(r"release (\d+\.\d+),", cuda_version_output)
            if match:
                cuda_version = match.group(1)
                print_status(f"CUDA Toolkit Version Detected: {cuda_version}")
                return cuda_version
            else:
                print_warning("Unable to decipher CUDA version from nvcc output.")
                return None
        else:
            print_warning("CUDA Toolkit not found.")
            return None
    except:
        print_warning("nvcc command not found. CUDA Toolkit appears to be missing.")
        return None

def main():
    print_banner()
    time.sleep(1)  # Pause for effect

    print_info("Welcome to the Temporal Lab Setup! 🚀")
    print_info("We're about to embark on a journey to activate the Temporal Prompt Engine.")
    print_info("Let's get started by connecting to the Time Portal...\n")
    time.sleep(1)

    # Define the main script path relative to setup.py
    script_dir = Path(__file__).parent.resolve()
    main_script = "TemporalPromptEngine.py"
    main_script_path = script_dir / main_script

    if not main_script_path.exists():
        print_error(f"\nMain script '{main_script}' not found at {main_script_path}. Please ensure it exists.")
        sys.exit(1)

    # Step 1: Check for pip
    if not check_pip(cwd=script_dir):
        if not install_pip(script_dir):
            print_error("\nPip integration failed. Unable to proceed with the setup.")
            sys.exit(1)
        else:
            print_status("Pip is now available for use.")

    # Step 2: Check for Ollama
    if not check_ollama(cwd=script_dir):
        if not prompt_install_ollama("https://ollama.ai/download", script_dir):
            print_error("\nOllama installation incomplete. Setup cannot continue.")
            sys.exit(1)
        else:
            print_status("Ollama is now connected to the Temporal Engine.")

    # Step 3: Check for NVIDIA GPU
    try:
        gpu_info = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, cwd=script_dir)
        if not gpu_info:
            print_error("No NVIDIA GPU detected. A CUDA-enabled NVIDIA GPU is essential for optimal performance.")
            sys.exit(1)
        else:
            print_status(f"NVIDIA GPU detected: {gpu_info.strip()}")
    except:
        print_error("nvidia-smi command not found. Please ensure NVIDIA drivers are installed.")
        sys.exit(1)

    # Step 4: Check for CUDA toolkit
    cuda_version = check_cuda_toolkit(cwd=script_dir)
    if not cuda_version:
        print_error("\nCUDA Toolkit is a vital component for the Temporal Engine's operations.")
        print_info("Please install the CUDA Toolkit from the official Temporal Gateway:")
        print("🌐 https://developer.nvidia.com/cuda-downloads")
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
        input("\n🕰️ After installing the CUDA Toolkit, press Enter to continue...")
        # Recheck after installation
        cuda_version = check_cuda_toolkit(cwd=script_dir)
        if not cuda_version:
            print_error("\nCUDA Toolkit installation not detected. Setup cannot proceed.")
            sys.exit(1)
        else:
            print_status("CUDA Toolkit has been successfully integrated.")

    # Step 5: Set up TemporalPromptEngineEnv virtual environment with current Python
    venv_dir = script_dir / "TemporalPromptEngineEnv"
    if not venv_dir.exists():
        if not create_virtualenv(venv_dir):
            print_error("\nFailed to establish the Virtual Chamber. Setup halted.")
            sys.exit(1)
    else:
        print_status("Virtual Chamber already exists and is ready.")

    # Step 6: Upgrade pip
    if not upgrade_pip(venv_dir):
        print_error("\nFailed to upgrade pip within the Virtual Chamber. Setup halted.")
        sys.exit(1)

    # Get the virtual environment executables
    python_executable, pip_executable = get_venv_executables(venv_dir)

    # Step 7: Install required packages for TemporalPromptEngineEnv
    packages_to_install = [
        {"name": "wheel", "version": "0.41.3"},
        {"name": "playsound", "version": "1.3.0"},
        {"name": "python-dotenv", "version": "1.0.1", "import_name": "dotenv"},
        {"name": "moviepy", "version": "1.0.3", "import_name": "moviepy.editor"},
        {"name": "pydub", "version": "0.25.1"},
        {"name": "Pillow", "version": "10.4.0", "import_name": "PIL"},
        {"name": "requests", "version": "2.32.3"},
        {"name": "pyperclip", "version": "1.9.0"},
        {"name": "scipy", "version": "1.14.1"},
        {"name": "tk", "version": "0.1.0", "import_name": "tkinter"},
        {"name": "accelerate", "version": "1.0.1"},
        {"name": "torch", "version": "2.0.1+cu117", "install_command": [str(pip_executable), "install", "torch==2.0.1+cu117", "--index-url", "https://download.pytorch.org/whl/cu117"]},
        {"name": "torchvision", "version": "0.15.2+cu117", "install_command": [str(pip_executable), "install", "torchvision==0.15.2+cu117", "--index-url", "https://download.pytorch.org/whl/cu117"]},
        {"name": "torchaudio", "version": "2.0.2+cu117", "install_command": [str(pip_executable), "install", "torchaudio==2.0.2+cu117", "--index-url", "https://download.pytorch.org/whl/cu117"]},
        {"name": "diffusers", "version": "0.21.1"},
        {"name": "transformers", "version": "4.30.2"},
        {"name": "audioldm2", "version": "0.1.0", "install_command": [str(pip_executable), "install", "git+https://github.com/haoheliu/AudioLDM2.git#egg=audioldm2"]},
        {"name": "huggingface-hub", "version": "0.25.2"},
        {"name": "pyyaml", "version": "6.0.1", "import_name": "yaml"},
        {"name": "openai", "version": "0.28.0"},
        {"name": "playsound", "version": "1.3.0"},
        {"name": "pynput", "version": "1.7.6"},
        {"name": "numba", "version": "0.57.1"},
        {"name": "numpy", "version": "1.23.5"},
        {"name": "matplotlib", "version": "3.9.2"},
        {"name": "altair", "version": "5.4.1"},
        {"name": "soundfile", "version": "0.12.1"},
        {"name": "python3.10-tk", "version": "3.10.15-1+focal1"},
    ]

    # Install and verify each package
    for pkg in packages_to_install:
        name = pkg["name"]
        version = pkg.get("version")
        install_command = pkg.get("install_command")
        import_name = pkg.get("import_name")
        if not install_and_verify_package(name, version, venv_dir, install_command=install_command, import_name=import_name):
            print_error(f"\nFailed to integrate and verify package '{name}'. Setup cannot continue.")
            sys.exit(1)

    # Step 8: Clone CogVideo repository and set up CogVx virtual environment using Python 3.9
    cogvideo_repo_path = script_dir / "CogVideo"
    cogvideo_gradio_demo_path = cogvideo_repo_path / "inference" / "gradio_composite_demo"
    cogvx_venv_dir = cogvideo_gradio_demo_path / "CogVx"

    # Clone the CogVideo repository
    clone_cogvideo_repo(cogvideo_repo_path)

    # Copy custom scripts after cloning the repository
    custom_scripts_dir = script_dir / "VideoGeneratorScripts"
    copy_custom_scripts(custom_scripts_dir, cogvideo_gradio_demo_path)

    # Set up CogVx virtual environment and install dependencies
    setup_cogvx(cogvideo_gradio_demo_path, cogvx_venv_dir)

    # Step 9: Create .env file
    create_env_file(script_dir)

    # Step 10: Create or validate JSON configuration files
    create_default_json_files(script_dir)

    # Step 11: Check for ffmpeg
    if not check_ffmpeg(cwd=script_dir):
        if not prompt_install_ffmpeg(script_dir):
            print_error("\nffmpeg installation incomplete. Setup cannot proceed.")
            sys.exit(1)
        else:
            print_status("ffmpeg has been successfully integrated.")

    # Step 12: Prompt to create desktop shortcut
    choice = input("\n🖥️ Would you like to create a desktop shortcut for easy access to the Temporal Prompt Engine? (y/n): ").strip().lower()
    if choice == 'y':
        create_desktop_shortcut(venv_dir, main_script_path)
    else:
        print_info("Desktop shortcut creation skipped. You can manually create one later if desired.")

    # Step 13: Prompt to launch the main script
    choice = input("\n🚀 Temporal Engine Setup Complete! Would you like to activate the Temporal Prompt Engine now? (y/n): ").strip().lower()
    if choice == 'y':
        print_info("\nActivating Temporal Prompt Engine...")
        try:
            # Use the virtual environment's Python executable to run the main script
            python_executable, _ = get_venv_executables(venv_dir)
            if not python_executable.exists():
                print_error(f"Virtual Environment's Python executable not found at {python_executable}.")
                print_warning("Please verify the Virtual Chamber setup and try again.")
                return
            subprocess.run([str(python_executable), str(main_script_path)], check=True)
            print_status("\nTemporalPromptEngine.py has been successfully launched into the Time Portal.")
        except subprocess.CalledProcessError as e:
            print_error("\nFailed to launch TemporalPromptEngine.py. Please review the setup and attempt to run the script manually.")
            print_error(f"Error Details: {e}")
    else:
        print_info("\nSetup concluded. You can activate the Temporal Prompt Engine anytime using the Virtual Chamber's Python environment.")

    print_status("\n============================================")
    print(f"{Fore.CYAN}        ✨ Temporal Lab Setup Complete ✨        ")
    print("============================================")
    print_info("The Temporal Prompt Engine is now ready to explore the realms of time and virtual reality!")
    input("🔒 Press Enter to close the Temporal Portal and exit setup...")

if __name__ == "__main__":
    main()
