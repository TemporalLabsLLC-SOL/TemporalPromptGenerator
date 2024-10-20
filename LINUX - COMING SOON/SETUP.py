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
        python_executable = venv_dir / "bin" / "python"
        pip_executable = venv_dir / "bin" / "pip"
    return python_executable, pip_executable

def check_pip(cwd=None):
    """
    Check if pip is installed.
    """
    print_info("Initiating Pip Verification...")
    try:
        output = run_command(["pip", "--version"], capture_output=True, cwd=cwd)
        if output:
            print_status(f"Pip is ready: {output}")
            return True
    except:
        pass
    print_warning("Pip is not detected.")
    return False

def install_pip(script_dir):
    """
    Install pip using ensurepip.
    """
    print_action("Installing Pip through Temporal Enhancement...")
    result = run_command([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=script_dir)
    if result is None:
        print_status("Pip has been successfully integrated.")
        return True
    else:
        print_error("Failed to integrate Pip into the Temporal Engine.")
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
        [str(pip_executable), "install", "torch==2.3.1+cu121", "torchvision==0.18.1+cu121", "torchaudio==2.3.1+cu121", "--index-url", "https://download.pytorch.org/whl/cu121"],
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
    Create and set up the virtual environment for CogVideo using Python 3.12 if not already present.
    """
    if not cogvx_venv_dir.exists():
        print_info("\nCogVx Virtual Environment not detected. Creating with Python 3.12...")
        # Use 'py -3.12' to create the virtual environment
        python_command = "py"
        python_args = ["-3.12", "-m", "venv", str(cogvx_venv_dir)]
        command = [python_command] + python_args
        result = run_command(command)
        if result is None:
            print_status("CogVx Virtual Environment established successfully.")
        else:
            print_error("Failed to establish CogVx Virtual Environment with Python 3.12. Halting setup.")
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
    print_info("Please install ffmpeg from the official Temporal Gateway:")
    print("🌐 https://ffmpeg.org/download.html")
    webbrowser.open("https://ffmpeg.org/download.html")
    input("\n🕰️ Once ffmpeg is installed, press Enter to finalize the setup...")
    # Recheck after installation
    return check_ffmpeg(cwd=script_dir)

def prompt_launch_script(venv_dir, main_script_path):
    """
    Prompt the user to launch the main Python script.
    """
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

def get_windows_desktop_path():
    """
    Get the path to the Windows desktop folder.
    """
    try:
        from ctypes import windll, wintypes, byref, create_unicode_buffer
        CSIDL_DESKTOPDIRECTORY = 0x0010
        SHGFP_TYPE_CURRENT = 0

        buf = create_unicode_buffer(wintypes.MAX_PATH)
        result = windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOPDIRECTORY, None, SHGFP_TYPE_CURRENT, buf)
        if result == 0:
            return buf.value
        else:
            print_warning("Unable to retrieve desktop path. Defaulting to user profile's Desktop.")
            return os.path.join(os.environ['USERPROFILE'], 'Desktop')
    except Exception as e:
        print_warning(f"Error obtaining desktop path: {e}")
        return os.path.join(os.environ['USERPROFILE'], 'Desktop')

def create_desktop_shortcut(venv_dir, main_script_path):
    """
    Create a desktop shortcut to run the main script using the virtual environment.
    """
    print_info("\nSetting Up Desktop Shortcut for Quick Access...")
    # Only proceed if on Windows
    if platform.system() != "Windows":
        print_warning("Desktop shortcut creation is currently only supported on Windows.")
        return

    try:
        from win32com.client import Dispatch
    except ImportError:
        # Attempt to install pywin32
        print_action("Installing 'pywin32' module required for shortcut creation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"], check=True)
            from win32com.client import Dispatch
            print_status("'pywin32' module installed successfully.")
        except Exception as e:
            print_error(f"Failed to install 'pywin32'. Shortcut creation aborted. Error: {e}")
            return

    desktop = get_windows_desktop_path()
    shortcut_path = os.path.join(desktop, 'Temporal Prompt Engine.lnk')
    target = str(venv_dir / 'Scripts' / 'python.exe')
    working_directory = str(main_script_path.parent)
    arguments = f'"{main_script_path}"'
    icon = str(main_script_path)

    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = target
    shortcut.Arguments = arguments
    shortcut.WorkingDirectory = working_directory
    shortcut.IconLocation = icon
    shortcut.save()

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
        {"name": "torch", "version": "2.4.1+cu118", "install_command": [str(pip_executable), "install", "torch==2.4.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118"]},
        {"name": "torchvision", "version": "0.19.1+cu118", "install_command": [str(pip_executable), "install", "torchvision==0.19.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118"]},
        {"name": "torchaudio", "version": "2.4.1+cu118", "install_command": [str(pip_executable), "install", "torchaudio==2.4.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118"]},
        {"name": "diffusers", "version": "0.21.1"},
        {"name": "transformers", "version": "4.30.2"},
        {"name": "audioldm2", "version": "0.1.0", "install_command": [str(pip_executable), "install", "git+https://github.com/haoheliu/AudioLDM2.git#egg=audioldm2"]},
        {"name": "huggingface-hub", "version": "0.25.2"},
        {"name": "aiofiles", "version": "23.2.1"},
        {"name": "altair", "version": "5.4.1"},
        {"name": "annotated-types", "version": "0.7.0"},
        {"name": "anyio", "version": "4.6.2.post1"},
        {"name": "attrs", "version": "24.2.0"},
        {"name": "audioread", "version": "3.0.1"},
        {"name": "babel", "version": "2.16.0"},
        {"name": "bibtexparser", "version": "2.0.0b7"},
        {"name": "certifi", "version": "2024.8.30"},
        {"name": "cffi", "version": "1.17.1"},
        {"name": "chardet", "version": "5.2.0"},
        {"name": "charset-normalizer", "version": "3.4.0"},
        {"name": "click", "version": "8.1.7"},
        {"name": "clldutils", "version": "3.22.2"},
        {"name": "colorama", "version": "0.4.6"},
        {"name": "colorlog", "version": "6.8.2"},
        {"name": "contourpy", "version": "1.3.0"},
        {"name": "csvw", "version": "3.3.1"},
        {"name": "cycler", "version": "0.12.1"},
        {"name": "decorator", "version": "4.4.2"},
        {"name": "dlinfo", "version": "1.2.1"},
        {"name": "einops", "version": "0.8.0"},
        {"name": "exceptiongroup", "version": "1.2.2"},
        {"name": "fastapi", "version": "0.115.2"},
        {"name": "ffmpy", "version": "0.4.0"},
        {"name": "filelock", "version": "3.16.1"},
        {"name": "fonttools", "version": "4.54.1"},
        {"name": "fsspec", "version": "2024.9.0"},
        {"name": "ftfy", "version": "6.3.0"},
        {"name": "gradio", "version": "3.50.2"},
        {"name": "gradio_client", "version": "0.6.1"},
        {"name": "h11", "version": "0.14.0"},
        {"name": "httpcore", "version": "1.0.6"},
        {"name": "httpx", "version": "0.27.2"},
        {"name": "idna", "version": "3.10"},
        {"name": "imageio", "version": "2.36.0"},
        {"name": "imageio-ffmpeg", "version": "0.5.1"},
        {"name": "importlib_metadata", "version": "8.5.0"},
        {"name": "importlib_resources", "version": "6.4.5"},
        {"name": "isodate", "version": "0.6.1"},
        {"name": "Jinja2", "version": "3.1.4"},
        {"name": "joblib", "version": "1.4.2"},
        {"name": "jsonschema", "version": "4.23.0"},
        {"name": "jsonschema-specifications", "version": "2024.10.1"},
        {"name": "julius", "version": "0.2.7"},
        {"name": "kiwisolver", "version": "1.4.7"},
        {"name": "language-tags", "version": "1.2.0"},
        {"name": "librosa", "version": "0.9.1"},
        {"name": "llvmlite", "version": "0.43.0"},
        {"name": "local-attention", "version": "1.9.15"},
        {"name": "lxml", "version": "5.3.0"},
        {"name": "Markdown", "version": "3.7"},
        {"name": "MarkupSafe", "version": "2.1.5"},
        {"name": "matplotlib", "version": "3.9.2"},
        {"name": "mpmath", "version": "1.3.0"},
        {"name": "narwhals", "version": "1.9.3"},
        {"name": "networkx", "version": "3.4.1"},
        {"name": "numba", "version": "0.60.0"},
        {"name": "numpy", "version": "1.23.5"},
        {"name": "orjson", "version": "3.10.7"},
        {"name": "packaging", "version": "24.1"},
        {"name": "pandas", "version": "2.2.3"},
        {"name": "phonemizer", "version": "3.3.0"},
        {"name": "platformdirs", "version": "4.3.6"},
        {"name": "pooch", "version": "1.8.2"},
        {"name": "primePy", "version": "1.3"},
        {"name": "proglog", "version": "0.1.10"},
        {"name": "progressbar", "version": "2.5"},
        {"name": "psutil", "version": "6.0.0"},
        {"name": "pycparser", "version": "2.22"},
        {"name": "pydantic", "version": "2.9.2"},
        {"name": "pydantic_core", "version": "2.23.4"},
        {"name": "pylatexenc", "version": "2.10"},
        {"name": "pyparsing", "version": "3.2.0"},
        {"name": "pystoi", "version": "0.4.1"},
        {"name": "python-dateutil", "version": "2.9.0.post0"},
        {"name": "python-multipart", "version": "0.0.12"},
        {"name": "pytz", "version": "2024.2"},
        {"name": "PyYAML", "version": "6.0.2"},
        {"name": "rdflib", "version": "7.0.0"},
        {"name": "referencing", "version": "0.35.1"},
        {"name": "regex", "version": "2024.9.11"},
        {"name": "resampy", "version": "0.4.3"},
        {"name": "rfc3986", "version": "1.5.0"},
        {"name": "rpds-py", "version": "0.20.0"},
        {"name": "safetensors", "version": "0.4.5"},
        {"name": "scikit-learn", "version": "1.5.2"},
        {"name": "segments", "version": "2.2.1"},
        {"name": "semantic-version", "version": "2.10.0"},
        {"name": "six", "version": "1.16.0"},
        {"name": "sniffio", "version": "1.3.1"},
        {"name": "soundfile", "version": "0.12.1"},
        {"name": "starlette", "version": "0.39.2"},
        {"name": "sympy", "version": "1.13.3"},
        {"name": "tabulate", "version": "0.9.0"},
        {"name": "threadpoolctl", "version": "3.5.0"},
        {"name": "timm", "version": "1.0.10"},
        {"name": "tokenizers", "version": "0.13.3"},
        {"name": "torch-audiomentations", "version": "0.11.1"},
        {"name": "torch-pitch-shift", "version": "1.2.4"},
        {"name": "torch-stoi", "version": "0.2.2"},
        {"name": "torchcrepe", "version": "0.0.20"},
        {"name": "torchdiffeq", "version": "0.2.4"},
        {"name": "torchfcpe", "version": "0.0.4"},
        {"name": "torchgen", "version": "0.0.1"},
        {"name": "torchlibrosa", "version": "0.1.0"},
        {"name": "torchmetrics", "version": "0.11.4"},
        {"name": "torchsde", "version": "0.2.6"},
        {"name": "torchvision", "version": "0.19.1+cu118", "install_command": [str(pip_executable), "install", "torchvision==0.19.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118"]},
        {"name": "tqdm", "version": "4.66.5"},
        {"name": "trampoline", "version": "0.1.2"},
        {"name": "transformers", "version": "4.30.2"},
        {"name": "typing_extensions", "version": "4.12.2"},
        {"name": "tzdata", "version": "2024.2"},
        {"name": "Unidecode", "version": "1.3.8"},
        {"name": "uritemplate", "version": "4.1.1"},
        {"name": "urllib3", "version": "2.2.3"},
        {"name": "uvicorn", "version": "0.31.1"},
        {"name": "wcwidth", "version": "0.2.13"},
        {"name": "websockets", "version": "11.0.3"},
        {"name": "zipp", "version": "3.20.2"},
    ]

    # Step 8: Install Additional Torch-Related Packages
    additional_torch_packages = [
        {"name": "torch-audiomentations", "version": "0.11.1"},
        {"name": "torch-pitch-shift", "version": "1.2.4"},
        {"name": "torch-stoi", "version": "0.2.2"},
        {"name": "torchcrepe", "version": "0.0.20"},
        {"name": "torchdiffeq", "version": "0.2.4"},
        {"name": "torchfcpe", "version": "0.0.4"},
        {"name": "torchgen", "version": "0.0.1"},
        {"name": "torchlibrosa", "version": "0.1.0"},
        {"name": "torchmetrics", "version": "0.11.4"},
        {"name": "torchsde", "version": "0.2.6"},
    ]

    # Combine all packages
    all_packages = packages_to_install + additional_torch_packages

    # Step 9: Install and verify each package
    for pkg in all_packages:
        name = pkg["name"]
        version = pkg.get("version")
        install_command = pkg.get("install_command")
        import_name = pkg.get("import_name")
        if not install_and_verify_package(name, version, venv_dir, install_command=install_command, import_name=import_name):
            print_error(f"\nFailed to integrate and verify package '{name}'. Setup cannot continue.")
            sys.exit(1)

    # Step 10: Clone CogVideo repository and set up CogVx virtual environment using Python 3.12
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

    # Step 11: Create .env file
    create_env_file(script_dir)

    # Step 12: Create or validate JSON configuration files
    create_default_json_files(script_dir)

    # Step 13: Check for ffmpeg
    if not check_ffmpeg(cwd=script_dir):
        if not prompt_install_ffmpeg(script_dir):
            print_error("\nffmpeg installation incomplete. Setup cannot proceed.")
            sys.exit(1)
        else:
            print_status("ffmpeg has been successfully integrated.")

    # Step 14: Prompt to create desktop shortcut
    choice = input("\n🖥️ Would you like to create a desktop shortcut for easy access to the Temporal Prompt Engine? (y/n): ").strip().lower()
    if choice == 'y':
        create_desktop_shortcut(venv_dir, main_script_path)
    else:
        print_info("Desktop shortcut creation skipped. You can manually create one later if desired.")

    # Step 15: Prompt to launch the main script
    prompt_launch_script(venv_dir, main_script_path)

    print_status("\n============================================")
    print(f"{Fore.CYAN}        ✨ Temporal Lab Setup Complete ✨        ")
    print("============================================")
    print_info("The Temporal Prompt Engine is now ready to explore the realms of time and virtual reality!")
    input("🔒 Press Enter to close the Temporal Portal and exit setup...")
    
if __name__ == "__main__":
    main()
