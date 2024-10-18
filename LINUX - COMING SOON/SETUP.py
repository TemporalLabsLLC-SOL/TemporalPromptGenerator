import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import json
import platform
import shutil

def run_command(command, capture_output=False, cwd=None):
    """
    Run a system command with enhanced logging.
    """
    command_str = ' '.join(command) if isinstance(command, list) else command
    print(f"\n💻 Running command: {command_str}")
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
                print(f"📄 Command output: {output}")
            return output
        else:
            subprocess.run(command, check=True, cwd=cwd)
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}")
        if e.stdout:
            print(f"📄 Output: {e.stdout.strip()}")
        if e.stderr:
            print(f"📝 Error Output: {e.stderr.strip()}")
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
    print("\n🔍 Checking for pip installation...")
    try:
        output = run_command(["pip", "--version"], capture_output=True, cwd=cwd)
        if output:
            print(f"✅ pip is installed: {output}")
            return True
    except:
        pass
    print("⚠️ pip is not installed.")
    return False

def install_pip(script_dir):
    """
    Install pip using ensurepip.
    """
    print("🔧 Attempting to install pip...")
    result = run_command([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=script_dir)
    if result is None:
        print("✅ pip installed successfully.")
        return True
    else:
        print("❌ Failed to install pip.")
        return False

def check_ollama(cwd=None):
    """
    Check if Ollama is installed.
    """
    print("\n🔍 Checking for Ollama installation...")
    try:
        output = run_command(["ollama", "--version"], capture_output=True, cwd=cwd)
        if output and "ollama version" in output.lower():
            print(f"✅ Ollama is installed: {output}")
            return True
    except:
        pass
    print("⚠️ Ollama is not installed.")
    return False

def prompt_install_ollama(download_url, script_dir):
    """
    Prompt the user to install Ollama.
    """
    print("\nOllama is not installed. Please install it from the following link:")
    print(download_url)
    webbrowser.open(download_url)
    input("📥 Press Enter after you have installed Ollama to continue...")
    # Recheck after installation
    return check_ollama(cwd=script_dir)

def create_virtualenv(venv_dir, python_command=None):
    """
    Create a virtual environment using the specified Python command.
    """
    print(f"\n🔧 Creating a virtual environment in '{venv_dir}'...")
    if python_command:
        command = [python_command, "-m", "venv", str(venv_dir)]
    else:
        command = [sys.executable, "-m", "venv", str(venv_dir)]

    result = run_command(command)
    if result is None:
        print("✅ Virtual environment created successfully.")
        return True
    else:
        print("❌ Failed to create virtual environment.")
        return False

def upgrade_pip(venv_dir):
    """
    Upgrade pip within the virtual environment.
    """
    print("\n🔧 Upgrading pip in the virtual environment...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"⚠️ pip executable not found at {pip_executable}.")
        return False
    result = run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"], cwd=venv_dir)
    if result is None:
        print("✅ pip upgraded successfully.")
        return True
    else:
        print("❌ Failed to upgrade pip.")
        return False

def install_and_verify_package(package_name, version, venv_dir, install_command=None, import_name=None):
    """
    Install a package using pip from the virtual environment and verify its installation.
    """
    print(f"\n📦 Installing package '{package_name}=={version}'...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"⚠️ pip executable not found at {pip_executable}.")
        return False

    # Use custom install command if provided
    if install_command is None:
        install_command = [str(pip_executable), "install", f"{package_name}=={version}"]
    else:
        install_command = install_command

    # Install the package
    try:
        run_command(install_command, cwd=venv_dir)
        print(f"✅ Package '{package_name}=={version}' installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install package '{package_name}=={version}'. Error: {e}")
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
        print(f"🔍 Skipping import verification for '{package_name}'.")
        return True

    # Verify installation by attempting to import the package
    try:
        # Use custom import name if provided
        if import_name is None:
            package_import_name = package_name.replace('-', '_')
            # Handle special cases
            if package_import_name == 'python_dotenv':
                package_import_name = 'dotenv'
            elif package_import_name == 'pyyaml':
                package_import_name = 'yaml'
            elif package_import_name == 'moviepy':
                package_import_name = 'moviepy.editor'
            elif package_import_name == 'Pillow':
                package_import_name = 'PIL'
            elif package_import_name == 'tk':
                package_import_name = 'tkinter'
            elif package_import_name.startswith('audioldm2'):
                package_import_name = 'audioldm2'
            elif package_import_name == 'pydub':
                package_import_name = 'pydub'
            else:
                package_import_name = package_import_name
        else:
            package_import_name = import_name

        command = [str(python_executable), "-c", f"import {package_import_name}"]
        run_command(command, cwd=venv_dir)
        print(f"✅ Package '{package_name}' verified successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to verify package '{package_name}'. Error: {e}")
        return False

def clone_cogvideo_repo(cogvideo_repo_path):
    """
    Clone the CogVideo repository into the specified path.
    If the repository already exists, it will be removed and re-cloned.
    """
    if cogvideo_repo_path.exists():
        print("\n📁 CogVideo repository already exists. Removing it to ensure a clean clone...")
        shutil.rmtree(cogvideo_repo_path)
    print("\n📦 Cloning CogVideo repository...")
    run_command(["git", "clone", "https://github.com/THUDM/CogVideo", str(cogvideo_repo_path)])

def copy_custom_scripts(custom_scripts_dir, gradio_demo_path):
    """
    Copy custom scripts from the user's directory to the gradio_composite_demo directory.
    """
    print("\n📂 Copying custom scripts to CogVideo gradio_composite_demo directory...")
    for script_name in ["TemporalCog-5b.py", "TemporalCog-2b.py", "PromptList2MP4Utility.py"]:
        source_file = custom_scripts_dir / script_name
        target_file = gradio_demo_path / script_name
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied {script_name} to {target_file}")
        else:
            print(f"⚠️ Custom script {script_name} not found in {custom_scripts_dir}. Skipping.")

def install_dependencies_for_cogvideo(cogvideo_gradio_demo_path, cogvx_venv_dir):
    """
    Install dependencies for CogVideo into the CogVx virtual environment.
    """
    print(f"\n🔧 Installing dependencies for CogVideo in '{cogvx_venv_dir}'...")

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
            print(f"✅ Command succeeded: {' '.join(command)}")
        except Exception as e:
            print(f"❌ Failed to run command: {' '.join(command)}. Error: {e}")
            sys.exit(1)

    print(f"✅ Installed dependencies for CogVx successfully.")

def setup_cogvx(cogvideo_gradio_demo_path, cogvx_venv_dir):
    """
    Create and set up the virtual environment for CogVideo using Python 3.12 if not already present.
    """
    if not cogvx_venv_dir.exists():
        print("\n🔍 CogVx virtual environment not found. Creating with Python 3.12...")
        # Use 'py -3.12' to create the virtual environment
        python_command = "py"
        python_args = ["-3.12", "-m", "venv", str(cogvx_venv_dir)]
        command = [python_command] + python_args
        result = run_command(command)
        if result is None:
            print("✅ CogVx virtual environment created successfully.")
        else:
            print("❌ Failed to create CogVx virtual environment with Python 3.12. Exiting setup.")
            sys.exit(1)
    else:
        print("✅ CogVx virtual environment already exists.")

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
        print("\nℹ️ .env file already exists. Skipping creation.")
        return

    print("\n📝 Creating a .env file...")

    # Ask the user for the COMFYUI_PROMPTS_FOLDER
    comfyui_prompts_folder = input("Please enter the full path to your ComfyUI prompts folder: ").strip()

    # Ask the user for the LAST_USED_DIRECTORY
    last_used_directory = input("Please enter the full path to your last used directory: ").strip()

    default_env = {
        "COMFYUI_PROMPTS_FOLDER": comfyui_prompts_folder,
        "LAST_USED_DIRECTORY": last_used_directory
    }

    with open(env_path, 'w') as f:
        for key, value in default_env.items():
            f.write(f"{key}={value}\n")

    print("✅ .env file created successfully.")

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
            print(f"\n📝 Creating default '{file_name}'...")
            with open(file_path, 'w') as f:
                json.dump(default_content, f, indent=4)
            print(f"✅ '{file_name}' created with default content.")
        else:
            # Validate JSON structure
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                print(f"ℹ️ '{file_name}' exists and contains valid JSON.")
            except json.JSONDecodeError:
                print(f"\n⚠️ '{file_name}' is corrupted or contains invalid JSON. Recreating with default content...")
                with open(file_path, 'w') as f:
                    json.dump(default_content, f, indent=4)
                print(f"✅ '{file_name}' has been recreated with default content.")

def check_ffmpeg(cwd=None):
    """
    Check if ffmpeg is installed.
    """
    print("\n🔍 Checking for ffmpeg installation...")
    try:
        result = run_command(["ffmpeg", "-version"], capture_output=True, cwd=cwd)
        if result and "ffmpeg version" in result.lower():
            print("✅ ffmpeg is installed.")
            return True
    except:
        pass
    print("⚠️ ffmpeg is not installed.")
    return False

def prompt_install_ffmpeg(script_dir):
    """
    Prompt the user to install ffmpeg.
    """
    print("\n⚠️ ffmpeg is required for video processing.")
    print("📥 Please install ffmpeg from the official website:")
    print("🌐 https://ffmpeg.org/download.html")
    webbrowser.open("https://ffmpeg.org/download.html")
    input("📥 Press Enter after you have installed ffmpeg to continue...")
    # Recheck after installation
    return check_ffmpeg(cwd=script_dir)

def prompt_launch_script(venv_dir, main_script_path):
    """
    Prompt the user to launch the main Python script.
    """
    choice = input("\n🎉 Setup complete! Would you like to launch the Temporal Prompt Engine now? (y/n): ").strip().lower()
    if choice == 'y':
        print("\n🚀 Launching the Temporal Prompt Engine...")
        try:
            # Use the virtual environment's Python executable to run the main script
            python_executable, _ = get_venv_executables(venv_dir)
            if not python_executable.exists():
                print(f"❌ Virtual environment's Python executable not found at {python_executable}.")
                print("🔧 Please ensure the virtual environment is set up correctly.")
                return
            subprocess.run([str(python_executable), str(main_script_path)], check=True)
            print("\n✅ TemporalPromptEngine.py launched successfully.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Failed to launch TemporalPromptEngine.py. Please check the setup and try running the script manually.")
            print(f"⚠️ Error: {e}")
    else:
        print("\n🛠️ Setup completed. You can run the Temporal Prompt Engine later using the virtual environment's Python.")

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
            print("⚠️ Unable to get desktop folder path. Using default path.")
            return os.path.join(os.environ['USERPROFILE'], 'Desktop')
    except Exception as e:
        print(f"⚠️ Error getting desktop folder path: {e}")
        return os.path.join(os.environ['USERPROFILE'], 'Desktop')

def create_desktop_shortcut(venv_dir, main_script_path):
    """
    Create a desktop shortcut to run the main script using the virtual environment.
    """
    print("\n🖥️ Creating a desktop shortcut to launch the Temporal Prompt Engine...")
    # Only proceed if on Windows
    if platform.system() != "Windows":
        print("⚠️ Desktop shortcut creation is currently only supported on Windows.")
        return

    try:
        from win32com.client import Dispatch
    except ImportError:
        # Attempt to install pywin32
        print("🔧 Installing pywin32 module required for shortcut creation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"], check=True)
            from win32com.client import Dispatch
        except Exception as e:
            print(f"❌ Failed to install pywin32. Shortcut creation aborted. Error: {e}")
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

    print(f"✅ Desktop shortcut created at {shortcut_path}")

def check_cuda_toolkit(cwd=None):
    """
    Check if CUDA toolkit is installed.
    """
    print("\n🔍 Checking for CUDA toolkit installation...")
    try:
        cuda_version_output = run_command(["nvcc", "--version"], capture_output=True, cwd=cwd)
        if cuda_version_output:
            # Parse the CUDA version from the output
            import re
            match = re.search(r"release (\d+\.\d+),", cuda_version_output)
            if match:
                cuda_version = match.group(1)
                print(f"✅ Detected CUDA toolkit version: {cuda_version}")
                return cuda_version
            else:
                print("⚠️ Could not parse CUDA version from nvcc output.")
                return None
        else:
            print("⚠️ CUDA toolkit not found.")
            return None
    except:
        print("⚠️ nvcc not found. CUDA toolkit is not installed.")
        return None

def main():
    print("============================================")
    print("     ✨ Temporal Prompt Engine Setup ✨      ")
    print("============================================\n")

    # Define the main script path relative to setup.py
    script_dir = Path(__file__).parent.resolve()
    main_script = "TemporalPromptEngine.py"
    main_script_path = script_dir / main_script

    if not main_script_path.exists():
        print(f"\n❌ Main script '{main_script}' not found at {main_script_path}. Please ensure it exists.")
        sys.exit(1)

    # Step 1: Check for pip
    if not check_pip(cwd=script_dir):
        if not install_pip(script_dir):
            print("\n❌ pip installation failed. Exiting setup.")
            sys.exit(1)

    # Step 2: Check for Ollama
    if not check_ollama(cwd=script_dir):
        if not prompt_install_ollama("https://ollama.ai/download", script_dir):
            print("\n❌ Ollama installation not completed. Exiting setup.")
            sys.exit(1)

    # Step 3: Check for NVIDIA GPU
    try:
        gpu_info = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, cwd=script_dir)
        if not gpu_info:
            print("❌ No NVIDIA GPU detected. A CUDA-enabled NVIDIA GPU is required.")
            sys.exit(1)
        else:
            print(f"✅ NVIDIA GPU detected: {gpu_info.strip()}")
    except:
        print("❌ nvidia-smi not found. Please ensure that NVIDIA drivers are installed.")
        sys.exit(1)

    # Step 4: Check for CUDA toolkit
    cuda_version = check_cuda_toolkit(cwd=script_dir)
    if not cuda_version:
        print("\n❌ CUDA toolkit is required for this application.")
        print("🌐 Please install the CUDA toolkit from the official NVIDIA website:")
        print("🌐 https://developer.nvidia.com/cuda-downloads")
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
        input("📥 Press Enter after you have installed the CUDA toolkit to continue...")
        # Recheck after installation
        cuda_version = check_cuda_toolkit(cwd=script_dir)
        if not cuda_version:
            print("\n❌ CUDA toolkit installation not detected. Exiting setup.")
            sys.exit(1)

    # Step 5: Set up TemporalPromptEngineEnv virtual environment with current Python
    venv_dir = script_dir / "TemporalPromptEngineEnv"
    if not venv_dir.exists():
        if not create_virtualenv(venv_dir):
            print("\n❌ Failed to create virtual environment. Exiting setup.")
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists.")

    # Step 6: Upgrade pip
    if not upgrade_pip(venv_dir):
        print("\n❌ Failed to upgrade pip. Exiting setup.")
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
            print(f"\n❌ Failed to install and verify package '{name}'. Exiting setup.")
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
            print("\n❌ ffmpeg installation not completed. Exiting setup.")
            sys.exit(1)

    # Step 14: Prompt to launch the main script
    prompt_launch_script(venv_dir, main_script_path)

    # Step 15: Prompt to create desktop shortcut
    choice = input("\n🖥️ Would you like to create a desktop shortcut to launch the Temporal Prompt Engine? (y/n): ").strip().lower()
    if choice == 'y':
        create_desktop_shortcut(venv_dir, main_script_path)
    else:
        print("📝 Shortcut creation skipped.")

    print("\n============================================")
    print("        ✨ Setup Completed Successfully ✨     ")
    print("============================================")
    input("🔒 Press Enter to exit...")

if __name__ == "__main__":
    main()
