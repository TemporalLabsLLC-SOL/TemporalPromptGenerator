import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import json
import platform
import urllib.request
import zipfile
import shutil
import tempfile
import re

def run_command(command, capture_output=False, cwd=None, shell=True):
    """
    Run a system command with friendly logging.
    """
    print(f"\n🔧 Running command: {command}")
    try:
        if capture_output:
            result = subprocess.run(
                command,
                shell=shell,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd
            )
            print(f"✅ Command succeeded.")
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=shell, check=True, cwd=cwd)
            print(f"✅ Command executed successfully.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}.")
        if e.stdout:
            print(f"📄 Output: {e.stdout}")
        if e.stderr:
            print(f"⚠️ Error Output: {e.stderr}")
        return None

def get_venv_executables(venv_dir):
    """
    Get paths to the Python and pip executables in the virtual environment.
    """
    python_executable = venv_dir / "bin" / "python"
    pip_executable = venv_dir / "bin" / "pip"
    return python_executable, pip_executable

def check_pip(cwd=None):
    """
    Check if pip is installed.
    """
    print("\n🔍 Checking for pip installation...")
    try:
        output = run_command("pip --version", capture_output=True, cwd=cwd)
        if output:
            print(f"✅ pip is installed: {output}")
            return True
    except:
        pass
    print("⚠️ pip is not installed.")
    return False

def install_pip_linux():
    """
    Install pip using the package manager.
    """
    print("\n🔧 Attempting to install pip...")
    distro = get_linux_distro()
    if distro in ['ubuntu', 'debian']:
        install_command = "sudo apt-get update && sudo apt-get install -y python3-pip"
    elif distro in ['fedora']:
        install_command = "sudo dnf install -y python3-pip"
    elif distro in ['centos', 'rhel']:
        install_command = "sudo yum install -y epel-release && sudo yum install -y python3-pip"
    else:
        print(f"❌ Unsupported Linux distribution: {distro}. Please install pip manually.")
        return False

    result = run_command(install_command)
    if result is None:
        print("✅ pip installed successfully.")
        return True
    else:
        print("❌ Failed to install pip.")
        return False

def get_linux_distro():
    """
    Detect the Linux distribution.
    """
    try:
        import distro
    except ImportError:
        print("\n🔧 Installing 'distro' package to detect Linux distribution...")
        run_command("sudo pip install --upgrade pip")
        run_command("sudo pip install distro")
        import distro
    return distro.id()

def check_git(cwd=None):
    """
    Check if Git is installed.
    """
    print("\n🔍 Checking for Git installation...")
    try:
        output = run_command("git --version", capture_output=True, cwd=cwd)
        if output:
            print(f"✅ Git is installed: {output}")
            return True
    except:
        pass
    print("⚠️ Git is not installed.")
    return False

def install_git_linux():
    """
    Install Git using the package manager.
    """
    print("\n🔧 Installing Git...")
    distro = get_linux_distro()
    if distro in ['ubuntu', 'debian']:
        install_command = "sudo apt-get update && sudo apt-get install -y git"
    elif distro in ['fedora']:
        install_command = "sudo dnf install -y git"
    elif distro in ['centos', 'rhel']:
        install_command = "sudo yum install -y git"
    else:
        print(f"❌ Unsupported Linux distribution: {distro}. Please install Git manually.")
        return False

    result = run_command(install_command)
    if result is None:
        print("✅ Git installed successfully.")
        return True
    else:
        print("❌ Failed to install Git.")
        return False

def prompt_install_ollama_linux():
    """
    Prompt the user to install Ollama with a friendly message.
    """
    print("\n🚀 Ollama is required for the Temporal Prompt Engine.")
    choice = input("🔗 Would you like me to open the browser to download Ollama? (y/n): ").strip().lower()
    if choice == 'y':
        webbrowser.open("https://ollama.com/download/Linux")
        print("\n📥 Please follow the instructions on the website to install Ollama.")
        print("⚠️ **IMPORTANT:** After installation, **RESTART YOUR TERMINAL OR SYSTEM** for the changes to take effect.")
        print("🔄 Then, please re-run the SETUP script to continue.")
    else:
        print("\n🛑 Installation of Ollama aborted by the user.")
    return False  # Exiting setup to allow user to install manually

def check_ollama(cwd=None):
    """
    Check if Ollama is installed.
    """
    print("\n🔍 Checking for Ollama installation...")
    try:
        output = run_command("ollama --version", capture_output=True, cwd=cwd)
        if output:
            print(f"✅ Ollama is installed: {output}")
            return True
    except:
        pass
    print("⚠️ Ollama is not installed.")
    return False

def check_cuda_toolkit(cwd=None):
    """
    Check if CUDA toolkit is installed and retrieve its version.
    """
    print("\n🔍 Checking for CUDA toolkit installation...")
    try:
        cuda_version_output = run_command("nvcc --version", capture_output=True, cwd=cwd)
        if cuda_version_output:
            # Parse the CUDA version from the output
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

def prompt_install_cuda_linux():
    """
    Prompt the user to install CUDA toolkit with a friendly message.
    """
    print("\n⚡ CUDA Toolkit is required for optimal performance.")
    choice = input("🔗 Would you like me to open the browser to download CUDA Toolkit? (y/n): ").strip().lower()
    if choice == 'y':
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
        print("\n📥 Please follow the instructions on the NVIDIA website to install the CUDA Toolkit.")
        print("⚠️ **IMPORTANT:** After installation, **RESTART YOUR TERMINAL OR SYSTEM** for the changes to take effect.")
        print("🔄 Then, please re-run the SETUP script to continue.")
    else:
        print("\n🛑 Installation of CUDA Toolkit aborted by the user.")
    return False  # Exiting setup to allow user to install manually

def create_virtualenv(venv_dir):
    """
    Create a virtual environment.
    """
    print(f"\n🔧 Creating a virtual environment in '{venv_dir}'...")
    result = run_command(f"python3 -m venv {venv_dir}")
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
        print(f"❌ pip executable not found at {pip_executable}.")
        return False
    result = run_command(f"{python_executable} -m pip install --upgrade pip", cwd=venv_dir)
    if result is None:
        print("✅ pip upgraded successfully.")
        return True
    else:
        print("❌ Failed to upgrade pip.")
        return False

def install_and_verify_package(package_name, venv_dir, install_command=None, import_name=None):
    """
    Install a package using pip from the virtual environment and verify its installation.
    """
    print(f"\n📦 Installing package '{package_name}'...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"❌ pip executable not found at {pip_executable}.")
        return False

    # Use custom install command if provided
    if install_command is None:
        install_command = f"{pip_executable} install {package_name}"

    # Install the package
    try:
        run_command(install_command, cwd=venv_dir)
        print(f"✅ Package '{package_name}' installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install package '{package_name}'. Error: {e}")
        return False

    # Verify installation by attempting to import the package
    try:
        # Use custom import name if provided
        if import_name is None:
            package_import_name = package_name.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
            if package_import_name.lower() == 'pillow':
                package_import_name = 'PIL'
            elif package_import_name.lower() == 'moviepy':
                package_import_name = 'moviepy.editor'
            elif package_import_name.lower() == 'tk':
                package_import_name = 'tkinter'
            elif package_import_name.lower().startswith('audioldm2'):
                package_import_name = 'audioldm2'
            else:
                package_import_name = package_import_name.replace('-', '_')
        else:
            package_import_name = import_name

        command = f"{python_executable} -c 'import {package_import_name}'"
        run_command(command, cwd=venv_dir)
        print(f"✅ Package '{package_name}' verified successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to verify package '{package_name}'. Error: {e}")
        return False

def install_audioldm2(venv_dir):
    """
    Install audioldm2 from GitHub.
    """
    print("\n📦 Installing audioldm2 from GitHub...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    install_command = f"{pip_executable} install git+https://github.com/haoheliu/AudioLDM2.git"
    try:
        run_command(install_command, cwd=venv_dir)
        print("✅ audioldm2 installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install audioldm2. Error: {e}")
        return False

    # Verify installation
    try:
        command = f"{python_executable} -c 'import audioldm2'"
        run_command(command, cwd=venv_dir)
        print("✅ audioldm2 verified successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to verify audioldm2. Error: {e}")
        return False

def install_diffusers_transformers_accelerate(venv_dir):
    """
    Install and upgrade diffusers from GitHub, and install transformers and accelerate.
    """
    print("\n📦 Installing and upgrading diffusers from GitHub...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    diffusers_install_command = f"{pip_executable} install --upgrade git+https://github.com/huggingface/diffusers.git"
    try:
        run_command(diffusers_install_command, cwd=venv_dir)
        print("✅ diffusers installed and upgraded successfully.")
    except Exception as e:
        print(f"❌ Failed to install/upgrade diffusers. Error: {e}")
        return False

    # Install transformers
    print("\n📦 Installing transformers...")
    transformers_install_command = f"{pip_executable} install transformers"
    try:
        run_command(transformers_install_command, cwd=venv_dir)
        print("✅ transformers installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install transformers. Error: {e}")
        return False

    # Install accelerate
    print("\n📦 Installing accelerate...")
    accelerate_install_command = f"{pip_executable} install accelerate"
    try:
        run_command(accelerate_install_command, cwd=venv_dir)
        print("✅ accelerate installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install accelerate. Error: {e}")
        return False

    # Upgrade transformers and diffusers to avoid potential errors
    print("\n🔄 Upgrading transformers and diffusers to ensure compatibility and avoid potential errors...")
    upgrade_command = f"{pip_executable} install --upgrade transformers diffusers"
    try:
        run_command(upgrade_command, cwd=venv_dir)
        print("✅ transformers and diffusers upgraded successfully.")
    except Exception as e:
        print(f"❌ Failed to upgrade transformers and diffusers. Error: {e}")
        return False

    # Verify installations
    try:
        command = f"{python_executable} -c 'import diffusers; import transformers; import accelerate'"
        run_command(command, cwd=venv_dir)
        print("✅ diffusers, transformers, and accelerate verified successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to verify diffusers, transformers, or accelerate. Error: {e}")
        return False

def install_torch_packages(venv_dir, cuda_version):
    """
    Install PyTorch and related packages with CUDA support.
    """
    print("\n🔧 Installing PyTorch and related packages with CUDA support...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"❌ pip executable not found at {pip_executable}.")
        return False

    # Define the appropriate index-url based on CUDA version
    cuda_version_map = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "11.7": "https://download.pytorch.org/whl/cu117",
        "11.6": "https://download.pytorch.org/whl/cu116",
        "11.3": "https://download.pytorch.org/whl/cu113",
        "10.2": "https://download.pytorch.org/whl/cu102",
    }

    if cuda_version not in cuda_version_map:
        print(f"⚠️ Unsupported CUDA version: {cuda_version}. Falling back to CPU-only installation.")
        install_command = f"{pip_executable} install torch torchvision torchaudio"
    else:
        index_url = cuda_version_map[cuda_version]
        install_command = f"{pip_executable} install torch torchvision torchaudio --index-url {index_url}"

    try:
        run_command(install_command, cwd=venv_dir)
        print("✅ PyTorch and related packages installed successfully with CUDA support.")
    except Exception as e:
        print(f"❌ Failed to install PyTorch with CUDA support. Error: {e}")
        return False

    # Verify installation by checking CUDA availability in PyTorch
    try:
        command = f"{python_executable} -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"
        output = run_command(command, capture_output=True, cwd=venv_dir)
        if output:
            print(f"🖥️ PyTorch Output:\n{output}")
            # Parse the torch version and CUDA availability
            lines = output.splitlines()
            if len(lines) >= 2:
                installed_version = lines[0].strip()
                cuda_available = lines[1].strip()
                print(f"✅ PyTorch Version: {installed_version}, CUDA Available: {cuda_available}")
                if cuda_available.lower() == "true":
                    print("✅ CUDA is available for PyTorch.")
                else:
                    print("⚠️ CUDA is not available for PyTorch. Please check your CUDA installation.")
            else:
                print("⚠️ Unexpected output during PyTorch verification.")
        else:
            print("⚠️ No output received during PyTorch verification.")
    except Exception as e:
        print(f"❌ Failed to verify PyTorch installation. Error: {e}")
        return False

    return True

def install_additional_torch_packages(venv_dir):
    """
    Install additional torch-related packages.
    """
    print("\n📦 Installing additional torch-related packages...")
    additional_torch_packages = [
        {"name": "torch-audiomentations==0.11.1"},
        {"name": "torch-pitch-shift==1.2.4"},
        {"name": "torch-stoi==0.2.2"},
        {"name": "torchcrepe==0.0.20"},
        {"name": "torchdiffeq==0.2.4"},
        {"name": "torchfcpe==0.0.4"},
        {"name": "torchgen==0.0.1"},
        {"name": "torchlibrosa==0.1.0"},
        {"name": "torchmetrics==0.11.4"},
        {"name": "torchsde==0.2.6"},
        # Add other packages as needed
    ]

    for pkg in additional_torch_packages:
        name = pkg["name"]
        if not install_and_verify_package(name, venv_dir):
            print(f"\n❌ Failed to install and verify package '{name}'. Exiting setup.")
            return False

    print("✅ All additional torch-related packages installed and verified successfully.")
    return True

def install_ffmpeg_linux():
    """
    Install FFmpeg using the package manager.
    """
    print("\n🔧 Installing FFmpeg...")
    distro = get_linux_distro()
    if distro in ['ubuntu', 'debian']:
        install_command = "sudo apt-get update && sudo apt-get install -y ffmpeg"
    elif distro in ['fedora']:
        install_command = "sudo dnf install -y ffmpeg ffmpeg-devel"
    elif distro in ['centos', 'rhel']:
        install_command = "sudo yum install -y epel-release && sudo yum install -y ffmpeg ffmpeg-devel"
    else:
        print(f"❌ Unsupported Linux distribution: {distro}. Please install FFmpeg manually.")
        return False

    result = run_command(install_command)
    if result is None:
        print("✅ FFmpeg installed successfully.")
        print("⚠️ **IMPORTANT:** Please **RESTART YOUR TERMINAL OR SYSTEM** for the changes to take effect.")
        return True
    else:
        print("❌ Failed to install FFmpeg.")
        return False

def install_ffmpeg(script_dir):
    """
    Install FFmpeg automatically if not installed.
    """
    if check_ffmpeg():
        return True
    else:
        return install_ffmpeg_linux()

def check_ffmpeg(cwd=None):
    """
    Check if FFmpeg is installed.
    """
    print("\n🔍 Checking for FFmpeg installation...")
    try:
        result = run_command("ffmpeg -version", capture_output=True, cwd=cwd)
        if result:
            print("✅ FFmpeg is installed.")
            return True
    except:
        pass
    print("⚠️ FFmpeg is not installed.")
    return False

def create_env_file(script_dir):
    """
    Create a default .env file if it doesn't exist.
    """
    env_path = script_dir / '.env'
    if env_path.exists():
        print("\nℹ️ .env file already exists. Skipping creation.")
        return
    print("\n📝 Creating a default .env file...")
    default_env = {
        "COMFYUI_PROMPTS_FOLDER": "path_to_comfyui_prompts_folder",
        "LAST_USED_DIRECTORY": "path_to_last_used_directory"
    }
    with open(env_path, 'w') as f:
        for key, value in default_env.items():
            f.write(f"{key}={value}\n")
    print("✅ .env file created with placeholder values. Please edit it to include your actual configuration.")

def create_default_json_files(script_dir):
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
        file_path = script_dir / file_name
        if not file_path.exists() or file_path.stat().st_size == 0:
            print(f"\n📝 Creating default '{file_name}'...")
            with open(file_path, 'w') as f:
                json.dump(default_content, f, indent=4)
            print(f"✅ '{file_name}' created with default content.")
        else:
            # Optionally, validate JSON structure here
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                print(f"ℹ️ '{file_name}' exists and contains valid JSON.")
            except json.JSONDecodeError:
                print(f"\n⚠️ '{file_name}' is corrupted or contains invalid JSON. Recreating with default content...")
                with open(file_path, 'w') as f:
                    json.dump(default_content, f, indent=4)
                print(f"✅ '{file_name}' has been recreated with default content.")

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

def install_torch_packages(venv_dir, cuda_version):
    """
    Install PyTorch and related packages with CUDA support.
    """
    print("\n🔧 Installing PyTorch and related packages with CUDA support...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"❌ pip executable not found at {pip_executable}.")
        return False

    # Define the appropriate index-url based on CUDA version
    cuda_version_map = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "11.7": "https://download.pytorch.org/whl/cu117",
        "11.6": "https://download.pytorch.org/whl/cu116",
        "11.3": "https://download.pytorch.org/whl/cu113",
        "10.2": "https://download.pytorch.org/whl/cu102",
    }

    if cuda_version not in cuda_version_map:
        print(f"⚠️ Unsupported CUDA version: {cuda_version}. Falling back to CPU-only installation.")
        install_command = f"{pip_executable} install torch torchvision torchaudio"
    else:
        index_url = cuda_version_map[cuda_version]
        install_command = f"{pip_executable} install torch torchvision torchaudio --index-url {index_url}"

    try:
        run_command(install_command, cwd=venv_dir)
        print("✅ PyTorch and related packages installed successfully with CUDA support.")
    except Exception as e:
        print(f"❌ Failed to install PyTorch with CUDA support. Error: {e}")
        return False

    # Verify installation by checking CUDA availability in PyTorch
    try:
        command = f"{python_executable} -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"
        output = run_command(command, capture_output=True, cwd=venv_dir)
        if output:
            print(f"🖥️ PyTorch Output:\n{output}")
            # Parse the torch version and CUDA availability
            lines = output.splitlines()
            if len(lines) >= 2:
                installed_version = lines[0].strip()
                cuda_available = lines[1].strip()
                print(f"✅ PyTorch Version: {installed_version}, CUDA Available: {cuda_available}")
                if cuda_available.lower() == "true":
                    print("✅ CUDA is available for PyTorch.")
                else:
                    print("⚠️ CUDA is not available for PyTorch. Please check your CUDA installation.")
            else:
                print("⚠️ Unexpected output during PyTorch verification.")
        else:
            print("⚠️ No output received during PyTorch verification.")
    except Exception as e:
        print(f"❌ Failed to verify PyTorch installation. Error: {e}")
        return False

    return True

def main():
    print("============================================")
    print("    🌌 Welcome to Temporal Setup 🌌       ")
    print("   Your powerful Audio/Video Prompting     ")
    print("               Engine Setup                ")
    print("============================================\n")

    # Define the main script path relative to setup.py
    script_dir = Path(__file__).parent.resolve()
    main_script = "TemporalPromptEngine.py"
    main_script_path = script_dir / main_script

    if not main_script_path.exists():
        print(f"\n❌ Main script '{main_script}' not found at {main_script_path}. Please ensure it exists.")
        sys.exit(1)

    # Step 1: Check and install Git
    if not check_git(cwd=script_dir):
        # Attempt to install Git automatically
        print("\n🔧 Attempting to install Git automatically...")
        if install_git_linux():
            print("✅ Git installed and verified successfully.")
        else:
            # Prompt user to install Git manually
            print("\n⚠️ Automatic Git installation failed.")
            prompt_install_git_linux()
            print("\n🔧 Please install Git and then re-run the SETUP script to continue.")
            sys.exit(1)

    # Step 2: Check for NVIDIA GPU
    print("\n🔍 Checking for NVIDIA GPU...")
    try:
        gpu_info = run_command("nvidia-smi --query-gpu=name --format=csv,noheader", capture_output=True, cwd=script_dir)
        if not gpu_info:
            print("❌ No NVIDIA GPU detected. A CUDA-enabled NVIDIA GPU is required.")
            sys.exit(1)
        else:
            print(f"✅ NVIDIA GPU detected: {gpu_info.strip()}")
    except:
        print("❌ nvidia-smi not found. Please ensure that NVIDIA drivers are installed.")
        sys.exit(1)

    # Step 3: Check for Ollama
    if not check_ollama(cwd=script_dir):
        if not prompt_install_ollama_linux():
            print("\n🔧 Please install Ollama and then re-run the setup script to continue.")
            sys.exit(1)

    # Step 4: Check for CUDA toolkit
    cuda_version = check_cuda_toolkit(cwd=script_dir)
    if not cuda_version:
        if not prompt_install_cuda_linux():
            print("\n🔧 Please install the CUDA Toolkit and then re-run the setup script to continue.")
            sys.exit(1)
        # After installation, recheck the CUDA version
        cuda_version = check_cuda_toolkit(cwd=script_dir)
        if not cuda_version:
            print("\n❌ CUDA Toolkit installation verification failed. Exiting setup.")
            sys.exit(1)

    # Step 5: Install FFmpeg
    if not check_ffmpeg(cwd=script_dir):
        if not install_ffmpeg(script_dir):
            print("\n🔧 Please install FFmpeg and then re-run the setup script to continue.")
            sys.exit(1)

    # Step 6: Set up virtual environment
    venv_dir = script_dir / "TemporalPromptEngineEnv"
    if not venv_dir.exists():
        if not create_virtualenv(venv_dir):
            print("\n❌ Failed to create virtual environment. Exiting setup.")
            sys.exit(1)

    # Step 7: Upgrade pip
    if not upgrade_pip(venv_dir):
        print("\n❌ Failed to upgrade pip. Exiting setup.")
        sys.exit(1)

    # Get the virtual environment executables
    python_executable, pip_executable = get_venv_executables(venv_dir)

    # Step 8: Install and verify required packages individually
    packages_to_install = [
        {"name": "python-dotenv>=1.0.1"},
        {"name": "moviepy>=1.0.3", "import_name": "moviepy.editor"},
        {"name": "pydub>=0.25.1"},
        {"name": "Pillow>=9.0.0", "import_name": "PIL"},
        {"name": "requests>=2.25.1"},
        {"name": "pyperclip>=1.8.2"},
        {"name": "playsound"},
        {"name": "scipy>=1.7.0"},
        {"name": "tk", "import_name": "tkinter"},
        {"name": "accelerate>=0.21.0"},
    ]

    for pkg in packages_to_install:
        name = pkg["name"]
        install_command = pkg.get("install_command")
        import_name = pkg.get("import_name")
        if not install_and_verify_package(name, venv_dir, install_command=install_command, import_name=import_name):
            print(f"\n❌ Failed to install and verify package '{name}'. Exiting setup.")
            sys.exit(1)

    # Step 9: Install audioldm2
    if not install_audioldm2(venv_dir):
        print("\n❌ Failed to install audioldm2. Exiting setup.")
        sys.exit(1)

    # Step 10: Install diffusers, transformers, and accelerate
    if not install_diffusers_transformers_accelerate(venv_dir):
        print("\n❌ Failed to install diffusers, transformers, or accelerate. Exiting setup.")
        sys.exit(1)

    # Step 11: Install torch and related packages with CUDA support
    if not install_torch_packages(venv_dir, cuda_version):
        print("\n❌ Failed to install PyTorch with CUDA support. Exiting setup.")
        sys.exit(1)

    # Step 12: Install additional torch-related packages
    if not install_additional_torch_packages(venv_dir):
        print("\n❌ Failed to install additional torch-related packages. Exiting setup.")
        sys.exit(1)

    # Step 13: Create .env file
    create_env_file(script_dir)

    # Step 14: Create or validate JSON configuration files
    create_default_json_files(script_dir)

    # Step 15: Prompt to launch the main script
    prompt_launch_script(venv_dir, main_script_path)

    print("\n============================================")
    print("       🎉 Temporal Setup Completed! 🎉      ")
    print("  Your Audio/Video Prompting Engine is ready ")
    print("============================================")
    input("\n🔒 Press Enter to exit...")

if __name__ == "__main__":
    main()
