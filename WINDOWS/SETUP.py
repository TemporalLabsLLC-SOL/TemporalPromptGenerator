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

def run_command(command, capture_output=False, cwd=None):
    """
    Run a system command with friendly logging.
    """
    print(f"\n🔧 Running command: {command}")
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
            print(f"✅ Command succeeded.")
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=True, cwd=cwd)
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
        output = run_command("pip --version", capture_output=True, cwd=cwd)
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
    print("\n🔧 Attempting to install pip...")
    result = run_command(f'"{sys.executable}" -m ensurepip --upgrade', cwd=script_dir)
    if result is None:
        print("✅ pip installed successfully.")
        return True
    else:
        print("❌ Failed to install pip.")
        return False

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

def install_git_windows(script_dir):
    """
    Automatically install Git on Windows.
    """
    print("\n🔧 Starting Git installation for Windows...")

    # Define Git installer URL for the latest version
    git_download_url = "https://github.com/git-for-windows/git/releases/latest/download/Git-2.42.0-64-bit.exe"

    # Define the path to save the Git installer
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            git_installer_path = tmpdir / "git_installer.exe"

            print(f"📥 Downloading Git from {git_download_url}...")
            urllib.request.urlretrieve(git_download_url, git_installer_path)
            print("✅ Download completed.")

            print("🔧 Running Git installer silently...")
            # Silent installation flags for Git
            # /VERYSILENT: Silent install
            # /NORESTART: Do not restart after install
            # /SP-: Disable the splash screen
            install_command = f'"{git_installer_path}" /VERYSILENT /NORESTART /SP-'
            run_command(install_command, cwd=tmpdir)
            print("✅ Git installed successfully.")

        # Verify Git installation
        return check_git(cwd=script_dir)

    except Exception as e:
        print(f"❌ An error occurred during Git installation: {e}")
        return False

def prompt_install_git(download_url, script_dir):
    """
    Prompt the user to install Git with a friendly message.
    """
    print("\n🚀 Oh! Looks like we need to install Git on your system.")
    choice = input("🔗 Would you like me to open the browser to download Git? (y/n): ").strip().lower()
    if choice == 'y':
        webbrowser.open(download_url)
        print("\n📥 Please install Git from your browser.")
        print("⚠️ **IMPORTANT:** After installation, **RESTART YOUR COMMAND PROMPT OR SYSTEM** for the changes to take effect.")
        print("🔄 Then, please re-run the SETUP script to continue.")
    else:
        print("\n🛑 Installation of Git aborted by the user.")
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

def prompt_install_ollama(download_url, script_dir):
    """
    Prompt the user to install Ollama with a friendly message.
    """
    print("\n🚀 Oh! Looks like we need to install Ollama on your system.")
    choice = input("🔗 Would you like me to open the browser to download Ollama? (y/n): ").strip().lower()
    if choice == 'y':
        webbrowser.open(download_url)
        print("\n📥 Please install Ollama from your browser.")
        print("⚠️ **IMPORTANT:** After installation, **RESTART YOUR COMMAND PROMPT OR SYSTEM** for the changes to take effect.")
        print("🔄 Then, please re-run the SETUP script to continue.")
    else:
        print("\n🛑 Installation of Ollama aborted by the user.")
    return False  # Exiting setup to allow user to install manually

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
    except Exception as e:
        print(f"⚠️ nvcc not found or an error occurred: {e}")
        print("⚠️ CUDA toolkit is not installed.")
        return None

def prompt_install_cuda(download_url, script_dir):
    """
    Prompt the user to install CUDA toolkit with a friendly message.
    """
    print("\n⚡ CUDA Toolkit is required for optimal performance.")
    choice = input("🔗 Would you like me to open the browser to download CUDA Toolkit? (y/n): ").strip().lower()
    if choice == 'y':
        webbrowser.open(download_url)
        print("\n📥 Please install the CUDA Toolkit from your browser.")
        print("⚠️ **IMPORTANT:** After installation, **RESTART YOUR COMMAND PROMPT OR SYSTEM** for the changes to take effect.")
        print("🔄 Then, please re-run the SETUP script to continue.")
    else:
        print("\n🛑 Installation of CUDA Toolkit aborted by the user.")
    return False  # Exiting setup to allow user to install manually

def create_virtualenv(venv_dir):
    """
    Create a virtual environment.
    """
    print(f"\n🔧 Creating a virtual environment in '{venv_dir}'...")
    result = run_command(f'"{sys.executable}" -m venv "{venv_dir}"')
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
    result = run_command(f'"{python_executable}" -m pip install --upgrade pip', cwd=venv_dir)
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
        install_command = f'"{pip_executable}" install {package_name}'

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

        command = f'"{python_executable}" -c "import {package_import_name}"'
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
    install_command = f'"{pip_executable}" install git+https://github.com/haoheliu/AudioLDM2.git'
    try:
        run_command(install_command, cwd=venv_dir)
        print("✅ audioldm2 installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install audioldm2. Error: {e}")
        return False

    # Verify installation
    try:
        command = f'"{python_executable}" -c "import audioldm2"'
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
    diffusers_install_command = f'"{pip_executable}" install --upgrade git+https://github.com/huggingface/diffusers.git'
    try:
        run_command(diffusers_install_command, cwd=venv_dir)
        print("✅ diffusers installed and upgraded successfully.")
    except Exception as e:
        print(f"❌ Failed to install/upgrade diffusers. Error: {e}")
        return False

    # Install transformers
    print("\n📦 Installing transformers...")
    transformers_install_command = f'"{pip_executable}" install transformers'
    try:
        run_command(transformers_install_command, cwd=venv_dir)
        print("✅ transformers installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install transformers. Error: {e}")
        return False

    # Install accelerate
    print("\n📦 Installing accelerate...")
    accelerate_install_command = f'"{pip_executable}" install accelerate'
    try:
        run_command(accelerate_install_command, cwd=venv_dir)
        print("✅ accelerate installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install accelerate. Error: {e}")
        return False

    # Upgrade transformers and diffusers to avoid potential errors
    print("\n🔄 Upgrading transformers and diffusers to ensure compatibility and avoid potential errors...")
    upgrade_command = f'"{pip_executable}" install --upgrade transformers diffusers'
    try:
        run_command(upgrade_command, cwd=venv_dir)
        print("✅ transformers and diffusers upgraded successfully.")
    except Exception as e:
        print(f"❌ Failed to upgrade transformers and diffusers. Error: {e}")
        return False

    # Verify installations
    try:
        command = f'"{python_executable}" -c "import diffusers; import transformers; import accelerate"'
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
        install_command = f'"{pip_executable}" install torch torchvision torchaudio'
    else:
        index_url = cuda_version_map[cuda_version]
        install_command = f'"{pip_executable}" install torch torchvision torchaudio --index-url {index_url}'

    try:
        run_command(install_command, cwd=venv_dir)
        print("✅ PyTorch and related packages installed successfully with CUDA support.")
    except Exception as e:
        print(f"❌ Failed to install PyTorch with CUDA support. Error: {e}")
        return False

    # Verify installation by checking CUDA availability in PyTorch
    try:
        command = f'"{python_executable}" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
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

def upgrade_transformers_diffusers(venv_dir):
    """
    Upgrade transformers and diffusers to ensure the latest versions are installed.
    """
    print("\n🔄 Upgrading transformers and diffusers to the latest versions...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    upgrade_command = f'"{pip_executable}" install --upgrade transformers diffusers'
    try:
        run_command(upgrade_command, cwd=venv_dir)
        print("✅ transformers and diffusers upgraded successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to upgrade transformers and diffusers. Error: {e}")
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
        install_command = f'"{pip_executable}" install torch torchvision torchaudio'
    else:
        index_url = cuda_version_map[cuda_version]
        install_command = f'"{pip_executable}" install torch torchvision torchaudio --index-url {index_url}'

    try:
        run_command(install_command, cwd=venv_dir)
        print("✅ PyTorch and related packages installed successfully with CUDA support.")
    except Exception as e:
        print(f"❌ Failed to install PyTorch with CUDA support. Error: {e}")
        return False

    # Verify installation by checking CUDA availability in PyTorch
    try:
        command = f'"{python_executable}" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
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

def install_torch_packages_corrected(venv_dir, cuda_version):
    """
    Corrected function to install PyTorch with CUDA support.
    """
    # This function has been updated to avoid duplicate definitions.
    return install_torch_packages(venv_dir, cuda_version)

def install_torch_packages_final(venv_dir, cuda_version):
    """
    Finalized function to install PyTorch with CUDA support.
    """
    return install_torch_packages(venv_dir, cuda_version)

def install_torch_packages(venv_dir, cuda_version):
    """
    Install PyTorch and related packages with CUDA support.
    """
    # Already defined above
    pass

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

def install_ffmpeg_windows(script_dir):
    """
    Automatically install FFmpeg on Windows.
    """
    print("\n🔧 Starting FFmpeg installation for Windows...")

    # Define FFmpeg download URL for the latest static build
    ffmpeg_download_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

    # Define the directory where FFmpeg will be installed
    ffmpeg_dir = script_dir / "ffmpeg"

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            zip_path = tmpdir / "ffmpeg.zip"

            print(f"📥 Downloading FFmpeg from {ffmpeg_download_url}...")
            urllib.request.urlretrieve(ffmpeg_download_url, zip_path)
            print("✅ Download completed.")

            print(f"📦 Extracting FFmpeg to temporary directory {tmpdir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            print("✅ Extraction completed.")

            # Find the extracted FFmpeg folder (it usually contains 'bin', 'doc', etc.)
            extracted_folders = [f for f in tmpdir.iterdir() if f.is_dir() and "ffmpeg" in f.name.lower()]
            if not extracted_folders:
                print("❌ Failed to locate the FFmpeg extracted folder.")
                return False
            ffmpeg_extracted = extracted_folders[0]

            # Move the extracted FFmpeg to the desired directory
            if ffmpeg_dir.exists():
                print(f"🗑️ FFmpeg directory {ffmpeg_dir} already exists. Removing it...")
                shutil.rmtree(ffmpeg_dir)
            print(f"📂 Moving FFmpeg to {ffmpeg_dir}...")
            shutil.move(str(ffmpeg_extracted), str(ffmpeg_dir))
            print("✅ FFmpeg moved successfully.")

        # Add FFmpeg to PATH
        ffmpeg_bin = ffmpeg_dir / "bin"
        current_path = os.environ.get("PATH", "")
        ffmpeg_bin_str = str(ffmpeg_bin)

        if ffmpeg_bin_str not in current_path:
            print(f"🔧 Adding FFmpeg to system PATH...")
            # Add to user PATH
            subprocess.run(f'setx PATH "%PATH%;{ffmpeg_bin_str}"', shell=True, check=True)
            print("✅ FFmpeg has been added to the system PATH.")
            print("⚠️ **IMPORTANT:** Please **RESTART YOUR COMMAND PROMPT OR SYSTEM** for the changes to take effect.")
        else:
            print("ℹ️ FFmpeg bin directory is already in the system PATH.")

        return True

    except Exception as e:
        print(f"❌ An error occurred during FFmpeg installation: {e}")
        return False

def install_ffmpeg(script_dir):
    """
    Install FFmpeg automatically if not installed.
    """
    system = platform.system()
    if system == "Windows":
        success = install_ffmpeg_windows(script_dir)
        if success:
            # Recheck after installation
            return check_ffmpeg(cwd=script_dir)
        else:
            print("\n❌ FFmpeg installation failed.")
            return False
    else:
        print("\n🔧 Automated FFmpeg installation is only supported on Windows by this script.")
        print("📥 Please install FFmpeg manually from the official website:")
        print("🌐 https://ffmpeg.org/download.html")
        choice = input("🔗 Would you like me to open the browser to download FFmpeg? (y/n): ").strip().lower()
        if choice == 'y':
            webbrowser.open("https://ffmpeg.org/download.html")
            print("\n📥 Please install FFmpeg from your browser.")
            print("⚠️ **IMPORTANT:** After installation, **RESTART YOUR COMMAND PROMPT OR SYSTEM** for the changes to take effect.")
            print("🔄 Then, please re-run the SETUP script to continue.")
        else:
            print("\n🛑 Installation of FFmpeg aborted by the user.")
        return False  # Exiting setup to allow user to install manually

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
        if install_git_windows(script_dir):
            print("✅ Git installed and verified successfully.")
        else:
            # Prompt user to install Git manually
            print("\n⚠️ Automatic Git installation failed.")
            prompt_install_git("https://git-scm.com/download/win", script_dir)
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
        if not prompt_install_ollama("https://ollama.com/download/OllamaSetup.exe", script_dir):
            print("\n🔧 Please install Ollama and then re-run the setup script to continue.")
            sys.exit(1)

    # Step 4: Check for CUDA toolkit
    cuda_version = check_cuda_toolkit(cwd=script_dir)
    if not cuda_version:
        if not prompt_install_cuda("https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe", script_dir):
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
    if not install_torch_packages_final(venv_dir, cuda_version):
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
