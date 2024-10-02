import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import json
import platform
import shutil
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

def check_ollama(cwd=None):
    """
    Check if Ollama is installed.
    """
    print("\nChecking for Ollama installation...")
    try:
        output = run_command("ollama --version", capture_output=True, cwd=cwd)
        if output:
            print(f"Ollama is installed: {output}")
            return True
    except:
        pass
    print("Ollama is not installed.")
    return False

def prompt_install_ollama(download_url, script_dir):
    """
    Prompt the user to install Ollama.
    """
    print("\nOllama is not installed. Please install it from the following link:")
    print(download_url)
    webbrowser.open(download_url)
    input("Press Enter after you have installed Ollama to continue...")
    # Recheck after installation
    return check_ollama(cwd=script_dir)

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
    Upgrade pip within the virtual environment.
    """
    print("\nUpgrading pip in the virtual environment...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False
    result = run_command(f'"{python_executable}" -m pip install --upgrade pip', cwd=venv_dir)
    if result is None:
        print("pip upgraded successfully.")
        return True
    else:
        print("Failed to upgrade pip.")
        return False

def install_and_verify_package(package_name, venv_dir, install_command=None, import_name=None):
    """
    Install a package using pip from the virtual environment and verify its installation.
    """
    print(f"\nInstalling package '{package_name}'...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False

    # Use custom install command if provided
    if install_command is None:
        install_command = f'"{pip_executable}" install {package_name}'

    # Install the package
    try:
        run_command(install_command, cwd=venv_dir)
        print(f"Package '{package_name}' installed successfully.")
    except Exception as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")
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
            elif package_import_name.lower() == 'pyyaml':
                package_import_name = 'yaml'
            else:
                package_import_name = package_import_name.replace('-', '_')
        else:
            package_import_name = import_name

        command = f'"{python_executable}" -c "import {package_import_name}"'
        run_command(command, capture_output=True, cwd=venv_dir)
        print(f"Package '{package_name}' verified successfully.")
        return True
    except Exception as e:
        print(f"Failed to verify package '{package_name}'. Error: {e}")
        return False

def install_torch(venv_dir):
    """
    Install specific versions of torch, torchvision, and torchaudio with CUDA 11.8 support.
    """
    print("\nInstalling torch, torchvision, and torchaudio with CUDA 11.8 support...")
    python_executable, pip_executable = get_venv_executables(venv_dir)
    if not pip_executable.exists():
        print(f"pip executable not found at {pip_executable}.")
        return False

    # Define the exact versions to install
    torch_version = "torch==2.4.1+cu118"
    torchvision_version = "torchvision==0.19.1+cu118"
    torchaudio_version = "torchaudio==2.4.1+cu118"
    index_url = "https://download.pytorch.org/whl/cu118"

    # Install torch
    try:
        run_command(f'"{pip_executable}" install {torch_version} --index-url {index_url}', cwd=venv_dir)
        print(f"Installed {torch_version} successfully.")
    except Exception as e:
        print(f"Failed to install {torch_version}. Error: {e}")
        return False

    # Install torchvision
    try:
        run_command(f'"{pip_executable}" install {torchvision_version} --index-url {index_url}', cwd=venv_dir)
        print(f"Installed {torchvision_version} successfully.")
    except Exception as e:
        print(f"Failed to install {torchvision_version}. Error: {e}")
        return False

    # Install torchaudio
    try:
        run_command(f'"{pip_executable}" install {torchaudio_version} --index-url {index_url}', cwd=venv_dir)
        print(f"Installed {torchaudio_version} successfully.")
    except Exception as e:
        print(f"Failed to install {torchaudio_version}. Error: {e}")
        return False

    # Verify torch installation
    try:
        command = f'"{python_executable}" -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
        output = run_command(command, capture_output=True, cwd=venv_dir)
        if output:
            print(f"Command output:\n{output}")
            # Parse the torch version and CUDA availability
            lines = output.splitlines()
            if len(lines) >= 2:
                installed_version = lines[0].strip()
                cuda_available = lines[1].strip()
                if installed_version == "2.4.1+cu118" and cuda_available == "True":
                    print("torch verified successfully with CUDA support.")
                    return True
                else:
                    print(f"torch version mismatch or CUDA not available. Installed version: {installed_version}, CUDA Available: {cuda_available}")
                    return False
            else:
                print("Unexpected output during torch verification.")
                return False
        else:
            print("No output received during torch verification.")
            return False
    except Exception as e:
        print(f"Failed to verify torch installation. Error: {e}")
        return False

def create_env_file(script_dir):
    """
    Create a default .env file if it doesn't exist.
    """
    env_path = script_dir / '.env'
    if env_path.exists():
        print("\n.env file already exists. Skipping creation.")
        return
    print("\nCreating a default .env file...")
    default_env = {
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "COMFYUI_PROMPTS_FOLDER": "/home/user/comfyui/prompts",
        "LAST_USED_DIRECTORY": "/home/user/last_used_directory"
    }
    with open(env_path, 'w') as f:
        for key, value in default_env.items():
            f.write(f"{key}={value}\n")
    print(".env file created with placeholder values. Please edit it to include your actual configuration.")

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

def check_ffmpeg(cwd=None):
    """
    Check if ffmpeg is installed.
    """
    print("\nChecking for ffmpeg installation...")
    try:
        result = run_command("ffmpeg -version", capture_output=True, cwd=cwd)
        if result:
            print("ffmpeg is installed.")
            return True
    except:
        pass
    print("ffmpeg is not installed.")
    return False

def prompt_install_ffmpeg(script_dir):
    """
    Prompt the user to install ffmpeg.
    """
    print("\nffmpeg is required for video processing.")
    print("Please install ffmpeg using your package manager. For example:")
    print("Ubuntu/Debian: sudo apt-get install ffmpeg")
    print("Fedora: sudo dnf install ffmpeg")
    print("Arch: sudo pacman -S ffmpeg")
    webbrowser.open("https://ffmpeg.org/download.html")
    input("Press Enter after you have installed ffmpeg to continue...")
    # Recheck after installation
    return check_ffmpeg(cwd=script_dir)

def prompt_launch_script(venv_dir, main_script_path):
    """
    Prompt the user to launch the main Python script.
    """
    choice = input("\nSetup complete. Would you like to launch the script now? (y/n): ").strip().lower()
    if choice == 'y':
        print("\nLaunching the main Python script...")
        try:
            # Use the virtual environment's Python executable to run the main script
            python_executable, _ = get_venv_executables(venv_dir)
            if not python_executable.exists():
                print(f"Virtual environment's Python executable not found at {python_executable}.")
                print("Please ensure the virtual environment is set up correctly.")
                return
            subprocess.run([str(python_executable), str(main_script_path)], check=True)
            print("\nTemporalPromptEngine.py launched successfully.")
        except subprocess.CalledProcessError as e:
            print("\nFailed to launch TemporalPromptEngine.py. Please check the setup and try running the script manually.")
            print(f"Error: {e}")
    else:
        print("\nSetup completed. You can run the script later using the virtual environment's Python.")

def check_cuda_toolkit(cwd=None):
    """
    Check if CUDA toolkit is installed.
    """
    print("\nChecking for CUDA toolkit installation...")
    try:
        cuda_version_output = run_command("nvcc --version", capture_output=True, cwd=cwd)
        if cuda_version_output:
            # Parse the CUDA version from the output
            match = re.search(r"release (\d+\.\d+),", cuda_version_output)
            if match:
                cuda_version = match.group(1)
                print(f"Detected CUDA toolkit version: {cuda_version}")
                return cuda_version
            else:
                print("Could not parse CUDA version from nvcc output.")
                return None
        else:
            print("CUDA toolkit not found.")
            return None
    except:
        print("nvcc not found. CUDA toolkit is not installed.")
        return None

def download_cuda_installer(script_dir):
    """
    Download the CUDA 12.6.2 installer.
    """
    print("\nDownloading CUDA 12.6.2 installer...")
    cuda_url = "https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run"
    installer_path = script_dir / "cuda_12.6.2_560.35.03_linux.run"
    try:
        if installer_path.exists():
            print("CUDA installer already downloaded.")
            return installer_path
        run_command(f"wget {cuda_url} -O {installer_path}", cwd=script_dir)
        if installer_path.exists():
            print("CUDA installer downloaded successfully.")
            return installer_path
        else:
            print("Failed to download CUDA installer.")
            return None
    except Exception as e:
        print(f"Error downloading CUDA installer: {e}")
        return None

def make_installer_executable(installer_path):
    """
    Make the CUDA installer executable.
    """
    print("\nMaking the CUDA installer executable...")
    try:
        run_command(f"chmod +x {installer_path}", cwd=installer_path.parent)
        print("Installer is now executable.")
        return True
    except Exception as e:
        print(f"Failed to make installer executable: {e}")
        return False

def install_cuda(installer_path):
    """
    Run the CUDA installer.
    """
    print("\nInstalling CUDA 12.6.2...")
    try:
        # Run the installer with silent options:
        # --silent: run without user interaction
        # --toolkit: install only the CUDA toolkit
        # --override: override any checks (use with caution)
        install_command = f"sudo sh {installer_path} --silent --toolkit --override"
        run_command(install_command)
        print("CUDA installed successfully.")
        return True
    except Exception as e:
        print(f"Failed to install CUDA: {e}")
        return False

def set_cuda_env():
    """
    Set up CUDA environment variables in .bashrc.
    """
    print("\nSetting up CUDA environment variables...")
    bashrc_path = Path.home() / ".bashrc"
    cuda_path = "/usr/local/cuda-12.6"

    # Check if the PATH is already set
    with open(bashrc_path, 'r') as f:
        bashrc_content = f.read()

    path_export = f"export PATH={cuda_path}/bin:$PATH"
    ld_library_export = f"export LD_LIBRARY_PATH={cuda_path}/lib64:$LD_LIBRARY_PATH"

    if path_export not in bashrc_content:
        run_command(f'echo "{path_export}" >> {bashrc_path}')
        print(f"Added PATH to {bashrc_path}")
    else:
        print("PATH for CUDA is already set in .bashrc.")

    if ld_library_export not in bashrc_content:
        run_command(f'echo "{ld_library_export}" >> {bashrc_path}')
        print(f"Added LD_LIBRARY_PATH to {bashrc_path}")
    else:
        print("LD_LIBRARY_PATH for CUDA is already set in .bashrc.")

    # Source the updated .bashrc
    run_command("source ~/.bashrc")
    print("Environment variables updated.")

def verify_cuda_installation():
    """
    Verify that CUDA was installed correctly.
    """
    print("\nVerifying CUDA installation...")
    try:
        cuda_version_output = run_command("nvcc --version", capture_output=True)
        if cuda_version_output:
            print(f"CUDA installation verified:\n{cuda_version_output}")
            return True
        else:
            print("CUDA installation verification failed.")
            return False
    except Exception as e:
        print(f"Error verifying CUDA installation: {e}")
        return False

def install_cuda_toolkit(script_dir):
    """
    Complete process to install CUDA toolkit.
    """
    installer_path = download_cuda_installer(script_dir)
    if not installer_path:
        print("\nFailed to download CUDA installer. Exiting setup.")
        sys.exit(1)

    if not make_installer_executable(installer_path):
        print("\nFailed to make CUDA installer executable. Exiting setup.")
        sys.exit(1)

    if not install_cuda(installer_path):
        print("\nCUDA installation failed. Exiting setup.")
        sys.exit(1)

    set_cuda_env()

    if not verify_cuda_installation():
        print("\nCUDA installation verification failed. Exiting setup.")
        sys.exit(1)

def install_ollama_linux(script_dir):
    """
    Install Ollama on Linux.
    """
    print("\nInstalling Ollama on Linux...")
    try:
        # Example installation using a hypothetical .sh installer
        # Replace the URL and installation steps with the actual ones provided by Ollama
        ollama_url = "https://ollama.com/download/OllamaSetup.run"
        installer_path = script_dir / "OllamaSetup.run"

        if not installer_path.exists():
            run_command(f"wget {ollama_url} -O {installer_path}", cwd=script_dir)
            if not installer_path.exists():
                print("Failed to download Ollama installer.")
                return False
            run_command(f"chmod +x {installer_path}", cwd=script_dir)
        
        # Run the installer
        run_command(f"sudo sh {installer_path} --silent", cwd=script_dir)
        print("Ollama installed successfully.")
        return True
    except Exception as e:
        print(f"Failed to install Ollama on Linux. Error: {e}")
        return False

def main():
    print("============================================")
    print("     Temporal Prompt Engine Setup Script    ")
    print("============================================\n")

    # Define the main script path relative to setup.py
    script_dir = Path(__file__).parent.resolve()
    main_script = "TemporalPromptEngine.py"
    main_script_path = script_dir / main_script

    if not main_script_path.exists():
        print(f"\nMain script '{main_script}' not found at {main_script_path}. Please ensure it exists.")
        sys.exit(1)

    # Step 1: Check for pip
    if not check_pip(cwd=script_dir):
        if not install_pip(script_dir):
            print("\npip installation failed. Exiting setup.")
            sys.exit(1)

    # Step 2: Check for Ollama
    if not check_ollama(cwd=script_dir):
        # Provide the Linux download URL for Ollama
        ollama_download_url = "https://ollama.com/download/OllamaSetup.run"  # Replace with actual Linux URL
        print("\nOllama is not installed. Attempting to install Ollama on Linux...")
        if not install_ollama_linux(script_dir):
            print("\nOllama installation failed. Please install it manually from the official website.")
            sys.exit(1)
        # Recheck after installation
        if not check_ollama(cwd=script_dir):
            print("\nOllama installation not detected after installation attempt. Exiting setup.")
            sys.exit(1)

    # Step 3: Check for NVIDIA GPU
    try:
        gpu_info = run_command("nvidia-smi --query-gpu=name --format=csv,noheader", capture_output=True, cwd=script_dir)
        if not gpu_info:
            print("No NVIDIA GPU detected. A CUDA-enabled NVIDIA GPU is required.")
            sys.exit(1)
        else:
            print(f"NVIDIA GPU detected: {gpu_info.strip()}")
    except:
        print("nvidia-smi not found. Please ensure that NVIDIA drivers are installed.")
        sys.exit(1)

    # Step 4: Check for CUDA toolkit
    cuda_version = check_cuda_toolkit(cwd=script_dir)
    if not cuda_version:
        print("\nCUDA toolkit is required for this application.")
        print("Installing CUDA toolkit 12.6.2...")
        install_cuda_toolkit(script_dir)
        # Recheck after installation
        cuda_version = check_cuda_toolkit(cwd=script_dir)
        if not cuda_version:
            print("\nCUDA toolkit installation not detected. Exiting setup.")
            sys.exit(1)

    # Step 5: Set up virtual environment
    venv_dir = script_dir / "TemporalPromptEngineEnv"
    if not venv_dir.exists():
        if not create_virtualenv(venv_dir):
            print("\nFailed to create virtual environment. Exiting setup.")
            sys.exit(1)

    # Step 6: Upgrade pip
    if not upgrade_pip(venv_dir):
        print("\nFailed to upgrade pip. Exiting setup.")
        sys.exit(1)

    # Get the virtual environment executables
    python_executable, pip_executable = get_venv_executables(venv_dir)

    # Step 7: Install and verify required packages individually
    packages_to_install = [
        {"name": "python-dotenv>=1.0.1"},
        {"name": "openai>=0.27.0"},
        {"name": "moviepy>=1.0.3", "import_name": "moviepy.editor"},
        {"name": "pydub>=0.25.1"},
        {"name": "Pillow>=9.0.0", "import_name": "PIL"},
        {"name": "requests>=2.25.1"},
        {"name": "pyperclip>=1.8.2"},
        {"name": "playsound"},
        {"name": "scipy>=1.7.0"},
        {"name": "tk", "import_name": "tkinter"},
        {"name": "accelerate>=0.21.0"},
        # Install specific versions of torch, torchvision, torchaudio with CUDA 11.8 support
        {"name": "torch==2.4.1+cu118", "install_command": f'"{pip_executable}" install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118'},
        {"name": "torchvision==0.19.1+cu118", "install_command": f'"{pip_executable}" install torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118'},
        {"name": "torchaudio==2.4.1+cu118", "install_command": f'"{pip_executable}" install torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118'},
        # Install diffusers and transformers with specific versions
        {"name": "diffusers==0.21.1", "install_command": f'"{pip_executable}" install diffusers==0.21.1'},
        {"name": "transformers==4.31.0", "install_command": f'"{pip_executable}" install transformers==4.31.0'},
        # Install audioldm2 from GitHub
        {"name": "audioldm2", "install_command": f'"{pip_executable}" install git+https://github.com/haoheliu/AudioLDM2.git#egg=audioldm2'},
    ]

    for pkg in packages_to_install:
        name = pkg["name"]
        install_command = pkg.get("install_command")
        import_name = pkg.get("import_name")
        if not install_and_verify_package(name, venv_dir, install_command=install_command, import_name=import_name):
            print(f"\nFailed to install and verify package '{name}'. Exiting setup.")
            sys.exit(1)

    # Step 8: (Optional) Install Additional Torch-Related Packages
    # If you have other torch-related packages, add them here with specific versions
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
            print(f"\nFailed to install and verify package '{name}'. Exiting setup.")
            sys.exit(1)

    # Step 9: Create .env file
    create_env_file(script_dir)

    # Step 10: Create or validate JSON configuration files
    create_default_json_files(script_dir)

    # Step 11: Check for ffmpeg
    if not check_ffmpeg(cwd=script_dir):
        if not prompt_install_ffmpeg(script_dir):
            print("\nffmpeg installation not completed. Exiting setup.")
            sys.exit(1)

    # Step 12: Prompt to launch the main script
    prompt_launch_script(venv_dir, main_script_path)

    print("\n============================================")
    print("         Setup Completed Successfully       ")
    print("============================================")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
