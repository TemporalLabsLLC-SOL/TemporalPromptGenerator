#!/bin/bash

# =====================================================
#        🤖 Temporal Lab: Time Portal Verification 🤖
# =====================================================

# Initialize color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print the welcome banner with ASCII Art
print_banner() {
    echo -e "${CYAN}"
    echo "============================================"
    echo "     🚀 Temporal Lab: Time Portal Setup 🚀    "
    echo "============================================"
    echo "           🕰️ Preparing for Expedition 🕰️        "
    echo "============================================"
    echo -e "${NC}"
}

# Function to print status messages
print_status() {
    echo -e "${GREEN}✔️  $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}❌  $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to print action messages
print_action() {
    echo -e "${MAGENTA}🔧  $1${NC}"
}

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    version=$1
    expected=$2
    py_version=$(py -$version --version 2>&1)
    if [[ "$py_version" == *"Python $expected"* ]]; then
        print_status "Temporal Python Engine v$expected is active and ready."
    else
        print_error "Temporal Python Engine v$expected is missing."
        handle_failure "python"
    fi
}

# Function to check NVIDIA GPU and CUDA
check_nvidia_cuda() {
    if command_exists nvidia-smi; then
        print_status "NVIDIA Temporal Processing Unit (TPU) detected and operational."
        nvcc_version=$(nvcc --version 2>&1)
        if [[ "$nvcc_version" == *"release 12."* ]]; then
            print_status "CUDA Toolkit 12.x for temporal computations is installed correctly."
        else
            print_error "CUDA Toolkit 12.x is not detected or improperly installed."
            handle_failure "cuda"
        fi
    else
        print_error "No NVIDIA Temporal Processing Unit detected or NVIDIA drivers are not installed."
        handle_failure "gpu"
    fi
}

# Function to check HuggingFace API key setup
check_hf_api_key() {
    if [ -f ~/.huggingface/token ]; then
        print_status "HuggingFace API key is securely configured."
    else
        print_warning "Please have your HuggingFace Token ready when initiating the Temporal Prompt Engine's first sequence."
    fi
}

# Function to handle failures
handle_failure() {
    case "$1" in
        python)
            echo ""
            print_info "🔍 System Check: Temporal Python Engine"
            echo ""
            print_action "Temporal Python Engine: Version 3.10.9"
            echo "🕒 Step 1: Access the Python Time Stream:"
            echo "       Visit the Python Downloads portal: https://www.python.org/downloads/release/python-309/"
            echo "🕒 Step 2: Download the Python 3.10.9 installer compatible with Windows."
            echo "🕒 Step 3: Initiate the Temporal Installation:"
            echo "       - Execute the Installer: Double-click the downloaded python-3.10.9-amd64.exe file."
            echo "       - **Crucial Temporal Configuration:**"
            echo "           If Python already exists in your system timeline:"
            echo "               Be cautious when adding Python to PATH to prevent temporal overlaps with existing Python versions."
            echo "               To maintain temporal integrity, install Python 3.10.9 without altering the PATH. Use the Python Launcher for Windows (py) to specify the version during commands."
            echo "           If Python is absent from your system timeline:"
            echo "               Select the 'Add Python to PATH' option at the installer’s interface."
            echo "               Click 'Install Now' to integrate Python into the temporal continuum."
            echo "🕒 Step 4: Confirm Temporal Alignment:"
            echo "       Open a new Command Prompt and execute:"
            echo "           py -3.10 --version"
            echo "       The response should confirm Python 3.10.9's presence."
            echo ""
            ask_to_open_link "https://www.python.org/downloads/release/python-309/"
            ;;
        git)
            echo ""
            print_info "🔍 System Check: Git Temporal Connectivity"
            echo ""
            print_action "Git: Temporal Version Control Installed and Configured"
            echo "🕒 Step 1: Acquire Git from the Temporal Repository: https://git-scm.com/download/win"
            echo "🕒 Step 2: Execute the installer, following the default temporal settings."
            echo "🕒 Step 3: Verify Temporal Connectivity:"
            echo "       Open a new Command Prompt and execute:"
            echo "           git --version"
            echo "       The response should display the installed Git version."
            echo ""
            ask_to_open_link "https://git-scm.com/download/win"
            ;;
        gpu)
            echo ""
            print_info "🔍 System Check: NVIDIA Temporal Processing Unit"
            echo ""
            print_action "NVIDIA Temporal Processing Unit: CUDA-enabled NVIDIA GPU for optimal temporal performance"
            echo "🕒 Step 1: Confirm possession of an NVIDIA GPU equipped for temporal computations with the latest drivers."
            echo "       Download NVIDIA Drivers: https://www.nvidia.com/Download/index.aspx"
            echo "🕒 Step 2: Install the latest drivers following the on-screen temporal instructions."
            echo "🕒 Step 3: Reboot your system to synchronize temporal changes."
            echo ""
            ask_to_open_link "https://www.nvidia.com/Download/index.aspx"
            ;;
        cuda)
            echo ""
            print_info "🔍 System Check: CUDA Toolkit Integration"
            echo ""
            print_action "CUDA Toolkit: Version 12.5 for Temporal Operations"
            echo "🕒 Step 1: Install the CUDA Toolkit for temporal computations:"
            echo "       Download the CUDA Toolkit from CUDA Toolkit Download: https://developer.nvidia.com/cuda-downloads"
            echo "🕒 Step 2: Execute the installer and follow the on-screen temporal instructions."
            echo "🕒 Step 3: Confirm Temporal Integration:"
            echo "       Open a new Command Prompt and execute:"
            echo "           nvcc --version"
            echo "       The response should display CUDA compilation tools version information."
            echo ""
            ask_to_open_link "https://developer.nvidia.com/cuda-downloads"
            ;;
        ffmpeg)
            echo ""
            print_info "🔍 System Check: FFmpeg Temporal Media Processor"
            echo ""
            print_action "FFmpeg: Temporal Media Processor Installed and Integrated into System PATH"
            echo "🕒 Step 1: Acquire FFmpeg:"
            echo "       Download the latest temporal build: https://ffmpeg.org/download.html#build-windows"
            echo "🕒 Step 2: Extract the temporal archive using any compatible archiver like WinRAR or 7z."
            echo "🕒 Step 3: Rename the extracted folder to 'ffmpeg' and place it in the root of the C: drive."
            echo "       Execute the following Command Prompt command to integrate FFmpeg into the temporal PATH:"
            echo "           setx /m PATH \"C:\\ffmpeg\\bin;%PATH%\""
            echo "🕒 Step 4: Confirm Temporal Integration:"
            echo "       Open a new Command Prompt and execute:"
            echo "           ffmpeg -version"
            echo "       The response should display FFmpeg version information."
            echo ""
            ask_to_open_link "https://ffmpeg.org/download.html#build-windows"
            ;;
        ollama)
            echo ""
            print_info "🔍 System Check: Ollama Temporal AI Assistant"
            echo ""
            print_action "Ollama: Temporal AI Assistant Installed"
            echo "🕒 Step 1: Acquire Ollama from the Temporal Downloads portal: https://ollama.com/download"
            echo "🕒 Step 2: Execute the installer and follow the on-screen temporal instructions to integrate Ollama."
            echo "🕒 Step 3: Confirm Temporal Integration:"
            echo "       Open a new Command Prompt and execute:"
            echo "           ollama --version"
            echo "       The response should display the installed Ollama version."
            echo ""
            ask_to_open_link "https://ollama.com/download"
            ;;
        *)
            print_error "Unknown temporal failure type: $1"
            ;;
    esac
}

# Function to ask user to open a link
ask_to_open_link() {
    read -p "${MAGENTA}🔗 Would you like me to open the installer link in your browser now? (y/n): ${NC}" response
    case "$response" in
        y|Y )
            # Open the URL using the default browser
            cmd.exe /c start "" "$1"
            print_info "I've opened the installer link in your default browser for you."
            ;;
        n|N )
            print_info "No worries! You can manually open the installer link later when you're ready to proceed."
            ;;
        * )
            print_warning "I didn't quite catch that. Please enter 'y' or 'n'."
            ask_to_open_link "$1"
            ;;
    esac
    echo ""
    print_info "Once you've completed the steps, please restart this setup script to continue."
    exit 1
}

# Function to perform all checks
perform_checks() {
    # Check for Windows environment (using Git Bash or WSL)
    if [[ $(uname -s) != *"NT"* ]]; then
        print_error "🛑 Temporal Environment Error: This script must be run in a Windows environment (using Git Bash or WSL)."
        exit 1
    else
        print_status "Windows temporal environment detected and confirmed."
    fi

    echo ""

    # Check Python installations
    print_info "🔄 Initializing connection to Python Time Streams..."
    check_python_version 3.10 3.10.9
    check_python_version 3.12 3.12.4

    echo ""

    # Check Git installation
    print_info "🔄 Establishing Temporal Git Connectivity..."
    if command_exists git; then
        print_status "Git Temporal Repository detected and operational."
    else
        print_error "Git Temporal Repository not found."
        handle_failure "git"
    fi

    echo ""

    # Check NVIDIA GPU and CUDA
    print_info "🔄 Engaging NVIDIA Temporal Processing Unit and CUDA Toolkit..."
    check_nvidia_cuda

    echo ""

    # Check FFmpeg installation
    print_info "🔄 Scanning Temporal Media Processor (FFmpeg)..."
    if command_exists ffmpeg; then
        print_status "FFmpeg Temporal Media Processor is active and integrated."
    else
        print_error "FFmpeg Temporal Media Processor not found or not integrated into PATH."
        handle_failure "ffmpeg"
    fi

    echo ""

    # Check Ollama installation
    print_info "🔄 Verifying Ollama Temporal AI Assistant..."
    if command_exists ollama; then
        print_status "Ollama Temporal AI Assistant is operational and ready."
    else
        print_error "Ollama Temporal AI Assistant not found."
        handle_failure "ollama"
    fi

    echo ""

    # Check HuggingFace API key setup
    print_info "🔄 Assessing HuggingFace API Key for Temporal Integration..."
    check_hf_api_key

    echo ""

    # Final message
    echo -e "${CYAN}"
    echo "============================================"
    echo "  🌌 Temporal Setup Verification Complete! 🌌"
    echo "============================================"
    echo -e "${NC}"
    # Example usage of colored echo
    echo -e "${BLUE}All systems are ${GREEN}stable${BLUE}. You may proceed to access the Time Portal Setup.${NC}"
}

confirm_and_run_setup() {
    echo ""
    read -p "🚀 Are you prepared to access your new Time Portal Access Tool? Shall I initiate the temporal setup now? (y/n): " choice

    case "$choice" in
        y|Y )
            print_info "🌟 Launching the Time Portal Access Tool Setup with Python 3.10... 🔥"
            # Directly execute the Python setup script
            py -3.10 SETUP.py
            
            # Check if the script was executed successfully
            if [ $? -eq 0 ]; then
                print_status "Temporal setup script executed successfully."
                echo -e "${GREEN}✨ Your Time Portal Access Tool is now active!✨${NC}"
            else
                print_error "Temporal setup script encountered anomalies. Please review the logs for further analysis."
            fi
            ;;
        n|N )
            print_info "🛑 Understood. You can initiate the setup later by running 'py -3.10 SETUP.py' when you're ready to access the Time Portal."
            ;;
        * )
            print_warning "❓ I didn't quite catch that. Please enter 'y' or 'n'."
            confirm_and_run_setup
            ;;
    esac
}



# Main script execution
main() {
    print_banner
    echo ""
    print_info "🎬 Welcome, Time Traveler! I'm Chronos, your personal assistant for the Temporal Lab's cinematic expedition setup. 🤖"
    print_info "🌌 Let's ensure all systems are aligned to activate the Temporal Prompt Engine's time-altering capabilities and prepare for your grand expedition."
    echo ""
    perform_checks
    echo ""
    # Prompt to run the setup batch script
    confirm_and_run_setup
}

# Run the main function
main

