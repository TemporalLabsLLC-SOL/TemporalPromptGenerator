#!/bin/bash

# =====================================================
#        Temporal Lab: Time Portal Verification
# =====================================================

# Initialize color variables
RED=$(tput setaf 1 2>/dev/null || echo "")
GREEN=$(tput setaf 2 2>/dev/null || echo "")
YELLOW=$(tput setaf 3 2>/dev/null || echo "")
BLUE=$(tput setaf 4 2>/dev/null || echo "")
MAGENTA=$(tput setaf 5 2>/dev/null || echo "")
CYAN=$(tput setaf 6 2>/dev/null || echo "")
NC=$(tput sgr0 2>/dev/null || echo "") # No Color

# Function to print the welcome banner
print_banner() {
    echo -e "${CYAN}"
    echo "============================================"
    echo "     Temporal Lab: Time Portal Setup     "
    echo "============================================"
    echo "           Preparing for Expedition         "
    echo "============================================"
    echo -e "${NC}"
}

# Function to print status messages
print_status() {
    echo -e "${GREEN}[OK] $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Function to print action messages
print_action() {
    echo -e "${MAGENTA}[ACTION] $1${NC}"
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
        print_status "Python $expected is active and ready."
    else
        print_error "Python $expected is missing."
        handle_failure "python$expected"
    fi
}

# Function to update pip
update_pip() {
    py_version=$1
    print_info "Updating pip for Python $py_version..."
    py -$py_version -m pip install --upgrade pip
}

# Function to check for required Python packages and install if missing
check_pip_package() {
    package=$1
    py_version=$2
    if py -$py_version -c "import $package" 2>/dev/null; then
        print_status "Python package '$package' is installed for Python $py_version."
    else
        print_warning "Python package '$package' is not installed for Python $py_version."
        print_info "Attempting to install '$package'..."
        py -$py_version -m pip install $package
        # Re-check if the package is installed after attempting installation
        if py -$py_version -c "import $package" 2>/dev/null; then
            # For pywin32, run the post-install script if necessary
            if [ "$package" == "pywin32" ]; then
                print_info "Running pywin32 post-install script..."
                py -$py_version -c "import pywin32_postinstall; pywin32_postinstall.install()"
            fi
            print_status "Successfully installed '$package' for Python $py_version."
        else
            print_error "Failed to install Python package '$package'."
            handle_failure "$package"
        fi
    fi
}

# Function to handle failures
handle_failure() {
    case "$1" in
        python3.10.9)
            echo ""
            print_info "Python 3.10.9 is required."
            echo "Please download and install Python 3.10.9 from:"
            echo "https://www.python.org/downloads/release/python-3109/"
            ;;
        colorama|pywin32)
            echo ""
            print_error "Automatic installation of '$1' failed."
            echo "Please install it manually by running:"
            echo "py -3.10 -m pip install $1"
            ;;
        *)
            print_error "Unknown failure type: $1"
            ;;
    esac
    exit 1
}

# Function to perform all checks
perform_checks() {
    # Check for Windows environment
    uname_out=$(uname -s)
    case "$uname_out" in
        CYGWIN*|MINGW*|MSYS*)
            print_status "Windows environment detected."
            ;;
        *)
            print_error "This script must be run in a Windows environment."
            exit 1
            ;;
    esac

    echo ""

    # Check Python installations
    print_info "Checking Python installations..."
    check_python_version 3.10 3.10.9
    check_python_version 3.12 3.12.4

    echo ""

    # Update pip for Python 3.10
    update_pip 3.10

    echo ""

    # Check for required Python packages
    print_info "Checking for required Python packages..."
    check_pip_package colorama 3.10
    check_pip_package pywin32 3.10

    echo ""

    # Check Git installation
    print_info "Checking Git installation..."
    if command_exists git; then
        print_status "Git is installed."
    else
        print_error "Git is not installed."
        handle_failure "git"
    fi

    echo ""

    # Check NVIDIA GPU and CUDA
    print_info "Checking NVIDIA GPU and CUDA Toolkit..."
    if command_exists nvidia-smi; then
        print_status "NVIDIA GPU detected."
        nvcc_version=$(nvcc --version 2>&1)
        if [[ "$nvcc_version" == *"release 12."* ]]; then
            print_status "CUDA Toolkit 12.x is installed."
        else
            print_error "CUDA Toolkit 12.x is not detected."
            handle_failure "cuda"
        fi
    else
        print_error "NVIDIA GPU not detected."
        handle_failure "gpu"
    fi

    echo ""

    # Check FFmpeg installation
    print_info "Checking FFmpeg installation..."
    if command_exists ffmpeg; then
        print_status "FFmpeg is installed."
    else
        print_error "FFmpeg is not installed."
        handle_failure "ffmpeg"
    fi

    echo ""

    # Check Ollama installation
    print_info "Checking Ollama installation..."
    if command_exists ollama; then
        print_status "Ollama is installed."
    else
        print_error "Ollama is not installed."
        handle_failure "ollama"
    fi

    echo ""

    # Check HuggingFace API key setup
    print_info "Checking HuggingFace API Key..."
    if [ -f ~/.huggingface/token ]; then
        print_status "HuggingFace API key is configured."
    else
        print_warning "HuggingFace API key not found."
    fi

    echo ""

    # Final message
    echo -e "${CYAN}"
    echo "============================================"
    echo "  Temporal Setup Verification Complete!  "
    echo "============================================"
    echo -e "${NC}"
    echo -e "${BLUE}All systems are ${GREEN}stable${BLUE}. You may proceed to access the Time Portal Setup.${NC}"
}

confirm_and_run_setup() {
    echo ""
    printf "Are you ready to initiate the setup now? (y/n): "
    read choice

    case "$choice" in
        y|Y )
            print_info "Launching the setup with Python 3.10..."
            # Directly execute the Python setup script
            py -3.10 SETUP.py

            # Check if the script was executed successfully
            if [ $? -eq 0 ]; then
                print_status "Setup script executed successfully."
                echo -e "${GREEN}Your Time Portal Access Tool is now active!${NC}"
            else
                print_error "Setup script encountered errors."
            fi
            ;;
        n|N )
            print_info "You can initiate the setup later by running 'py -3.10 SETUP.py'."
            ;;
        * )
            print_warning "Invalid input. Please enter 'y' or 'n'."
            confirm_and_run_setup
            ;;
    esac
}

# Main script execution
main() {
    print_banner
    echo ""
    print_info "Welcome to the Temporal Lab's setup."
    echo ""
    perform_checks
    echo ""
    confirm_and_run_setup
}

# Run the main function
main
