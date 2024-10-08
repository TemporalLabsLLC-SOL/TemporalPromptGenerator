# Temporal Prompt Engine: Local, Open-Soure, Intuitive, Cinematic Prompt Engine + Video and Audio Generation Suite for Nvidia GPUs

## Table of Contents
1. [Introduction](#1-introduction)
2. [Features Overview](#2-features-overview)
3. [Installation](#3-installation)
   - [Prerequisites](#prerequisites)
4. [Quick Start Guide](#4-quick-start-guide)
5. [API Key Setup](#5-api-key-setup)
6. [Story Mode: Unleash Epic Narratives](#6-story-mode-unleash-epic-narratives)
7. [Inspirational Use Cases](#7-inspirational-use-cases)
8. [Harnessing the Power of ComfyUI](#8-harnessing-the-power-of-comfyui)
9. [Local Video Generation Using CogVideo](#9-local-video-generation-using-cogvideo)
10. [Join the Temporal Labs Journey](#10-join-the-temporal-labs-journey)
11. [Donations and Support](#11-donations-and-support)
12. [Additional Services Offered](#12-additional-services-offered)
13. [Attribution and Courtesy Request](#13-attribution-and-courtesy-request)
14. [Contact](#14-contact)
15. [Acknowledgments](#15-acknowledgments)

---

<a name="1-introduction"></a>
## 1. Introduction

Welcome to the **Temporal Prompt Engine**, your ultimate tool for crafting immersive video and audio experiences. This engine empowers you to generate high-quality prompts with unparalleled control over cinematic elements, all while being intuitive and accessible for users.

### Unleash Your Creativity

Imagine capturing the world through the eyes of an ancient philosopher contemplating the cosmos, visualizing crypto-animals roaming digital landscapes, or peering into potential futures shaped by advanced technologies and societal shifts. With the **Temporal Prompt Engine**, these narratives are not just possible—they're straightforward and exhilarating to create.

This engine allows you to select standard film terms through user-friendly dropdown menus combined with a basic input concept. These selections are transformed into detailed lists of visual and audio prompts, ensuring that every aspect of your scene aligns perfectly with your creative vision. Additionally, the engine is capable of generating custom sound effects using the powerful **AudioLDM2**, giving you the tools to create truly unique and immersive audio landscapes.

---

<a name="2-features-overview"></a>
## 2. Features Overview

- **Cinematic Video Prompts**: Tailor every aspect of your scene—from camera type and lens to lighting and framing.
- **Adaptive Audio Prompts**: Generate immersive soundscapes that perfectly match your visuals.
- **Historical and Futuristic Perspectives**: Choose from ancient viewpoints to futuristic visions, adding depth and context to your scenes.
- **Dynamic Variables**: Adjust settings like resolution, frame rate, and aspect ratio to guide the AI in generating content that matches your vision.
- **Special Modes**:
  - **Story Mode**: Seamlessly blend prompts across frames to create cohesive narratives, enabling you to craft epic stories or intimate tales.
  - **Holiday Mode**: Generate seasonal content tailored to holidays, perfect for festive branding and marketing.
  - **Chaos Mode**: Add unpredictability with Chaos Mode.
- **Interconnected Settings**: Experience how choices like selecting an ancient art style influence other variables like color palette and texture, creating a cohesive and authentic output.
- **Cross-Platform Compatibility**: Available for **Windows**, with **Linux** support coming soon.

---

<a name="3-installation"></a>
## 3. Installation

Setting up the Temporal Prompt Engine is simple and hassle-free, allowing you to focus on creation rather than configuration. This section provides installation steps for **Windows** users.

<a name="prerequisites"></a>
### Prerequisites

Before installing the Temporal Prompt Engine, ensure that your system meets the following requirements:

- **Operating System**: Windows 10 or later
- **Python**: Version 3.10.9
    - **Download Python Installer**:
      - Visit the [Python Downloads](https://www.python.org/downloads/release/python-3109/) page.
      - Download the **Python 3.10.9** installer for Windows.
    - **Install Python**:
      - **Run the Installer**:
        - Double-click the downloaded `python-3.10.9-amd64.exe` file.
        - **Important**:
          - **If you already have Python installed**:
            - Be cautious when adding Python to PATH, as it may overwrite your existing Python version in the system PATH.
            - To avoid conflicts, you can install Python 3.10.9 without adding it to the PATH. Instead, use the Python Launcher for Windows (`py`) to specify the version when running commands.
          - **If you don't have Python installed**:
            - Check the box **"Add Python to PATH"** at the bottom of the installer window.
        - Click **"Install Now"**.
    - **Verify Installation**:
      - Open **Command Prompt**.
      - Run:
        ```bash
        py -3.10 --version
        ```
      - You should see `Python 3.10.9`.
      
- **Git**: Installed and configured
    - **Install Git**:
      - Download Git from [Git for Windows](https://git-scm.com/download/win).
      - Run the installer with default settings.
    - **Verify Installation**:
      - Open **Command Prompt**.
      - Run:
        ```bash
        git --version
        ```
      - You should see the installed Git version.

- **NVIDIA GPU**: CUDA-enabled NVIDIA GPU for optimal performance
  - Ensure you have an NVIDIA GPU with the latest drivers.
  
- **CUDA Toolkit**: Version 11.8 compatible with your GPU and installed
  - **Install CUDA Toolkit**:
      - Download the CUDA Toolkit from [CUDA Toolkit Download](https://developer.nvidia.com/cuda-11-8-0-download-archive).
      - Run the installer and follow the on-screen instructions.
  - **Verify Installation**:
      - Open Command Prompt and run:
        ```bash
        nvcc --version
        ```
      - You should see the CUDA compilation tools version information.

- **FFmpeg**: Installed and added to system `PATH`
    - Follow this guide to install: [How to Install FFmpeg on Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
    - **Verify Installation**:
      - Open Command Prompt and run:
        ```bash
        ffmpeg -version
        ```
      - You should see FFmpeg version information.
      
- **Ollama**: Download from [Ollama Setup](https://ollama.com/download/OllamaSetup.exe) and follow the on-screen instructions to install.

### Installation Steps for Temporal Prompt Engine

1. **Download the Repository**:
   - Visit the [TemporalPromptGenerator GitHub Repository](https://github.com/TemporalLabsLLC-SOL/TemporalPromptGenerator).
   - Click on the **"Code"** button and select **"Download ZIP"**.
   - Extract the downloaded ZIP file to your desired location (e.g., `C:\TemporalPromptEngine`).

2. **Run the Setup Script**:

   - Open Command Prompt and navigate to the extracted `TemporalPromptEngine` directory:
     ```bash
     cd C:\TemporalPromptEngine-main
     ```
   - Navigate to the `WINDOWS` folder:
     ```bash
     cd WINDOWS
     ```
   - Run the setup script:
     ```bash
     py -3.10 SETUP.py
     ```
   - Follow the on-screen prompts. The script will automatically set up the environment, install necessary packages, and configure settings.

   **IMPORTANT**: If the script prompts for a restart, **RESTART YOUR COMPUTER** to apply changes.

3. **Activate the Virtual Environment**:

   - The `SETUP.py` script creates a virtual environment named `TemporalPromptEngineEnv` within the project directory.
   - Activate the virtual environment:
     ```bash
     TemporalPromptEngineEnv\Scripts\activate
     ```
     You should see `(TemporalPromptEngineEnv)` prefixed in your command prompt.

4. **Launch the Temporal Prompt Engine**:

   With the virtual environment activated, run the main script:
   ```bash
   py -3.10 TemporalPromptEngine.py
   ```

   The application will launch, guiding you through the initial setup.

---

<a name="4-quick-start-guide"></a>
## 4. Quick Start Guide

1. **Activate the Virtual Environment**:
   ```bash
   cd C:\TemporalPromptEngine-main
   TemporalPromptEngineEnv\Scripts\activate
   ```

2. **Launch the Application**:
   ```bash
   py -3.10 TemporalPromptEngine.py
   ```

3. **Enter Your Scene Concept**:

   Input your creative idea or scene description (up to 450 characters).

   **Examples**:
   - **Ancient Perspectives**:
     > "View the world through the eyes of an ancient astronomer, mapping the stars with rudimentary tools under a vast, unpolluted night sky."
   - **Crypto-Animal Footage**:
     > "Documentary-style footage capturing the elusive 'cryptolion,' a mythical creature that embodies digital patterns and luminescent fur."

4. **Configure Video and Audio Prompt Options**:
   Tailor your video and audio prompts using dropdowns and input fields.

5. **Generate Video and Audio Prompts**:
   - Click **Generate Video Prompts

**.
   - Click **Generate Audio Prompts**.

6. **Save and Export**:
   All media and prompts are saved in your designated output directory.

---

<a name="5-api-key-setup"></a>
## 5. API Key Setup

1. **Sign Up at HuggingFace**:
   - Visit [HuggingFace Sign Up](https://huggingface.co/join) and create an account.

2. **Generate API Key**:
   - Navigate to **Access Tokens** in your HuggingFace settings.
   - Click **"New Token"**, name it, and set the scope to **"Read"**.
   - Copy the API key.

3. **Enter API Key in the Application**:
   - Paste your HuggingFace API key when prompted during the setup of the Temporal Prompt Engine.

---

<a name="6-story-mode-unleash-epic-narratives"></a>
## 6. Story Mode: Unleash Epic Narratives

**Story Mode** allows you to create cohesive stories by blending prompts across frames. Adjust variables like pacing, lighting, and camera movement to reflect the progression of your narrative.

---

<a name="7-inspirational-use-cases"></a>
## 7. Inspirational Use Cases

Explore the full potential of the Temporal Prompt Engine with these unique configurations:

1. **Ancient Philosophical Journey**:
   > "Through the eyes of an ancient philosopher, journey from the bustling streets of Athens to the serene temples of the Far East."

2. **Crypto-Animal Documentary**:
   > "Track the elusive 'cryptowolf' through digital forests where code and nature intertwine."

3. **Visions of the Future**:
   > "A day in the life of a future city powered by renewable energy and AI."

---

<a name="8-harnessing-the-power-of-comfyui"></a>
## 8. Harnessing the Power of ComfyUI

The Temporal Prompt Engine integrates with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for AI-based video generation. Import your prompts, generate videos, and iterate on your scenes with ComfyUI's powerful capabilities.

---

<a name="9-local-video-generation-using-cogvideo"></a>
## 9. Local Video Generation Using CogVideo

You can generate videos locally using **CogVideo**. The setup for CogVideo must be done **outside** of the Temporal Prompt Engine's virtual environment, as it is a separate application.

### Installation Steps

1. **Clone the CogVideo Repository**:
   ```bash
   git clone https://github.com/THUDM/CogVideo
   ```

2. **Copy Utility Script**:
   - Copy `PromptList2MP4Utility.py` into the `CogVideo/inference/gradio_composite_demo` folder.

3. **Create Virtual Environment**:
   ```bash
   cd CogVideo/inference/gradio_composite_demo
   python -m venv PL2MU
   PL2MU\Scripts\activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
   pip install moviepy==2.0.0.dev2
   ```

### Running the Utility

1. **Activate the CogVideo Environment**:
   ```bash
   cd CogVideo/inference/gradio_composite_demo
   PL2MU\Scripts\activate
   ```

2. **Open the Utility Interface**:
   ```bash
   python PromptList2MP4Utility.py
   ```

This will open the interface to generate CogVideoX videos based on the prompt list from the Temporal Prompt Engine.

---

<a name="10-join-the-temporal-labs-journey"></a>
## 10. Join the Temporal Labs Journey

Support the mission of pushing the boundaries of AI and technology. Join as an investor, developer, or client.

---

<a name="11-donations-and-support"></a>
## 11. Donations and Support

If you'd like to support the project, you can donate via the following crypto addresses:

- **Ethereum (ETH)**: `0x5616b3415ED6Ea7005595eF144A2054d4cD5767B`
- **Bitcoin (BTC)**: `bc1qpsfn8a7cs75fxxwv3ax7gtnurm44n5x2fmh59c`
- **Solana (SOL)**: `FVPGxfGT7QWfQEWvXpFkwdgiiKFM3VdvzNG6mEmX8pgi`
- **Litecoin (LTC)**: `ltc1qwlyjz8aymy9uagqhht5a4kaq06kmv58dxlzyww`
- **Dogecoin (DOGE)**: `DAeWAroHCy8nXCoUsobderPRSNXNu1WY34`
- **Venmo**: `@Utah-DM`

---

<a name="12-additional-services-offered"></a>
## 12. Additional Services Offered

- **Tutoring**, **Development**, **Design**, **Consulting**, and **Workshops** are available to meet your AI and technology needs.

---

<a name="13-attribution-and-courtesy-request"></a>
## 13. Attribution and Courtesy Request

If you create videos or content using prompts generated by the Temporal Prompt Engine, we kindly request that you include the phrase **"using Temporal Prompt Engine"** in your final release.

---

<a name="14-contact"></a>
## 14. Contact

For questions, support, collaboration opportunities, or to discuss how we can work together:

- **Email**: [Sol@TemporalLab.com](mailto:Sol@TemporalLab.com)
- **Phone**: +1-385-222-9920

---

<a name="15-acknowledgments"></a>
## 15. Acknowledgments

Thanks to the developers and communities behind **Git**, **Python**, **FFmpeg**, **Ollama**, **AudioLDM2**, **CogVideo**, **ComfyUI**, and **HuggingFace** for making this project possible.

---

