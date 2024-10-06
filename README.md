# Temporal Prompt Engine: Intuitive Scene Builder for Video and Audio Generation

---

### Table of Contents
1. [Introduction](#1-introduction)
2. [Features Overview](#2-features-overview)
3. [Installation](#3-installation)
   - [Prerequisites](#prerequisites)
   - [Automated Installation](#automated-installation)
   - [Manual Installation](#manual-installation)
     - [Windows](#windows)
     - [Linux](#linux)
4. [Quick Start Guide](#4-quick-start-guide)
5. [API Key Setup](#5-api-key-setup)
6. [Harnessing the Power of ComfyUI](#6-harnessing-the-power-of-comfyui)
7. [Join the Temporal Labs Journey](#7-join-the-temporal-labs-journey)
8. [Donations and Support](#8-donations-and-support)
9. [Contact](#9-contact)
10. [Acknowledgments](#10-acknowledgments)

---

<a name="1-introduction"></a>

## 1. Introduction

Welcome to the **Temporal Prompt Engine**, your ultimate tool for crafting immersive video and audio experiences. This engine empowers you to generate high-quality prompts with unparalleled control over cinematic elements, all while being intuitive and accessible for both Windows and Linux users.

### **Unleash Your Creativity**

Imagine directing a scene set in a cyberpunk future, with neon-lit streets captured through the lens of a 1980s camera, and accompanied by a rich, layered soundscape that adapts dynamically to your visuals. With the **Temporal Prompt Manager**, this isn't just possible—it's straightforward and exhilarating.

The **Temporal Prompt Manager** allows you to select standard film terms through user-friendly dropdown menus combined with a basic input concept. These selections are transformed into detailed lists of visual and audio prompts, ensuring that every aspect of your scene aligns perfectly with your creative vision. Additionally, the engine is capable of generating custom sound effects using the powerful **AudioLDM2**, giving you the tools to create truly unique and immersive audio landscapes.

### **Why Temporal Prompt Engine?**

- **Intuitive Interface**: Designed for both beginners and professionals, making complex scene creation accessible to everyone.
- **Advanced Customization**: Select from a wide range of film terms to fine-tune your prompts.
- **Seamless Integration**: Works flawlessly with tools like ComfyUI for enhanced video generation.
- **Cross-Platform Compatibility**: Available for both **Windows** and **Linux** systems, ensuring broad accessibility.
- **Community-Driven**: Join a vibrant community of innovators, developers, and artists pushing the boundaries of AI-driven media creation.

Whether you're a filmmaker, game developer, digital artist, or an AI enthusiast, the Temporal Prompt Engine is designed to unleash your creativity and bring your visions to life with precision and style.

---

<a name="2-features-overview"></a>

## 2. Features Overview

### **Unleash Your Creativity with Advanced Controls**

- **Cinematic Video Prompts**: Tailor every aspect of your scene—from camera type and lens to lighting and framing.
- **Adaptive Audio Prompts**: Generate immersive soundscapes that perfectly match your visuals.
- **Historical and Futuristic Cameras**: Choose from an extensive list of camera models across decades, adding authenticity or a futuristic touch to your scenes.
- **Dynamic Variables**: Adjust settings like resolution, frame rate, and aspect ratio to guide the AI in generating content that matches your vision.
- **Special Modes**: Incorporate thematic elements with Holiday Mode or add creative unpredictability with Chaos Mode.
- **Interconnected Settings**: Experience how choices like selecting a vintage camera influence other variables like resolution and aspect ratio, creating a cohesive and authentic output.

### **Intuitive and User-Friendly Interface**

- Designed for simplicity and ease of use, allowing both beginners and professionals to navigate effortlessly.
- Compatible with both **Windows** and **Linux** systems.

---

<a name="3-installation"></a>

## 3. Installation

Setting up the Temporal Prompt Engine is simple and hassle-free, allowing you to focus on creation rather than configuration. This section provides both automated and manual installation steps for **Windows** and **Linux** users.

### <a name="prerequisites"></a>Prerequisites

Before installing the Temporal Prompt Engine, ensure that your system meets the following requirements:

- **Operating System**: Windows 10 or later / Linux (Ubuntu, Debian, Fedora, CentOS, RHEL)
- **Pre-install Essential Tools**:

   - **Install 7-Zip (if needed)**:
     - Download 7-Zip from [7-Zip Download](https://www.7-zip.org/download.html).
     - Run the installer and follow the on-screen instructions.
   - **Install WinRAR (optional)**:
     - Download WinRAR from [WinRAR Download](https://www.win-rar.com/download.html).
     - Run the installer and follow the on-screen instructions.
- **Python**: Version 3.10.9
- **Download Python Installer**:
   - Visit the [Python Downloads](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) page.
   - Download the **Python 3.10.9** installer for Windows.
   - **Install Python**:
   - **Run the Installer**:
     - Double-click the downloaded `python-3.10.9-amd64.exe` file.
     - **Important**: **Check the box "Add Python to PATH"** at the bottom of the installer window.
     - Click **"Install Now"**.
   - **Verify Installation**:
     - Open **Command Prompt**.
     - Run:
       ```bash
       python --version
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
- **CUDA Toolkit**: Version 11.8 Compatible with your GPU and installed
- **Install CUDA Toolkit**:
     - Ensure you have an NVIDIA GPU with the latest drivers.
     - Download the CUDA Toolkit from [CUDA Toolkit Download](https://developer.nvidia.com/cuda-11-8-0-download-archive).
     - Run the installer and follow the on-screen instructions.
- **FFmpeg**: Installed and added to system `PATH`
- **Install FFMPEG**:
     - `https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/`
- **Install Ollama**:
     - Download Ollama from [Ollama Setup](https://ollama.com/download/OllamaSetup.exe).
     - Run the installer and follow the on-screen instructions.
     
 
1. **Download the Repository**:
   - Visit the [TemporalPromptGenerator GitHub Repository](https://github.com/TemporalLabsLLC-SOL/TemporalPromptGenerator).
   - Click on the **"Code"** button and select **"Download ZIP"**.
   - Extract the downloaded ZIP file to your desired location, e.g., `C:\TemporalPromptEngine`.
 

2. **Run the Setup Script**:
   - **Open Command Prompt**:
     - Navigate to the extracted `TemporalPromptEngine` directory:
       ```bash
       cd C:\TemporalPromptEngine-main
       ```
       (OR Whatever you extracted it as)

   - Navigate to the respective `WINDOWS or LINUX` folder depending on your system. ONLY WINDOWS WORKS RELIABLY RIGHT NOW.
     - ```bash
       cd WINDOWS
       ```

       OR

       ```bash
       cd LINUX
       ```
   - **Run the Setup Script**:
     ```bash
     python SETUP.py
     ```
   - **Follow On-Screen Prompts**:
     - The script will automatically set up the environment, install necessary packages, and configure settings.
     - **IMPORTANT**: If the script prompts for a restart, **RESTART YOUR COMPUTER** to apply changes.

**Activate the Virtual Environment**:
   - The `SETUP.py` script creates a virtual environment named `TemporalPromptEngineEnv` within the project directory.
   - **Activate the Virtual Environment**:
     - In **Terminal**, navigate to the project directory if not already there:
       ```bash
       cd ~/TemporalPromptGenerator
       ```
     - Activate the virtual environment:
       ```bash
       source TemporalPromptEngineEnv/bin/activate
       ```
     - **Confirmation**: You should see `(TemporalPromptEngineEnv)` prefixed in your terminal, indicating that the virtual environment is active.

**Launch the Temporal Prompt Engine**:
   - With the virtual environment activated, run the main script:
     ```bash
     python TemporalPromptEngine.py
     ```
   - The application will launch, guiding you through the initial setup.


<a name="4-quick-start-guide"></a>

## 4. Quick Start Guide

Follow these simple steps to begin creating your immersive video and audio scenes:

1. **Activate the Virtual Environment**:
   - **Windows**:
     - Open **Command Prompt**.
     - Navigate to the `TemporalPromptEngine` directory:
       ```bash
       cd C:\TemporalPromptEngine
       ```
     - Activate the virtual environment:
       ```bash
       TemporalPromptEngineEnv\Scripts\activate
       ```
     - **Confirmation**: You should see `(TemporalPromptEngineEnv)` prefixed in your command prompt.
   - **Linux**:
     - Open **Terminal**.
     - Navigate to the `TemporalPromptEngine` directory:
       ```bash
       cd ~/TemporalPromptGenerator
       ```
     - Activate the virtual environment:
       ```bash
       source TemporalPromptEngineEnv/bin/activate
       ```
     - **Confirmation**: You should see `(TemporalPromptEngineEnv)` prefixed in your terminal.

2. **Launch the Application**:
   - With the virtual environment activated, run the main script.
   - **Windows**:
     ```bash
     python TemporalPromptEngine.py
     ```
   - **Linux**:
     ```bash
     python3 TemporalPromptEngine.py
     ```
   - The application will launch, guiding you through the initial setup.

3. **Enter Your Scene Concept**:
   - Input your creative idea or scene description (up to 450 characters).
   - **Example**:
     > "A futuristic cityscape bustling with flying cars, towering skyscrapers adorned with holographic advertisements, under the glow of a setting sun."

4. **Configure Video Prompt Options**:
   - Select standard film terms through dropdown menus.
   - **Example Settings**:
     - **Theme**: Sci-Fi
     - **Art Style**: Cyberpunk
     - **Lighting**: Neon Lighting
     - **Framing**: Wide Shot
     - **Camera Movement**: Tracking Shot
     - **Time of Day**: Dusk
     - **Decade**: 1980s
     - **Camera**: Panavision Panaflex
     - **Lens**: Anamorphic
     - **Resolution**: 4K UHD
   - **Note**: Selecting the **1980s** decade filters the available cameras to those from that era, and choosing the **Anamorphic** lens influences the aspect ratio and cinematic feel of your scene.

5. **Generate Video Prompts**:
   - Click **Generate Video Prompts**.
   - The engine transforms your inputs into detailed video prompts ready for generation.

6. **Configure Audio Prompt Options**:
   - Tailor your soundscape using dropdowns and input fields.
   - **Example Settings**:
     - **Exclude Music**: Enabled (focus on ambient sounds)
     - **Holiday Mode**: Disabled
     - **Specific Modes**: SoundScape Mode
     - **Layer Intensity**: High

7. **Generate Audio Prompts**:
   - Click **Generate Audio Prompts** to create adaptive audio that complements your visuals.

8. **Generate Sound Effects**:
   - Specify the duration of each sound effect.
   - Click **Generate Sound Effects** to produce layered audio, enriching your scenes.

9. **Harnessing the Power of ComfyUI**:
   - Use the included [ComfyUI Workflow](https://github.com/TemporalLabsLLC-SOL/TemporalPromptGenerator) to generate your videos.
   - The engine seamlessly integrates with ComfyUI, allowing you to take the generated video prompts and produce stunning visuals with ease.

10. **Combine Video and Audio**:
    - Use the **COMBINE** feature to merge your video and audio into a seamless, immersive experience.

11. **Save and Export**:
    - All media and prompts are saved in your designated output directory for easy access and future use.

---

<a name="5-api-key-setup"></a>

## 5. API Key Setup

Unlock the full potential of the Temporal Prompt Engine by setting up your API key.

### **HuggingFace API Key**

- **Purpose**: Enables advanced AI-driven prompt generation for both video and audio.
- **Setup**:
  1. **Sign Up at HuggingFace**:
     - Visit [HuggingFace Sign Up](https://huggingface.co/join) and create an account if you don't have one.
  2. **Generate API Key**:
     - After logging in, navigate to your account settings by clicking on your profile picture and selecting **"Settings"**.
     - Click on **"Access Tokens"** in the sidebar.
     - Click **"New Token"**, name it (e.g., "TemporalPromptEngine"), and set the scope to **"Read"**.
     - Click **"Generate"** and **copy your API key**.
  3. **Enter API Key in the Application**:
     - Run the Temporal Prompt Engine.
     - When prompted, paste your HuggingFace API key.
     - The application will store it securely for future use.

**Note**: Your API keys are stored securely, and you only need to enter them once.

---

<a name="6-harnessing-the-power-of-comfyui"></a>

## 6. Harnessing the Power of ComfyUI

The Temporal Prompt Engine is designed to work seamlessly with [ComfyUI](https://comfyui.com/), a powerful tool for AI-based video generation.

- **Integration**: After generating your video prompts, use the included [ComfyUI Workflow](https://github.com/TemporalLabsLLC-SOL/TemporalPromptGenerator) to produce your videos.
- **Process**:
  - **Import Prompts**: The engine exports prompts in a format compatible with ComfyUI.
  - **Generate Videos**: Utilize ComfyUI's advanced capabilities to bring your scenes to life.
  - **Refinement**: Iterate on your prompts and settings based on the outputs, refining your vision.

By combining the strengths of the Temporal Prompt Engine and ComfyUI, you unlock a workflow that is both powerful and user-friendly, streamlining the path from concept to creation.

---

<a name="7-join-the-temporal-labs-journey"></a>

## 7. Join the Temporal Labs Journey

I am a lone futurist on a mission, dedicated to pushing the boundaries of what's possible in AI and technology. Temporal Labs LLC is not just about tools—it's about fostering a community of innovators, developers, artists, and visionaries.

### **Seeking Collaborators and Support**

- **Investors**: Partner with me to accelerate innovation and bring groundbreaking ideas to life.
- **Developers and Students**: Join the journey, contribute to cutting-edge projects, and grow alongside like-minded individuals.
- **Clients and Associates**: Let's work together to create custom solutions that meet your unique needs.
- **Supporters**: Your encouragement and support fuel the mission.

### **Beyond the Temporal Prompt Engine**

Temporal Labs LLC offers a range of AI and tech services across software and hardware. Whether it's developing custom AI models, consulting on technological strategies, or crafting innovative hardware solutions, I'm committed to delivering excellence.

---

<a name="8-donations-and-support"></a>

## 8. Donations and Support

Your support helps continue the mission of innovation and the development of tools like the Temporal Prompt Engine. If you find this tool valuable, consider making a donation.

### **Crypto Donations**

- **Ethereum (ETH)**:
  - Address: `0x5616b3415ED6Ea7005595eF144A2054d4cD5767B`

- **Bitcoin (BTC)**:
  - Address: `bc1qpsfn8a7cs75fxxwv3ax7gtnurm44n5x2fmh59c`

- **Solana (SOL)**:
  - Address: `FVPGxfGT7QWfQEWvXpFkwdgiiKFM3VdvzNG6mEmX8pgi`

- **Litecoin (LTC)**:
  - Address: `ltc1qwlyjz8aymy9uagqhht5a4kaq06kmv58dxlzyww`

- **Dogecoin (DOGE)**:
  - Address: `DAeWAroHCy8nXCoUsobderPRSNXNu1WY34`

*You can copy any of these addresses to your clipboard for easy transfer.*

### **Venmo**

- **Venmo ID**: `@Utah-DM`

---

<a name="9-contact"></a>

## 9. Contact

For questions, support, or to discuss collaboration opportunities, I'd love to hear from you:

- **Email**: [Sol@TemporalLab.com](mailto:Sol@TemporalLab.com)
- **Phone**: +1-385-222-9920

---

<a name="10-acknowledgments"></a>

## 10. Acknowledgments

The development of the Temporal Prompt Engine leverages a variety of open-source tools and libraries:

- **[Git](https://git-scm.com/)**: Version control system.
- **[Python](https://www.python.org/)**: Programming language.
- **[FFmpeg](https://ffmpeg.org/)**: Multimedia framework.
- **[Ollama](https://ollama.com/)**: AI-powered tools.
- **[AudioLDM2](https://github.com/haoheliu/AudioLDM2)**: Audio generation model.
- **[ComfyUI](https://comfyui.com/)**: User interface for AI-based video generation.
- **[HuggingFace](https://huggingface.co/)**: AI models and APIs.

A special thanks to the developers and communities behind these tools for making innovative projects like the Temporal Prompt Engine possible.

---

**Embark on the Temporal Labs Journey**

This is more than just a tool—it's a gateway to infinite creative possibilities. The Temporal Prompt Engine is crafted to inspire and empower, offering you a canvas limited only by your imagination. With its extensive features and intuitive design, creating visually stunning and sonically rich scenes has never been more accessible.

Join me in exploring the future of AI-driven media creation. Your support fuels innovation, and I'm excited to see what you'll create.

---

Visit us at [www.TemporalLab.com](http://www.TemporalLab.com)

---

*Temporal Labs LLC is committed to ethical AI development, focusing on open-source solutions and community-driven innovation. By choosing the Temporal Prompt Engine, you're not just using a tool—you're joining a movement towards responsible and groundbreaking technological advancement.*

---

# Additional Notes

- **Repository Link**: [TemporalPromptGenerator GitHub Repository](https://github.com/TemporalLabsLLC-SOL/TemporalPromptGenerator)
- **Issue Reporting**: For any issues or feature requests, please [create an issue](https://github.com/TemporalLabsLLC-SOL/TemporalPromptGenerator/issues) on the GitHub repository.
- **Security**: Ensure that your API keys are kept secure and never shared publicly.

---

This comprehensive README provides clear, step-by-step instructions tailored for even the most amateur users, ensuring a smooth installation and setup process for the Temporal Prompt Engine. It acknowledges all necessary dependencies and tools used in the `SETUP.py` script, fostering transparency and trust. With inspirational content and practical guidance, users are encouraged to harness the full potential of the engine to create stunning video and audio scenes.
