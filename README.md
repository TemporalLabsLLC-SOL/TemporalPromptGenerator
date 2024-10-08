# Temporal Prompt Engine: Intuitive Scene Builder for Video and Audio Generation

## Table of Contents

- [Introduction](#1-introduction)
- [Features Overview](#2-features-overview)
- [Installation](#3-installation)
  - [Prerequisites](#prerequisites)
- [Quick Start Guide](#4-quick-start-guide)
- [API Key Setup](#5-api-key-setup)
- [Story Mode: Unleash Epic Narratives](#6-story-mode-unleash-epic-narratives)
- [Inspirational Use Cases](#7-inspirational-use-cases)
- [Harnessing the Power of ComfyUI](#8-harnessing-the-power-of-comfyui)
- [Join the Temporal Labs Journey](#9-join-the-temporal-labs-journey)
- [Donations and Support](#10-donations-and-support)
- [Additional Services Offered](#11-additional-services-offered)
- [Attribution and Courtesy Request](#12-attribution-and-courtesy-request)
- [Contact](#13-contact)
- [Acknowledgments](#14-acknowledgments)

---

<a name="1-introduction"></a>
## 1. Introduction

Welcome to the Temporal Prompt Engine, your ultimate tool for crafting immersive video and audio experiences. This engine empowers you to generate high-quality prompts with unparalleled control over cinematic elements, all while being intuitive and accessible for users.

### Unleash Your Creativity

Imagine capturing the world through the eyes of an ancient philosopher contemplating the cosmos, visualizing crypto-animals roaming digital landscapes, or peering into potential futures shaped by advanced technologies and societal shifts. Envision dreams materializing into surreal scenes, or exploring the depths of the human psyche through abstract visuals.

With the Temporal Prompt Engine, these narratives are not just possible—they're straightforward and exhilarating to create.

The Temporal Prompt Engine allows you to select standard film terms through user-friendly dropdown menus combined with a basic input concept. These selections are transformed into detailed lists of visual and audio prompts, ensuring that every aspect of your scene aligns perfectly with your creative vision. Additionally, the engine is capable of generating custom sound effects using the powerful AudioLDM2, giving you the tools to create truly unique and immersive audio landscapes.

### Why Temporal Prompt Engine?

- **Intuitive Interface**: Designed for both beginners and professionals, making complex scene creation accessible to everyone.
- **Advanced Customization**: Select from a wide range of film terms to fine-tune your prompts.
- **Story Mode - BIG UPDATE IMPLEMENTED**: Seamlessly blend prompts across frames to create cohesive narratives, enabling you to craft epic stories or intimate tales.
- **Holiday Mode**: Generate seasonal content tailored to holidays, perfect for festive branding and marketing.
- **Seamless Integration**: Works flawlessly with tools like ComfyUI for enhanced video generation.
- **Cross-Platform Compatibility**: Available for Windows, with Linux support coming soon.
- **Community-Driven**: Join a vibrant community of innovators, developers, artists, and visionaries pushing the boundaries of AI-driven media creation.

Whether you're a filmmaker, digital artist, educator, marketer, or an AI enthusiast, the Temporal Prompt Engine is designed to unleash your creativity and bring your visions to life with precision and style.

---

<a name="2-features-overview"></a>
## 2. Features Overview

### Unleash Your Creativity with Advanced Controls

- **Cinematic Video Prompts**: Tailor every aspect of your scene—from camera type and lens to lighting and framing.
- **Adaptive Audio Prompts**: Generate immersive soundscapes that perfectly match your visuals.
- **Historical and Futuristic Perspectives**: Choose from ancient viewpoints to futuristic visions, adding depth and context to your scenes.
- **Dynamic Variables**: Adjust settings like resolution, frame rate, and aspect ratio to guide the AI in generating content that matches your vision.

#### Special Modes:

- **Story Mode**: Seamlessly blend prompts across frames to create cohesive narratives, enabling you to craft stories ranging from epic adventures to intimate tales.
- **Holiday Mode**: Incorporate holiday themes into your content, perfect for seasonal branding and marketing.
- **Chaos Mode**: Add creative unpredictability with Chaos Mode.
- **Interconnected Settings**: Experience how choices like selecting an ancient art style influence other variables like color palette and texture, creating a cohesive and authentic output.

### Intuitive and User-Friendly Interface

- Designed for simplicity and ease of use, allowing both beginners and professionals to navigate effortlessly.
- Compatible with Windows, with Linux support on the horizon.

---

<a name="3-installation"></a>
## 3. Installation

Setting up the Temporal Prompt Engine is simple and hassle-free, allowing you to focus on creation rather than configuration. This section provides installation steps for Windows users.

<a name="prerequisites"></a>
### Prerequisites

Before installing the Temporal Prompt Engine, ensure that your system meets the following requirements:

- **Operating System**: Windows 10 or later

#### Pre-install Essential Tools:

##### Python: Version 3.10.9

1. **Download Python Installer**:
   - Visit the [Python Downloads](https://www.python.org/downloads/release/python-3109/) page.
   - Download the Python 3.10.9 installer for Windows.

2. **Install Python**:
   - **Run the Installer**:
     - Double-click the downloaded `python-3.10.9-amd64.exe` file.

   - **Important**:
     - **If you already have Python installed**:
       - Be cautious when adding Python to PATH, as it may overwrite your existing Python version in the system PATH.
       - To avoid conflicts, you can install Python 3.10.9 without adding it to the PATH. Instead, use the Python Launcher for Windows (`py`) to specify the version when running commands.
     - **If you don't have Python installed**:
       - Check the box **"Add Python to PATH"** at the bottom of the installer window.
       - Click **"Install Now"**.

3. **Verify Installation**:
   - Open Command Prompt.
   - Run:

     ```bash
     py -3.10 --version
     ```

   - You should see `Python 3.10.9`.

##### Git: Installed and configured

1. **Install Git**:
   - Download Git from [Git for Windows](https://gitforwindows.org/).
   - Run the installer with default settings.

2. **Verify Installation**:
   - Open Command Prompt.
   - Run:

     ```bash
     git --version
     ```

   - You should see the installed Git version.

##### NVIDIA GPU: CUDA-enabled NVIDIA GPU for optimal performance

##### CUDA Toolkit: Version 11.8 compatible with your GPU and installed

1. **Install CUDA Toolkit**:
   - Ensure you have an NVIDIA GPU with the latest drivers.
   - Download the CUDA Toolkit from [CUDA Toolkit Download](https://developer.nvidia.com/cuda-11-8-0-download-archive).
   - Run the installer and follow the on-screen instructions.

2. **Verify Installation**:
   - Open Command Prompt.
   - Run:

     ```bash
     nvcc --version
     ```

   - You should see the CUDA compilation tools version information.

##### FFmpeg: Installed and added to system PATH

1. **Install FFmpeg**:
   - Follow this guide: [How to Install FFmpeg on Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)

2. **Verify Installation**:
   - Open Command Prompt.
   - Run:

     ```bash
     ffmpeg -version
     ```

   - You should see FFmpeg version information.

##### Install Ollama

1. **Download Ollama**:
   - Download Ollama from [Ollama Setup](https://ollama.ai/).

2. **Install Ollama**:
   - Run the installer and follow the on-screen instructions.

#### Linux Support Coming Soon

We're actively working on developing a dedicated Linux build. This requires some basic Python skills to use in its current state, and we're committed to investing the time over the next few months to make it accessible to all. Your support and contributions can help accelerate this process.

### Installation Steps for Windows

1. **Download the Repository**:

   - Visit the [TemporalPromptGenerator GitHub Repository](https://github.com/TemporalLabs/TemporalPromptGenerator).
   - Click on the **"Code"** button and select **"Download ZIP"**.
   - Extract the downloaded ZIP file to your desired location, e.g., `C:\TemporalPromptEngine`.

2. **Run the Setup Script**:

   - **Open Command Prompt**:
     - Navigate to the extracted `TemporalPromptEngine` directory:

       ```bash
       cd C:\TemporalPromptEngine-main
       ```

       (Replace with the actual directory name if different.)

     - Navigate to the `WINDOWS` folder:

       ```bash
       cd WINDOWS
       ```

   - **Run the Setup Script**:

     ```bash
     py -3.10 SETUP.py
     ```

   - **Follow On-Screen Prompts**:
     - The script will automatically set up the environment, install necessary packages, and configure settings.
     - **IMPORTANT**: If the script prompts for a restart, **RESTART YOUR COMPUTER** to apply changes.

3. **Activate the Virtual Environment**:

   - The `SETUP.py` script creates a virtual environment named `TemporalPromptEngineEnv` within the project directory.

   - **Activate the Virtual Environment**:
     - Navigate to the project directory if not already there:

       ```bash
       cd C:\TemporalPromptEngine-main
       ```

     - Activate the virtual environment:

       ```bash
       TemporalPromptEngineEnv\Scripts\activate
       ```

     - **Confirmation**: You should see `(TemporalPromptEngineEnv)` prefixed in your command prompt, indicating that the virtual environment is active.

4. **Launch the Temporal Prompt Engine**:

   - With the virtual environment activated, run the main script:

     ```bash
     py -3.10 TemporalPromptEngine.py
     ```

   - The application will launch, guiding you through the initial setup.

---

<a name="4-quick-start-guide"></a>
## 4. Quick Start Guide

Follow these simple steps to begin creating your immersive video and audio scenes:

### Activate the Virtual Environment:

**Windows**:

1. Open Command Prompt.
2. Navigate to the `TemporalPromptEngine` directory:

   ```bash
   cd C:\TemporalPromptEngine-main
   ```

3. Activate the virtual environment:

   ```bash
   TemporalPromptEngineEnv\Scripts\activate
   ```

4. **Confirmation**: You should see `(TemporalPromptEngineEnv)` prefixed in your command prompt.

### Launch the Application:

With the virtual environment activated, run the main script:

```bash
py -3.10 TemporalPromptEngine.py
```

The application will launch, guiding you through the initial setup.

### Enter Your Scene Concept:

Input your creative idea or scene description (up to 450 characters).

**Examples**:

- **Ancient Perspectives**:

  > "View the world through the eyes of an ancient astronomer, mapping the stars with rudimentary tools under a vast, unpolluted night sky."

- **Crypto-Animal Footage**:

  > "Documentary-style footage capturing the elusive 'cryptolion,' a mythical creature that embodies digital patterns and luminescent fur."

- **Glimpses at Potential Futures**:

  > "A bustling metropolis powered by sustainable energy, where drones fill the sky and vertical gardens adorn skyscrapers."

- **Dream Envisioning**:

  > "A surreal landscape where gravity is inverted, and floating islands house castles made of clouds."

- **Exploration of an Atom**:

  > "A microscopic voyage into the heart of an atom, revealing swirling electrons and the mysterious nucleus within."

### Configure Video Prompt Options:

- **Enable Special Modes (optional)**:
  - **Story Mode**: Toggle to **ON** if you wish to create a narrative that blends across frames.
  - **Holiday Mode**: Toggle to **ON** to incorporate holiday themes into your content.
- **Set the Number of Frames** for your story if using Story Mode.
- **Select standard film terms** through dropdown menus.

**Example Settings**:

- **Theme**: Science Fiction, Fantasy, Documentary, etc.
- **Art Style**: Surrealism, Futurism, Ancient Art, Digital Art
- **Lighting**: Bioluminescent, Cosmic Light, Soft Dreamy Glow
- **Framing**: Close-Up, Wide Shot, Aerial View
- **Camera Movement**: Steadicam, Drone Shot, Slow Motion
- **Time of Day**: Twilight, Midnight, Dawn
- **Resolution**: 4K UHD

**Note**: Enabling Story Mode allows the engine to blend your prompts across the frames, creating a cohesive narrative experience.

### Generate Video Prompts:

Click **Generate Video Prompts**.

The engine transforms your inputs into detailed video prompts ready for generation.

### Configure Audio Prompt Options:

Tailor your soundscape using dropdowns and input fields.

**Example Settings**:

- **Exclude Music**: Enabled or Disabled
- **Specific Modes**: SoundScape Mode, Ambient Sounds
- **Layer Intensity**: High

### Generate Audio Prompts:

Click **Generate Audio Prompts** to create adaptive audio that complements your visuals.

### Generate Sound Effects:

- Specify the **duration** of each sound effect.
- Click **Generate Sound Effects** to produce layered audio, enriching your scenes.

### Combine Video and Audio:

Use the **COMBINE** feature to merge your video and audio into a seamless, immersive experience.

### Save and Export:

All media and prompts are saved in your designated output directory for easy access and future use.

---

<a name="5-api-key-setup"></a>
## 5. API Key Setup

Unlock the full potential of the Temporal Prompt Engine by setting up your API key.

### HuggingFace API Key

**Purpose**: Enables advanced AI-driven prompt generation for both video and audio.

**Setup**:

1. **Sign Up at HuggingFace**:
   - Visit [HuggingFace Sign Up](https://huggingface.co/join) and create an account if you don't have one.

2. **Generate API Key**:
   - After logging in, navigate to your account settings by clicking on your profile picture and selecting **"Settings"**.
   - Click on **"Access Tokens"** in the sidebar.
   - Click **"New Token"**, name it (e.g., "TemporalPromptEngine"), and set the scope to **"Read"**.
   - Click **"Generate"** and copy your API key.

3. **Enter API Key in the Application**:
   - Run the Temporal Prompt Engine.
   - If prompted, paste your HuggingFace API key.
     - **Note**: If you have previously provided an API key, you can close the popup. Otherwise, follow the instructions above to obtain one.
   - The application will store it securely for future use.

**Note**: Your API keys are stored securely, and you only need to enter them once.

---

<a name="6-story-mode-unleash-epic-narratives"></a>
## 6. Story Mode: Unleash Epic Narratives

Story Mode is a powerful feature of the Temporal Prompt Engine that allows you to create cohesive and compelling stories by blending prompts across a sequence of frames. Whether you're crafting a philosophical journey through ancient wisdom, visualizing the evolution of crypto-animals, or projecting potential futures shaped by technology, Story Mode enables you to bring your visions to life in a seamless and engaging way.

### How Story Mode Works

- **Blended Prompts**: Story Mode intelligently transitions your scene descriptions and settings across frames, ensuring smooth visual and thematic continuity.
- **Narrative Control**: Adjust the number of frames, pacing, and key narrative points to shape the flow of your story.
- **Dynamic Variables**: Utilize the full range of cinematic controls to enhance storytelling, such as changing lighting, camera movement, and art styles to reflect the progression of your narrative.

---

<a name="7-inspirational-use-cases"></a>
## 7. Inspirational Use Cases

Explore the full potential of the Temporal Prompt Engine with these unique configurations that showcase its power:

### Ancient Philosophical Journey

- **Scenario**: An exploration of ancient civilizations and their understanding of the universe.
- **Settings**:
  - **Concept**: "Through the eyes of an ancient philosopher, journey from the bustling streets of Athens to the serene temples of the Far East."
  - **Theme**: Historical, Philosophical
  - **Art Style**: Classical Art
  - **Camera Movement**: Slow Pan, Steadicam
- **Outcome**: Create a thoughtful and immersive narrative that delves into ancient wisdom and perspectives.

### Crypto-Animal Documentary

- **Scenario**: A wildlife documentary capturing mythical creatures that represent the fusion of nature and digital technology.
- **Settings**:
  - **Concept**: "Track the elusive 'cryptowolf' through digital forests where code and nature intertwine."
  - **Theme**: Sci-Fi, Fantasy
  - **Art Style**: Digital Surrealism
  - **Lighting**: Bioluminescent Glow
- **Outcome**: Generate captivating visuals that blend natural elements with digital aesthetics.

### Visions of the Future

- **Scenario**: A speculative look at future societies and technologies.
- **Settings**:
  - **Concept**: "A day in the life of a future city powered by renewable energy and AI, where humans and robots coexist."
  - **Theme**: Futuristic, Utopian/Dystopian
  - **Art Style**: Futurism
  - **Camera Movement**: Drone Shots, Time-Lapse
- **Outcome**: Create thought-provoking content that inspires discussions about potential futures.

### Dreamscape Exploration

- **Scenario**: Visualizing the abstract and surreal nature of dreams.
- **Settings**:
  - **Concept**: "Navigate a dream where gravity is fluid, colors are vivid, and the laws of physics bend."
  - **Theme**: Surrealism
  - **Art Style**: Abstract Art
  - **Camera Movement**: Fluid Transitions
- **Outcome**: Produce mesmerizing visuals that capture the essence of dreams.

### Environmental Awareness

- **Scenario**: Highlighting the impacts of climate change through visual storytelling.
- **Settings**:
  - **Concept**: "Contrast the pristine beauty of untouched ecosystems with scenes of environmental degradation."
  - **Theme**: Documentary, Environmental
  - **Art Style**: Realism
  - **Lighting**: Natural Light, Stark Contrasts
- **Outcome**: Generate powerful imagery that raises awareness and motivates action.

### Cultural Festivals Around the World

- **Scenario**: Showcasing the vibrancy of global festivals.
- **Settings**:
  - **Concept**: "Experience the colors and energy of festivals from Rio Carnival to Holi in India."
  - **Holiday Mode**: Enabled
  - **Theme**: Cultural, Celebratory
  - **Camera Movement**: Handheld, Immersive Angles
- **Outcome**: Create engaging content that celebrates diversity and culture.

### Inner Space Exploration

- **Scenario**: A journey inside the human body or mind.
- **Settings**:
  - **Concept**: "Travel through neural pathways, witnessing thoughts as bursts of light and emotion."
  - **Theme**: Science Fiction, Educational
  - **Art Style**: Biotech Art
  - **Lighting**: Luminous Glows
- **Outcome**: Produce educational and visually stunning content that explores inner worlds.

### Abstract Music Visualization

- **Scenario**: Creating visuals that represent music or sound.
- **Settings**:
  - **Concept**: "Visualize a symphony as an evolving landscape of shapes and colors that react to each note."
  - **Specific Modes**: SoundScape Mode
  - **Art Style**: Abstract Expressionism
  - **Camera Movement**: Dynamic, Sync with Music Tempo
- **Outcome**: Generate synesthetic experiences where sound and visuals merge.

These examples illustrate how the Temporal Prompt Engine can be tailored to a vast array of creative projects, pushing the boundaries of what's possible in visual and auditory storytelling.

---

<a name="8-harnessing-the-power-of-comfyui"></a>
## 8. Harnessing the Power of ComfyUI

The Temporal Prompt Engine is designed to work seamlessly with [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful tool for AI-based video generation.

- **Integration**: After generating your video prompts, use the included ComfyUI Workflow to produce your videos.
- **Process**:
  - **Import Prompts**: The engine exports prompts in a format compatible with ComfyUI.
  - **Generate Videos**: Utilize ComfyUI's advanced capabilities to bring your scenes to life. Story Mode works exceptionally well with ComfyUI, allowing for smooth transitions and cohesive narratives.
  - **Refinement**: Iterate on your prompts and settings based on the outputs, refining your vision.

By combining the strengths of the Temporal Prompt Engine and ComfyUI, you unlock a workflow that is both powerful and user-friendly, streamlining the path from concept to creation.

---

<a name="9-join-the-temporal-labs-journey"></a>
## 9. Join the Temporal Labs Journey

I am a lone futurist on a mission, dedicated to pushing the boundaries of what's possible in AI and technology. Temporal Labs LLC is not just about tools—it's about fostering a community of innovators, developers, artists, and visionaries.

### Seeking Collaborators and Support

- **Investors**: Partner with me to accelerate innovation and bring groundbreaking ideas to life.
- **Developers and Students**: Join the journey, contribute to cutting-edge projects, and grow alongside like-minded individuals.
- **Clients and Associates**: Let's work together to create custom solutions that meet your unique needs.
- **Supporters**: Your encouragement and support fuel the mission.

### Linux Development and Collaboration

I'm actively working on developing a dedicated Linux build for the Temporal Prompt Engine. This endeavor requires significant time and resources, and I'm willing to invest the necessary effort over the next few months. Your support through donations or collaboration can help bring this vision to fruition more rapidly.

---

<a name="10-donations-and-support"></a>
## 10. Donations and Support

Your support helps continue the mission of innovation and the development of tools like the Temporal Prompt Engine. If you find this tool valuable or wish to contribute to the development of the Linux build, consider making a donation.

### Crypto Donations

**Ethereum (ETH)**:

```
0x5616b3415ED6Ea7005595eF144A2054d4cD5767B
```

**Bitcoin (BTC)**:

```
bc1qpsfn8a7cs75fxxwv3ax7gtnurm44n5x2fmh59c
```

**Solana (SOL)**:

```
FVPGxfGT7QWfQEWvXpFkwdgiiKFM3VdvzNG6mEmX8pgi
```

**Litecoin (LTC)**:

```
ltc1qwlyjz8aymy9uagqhht5a4kaq06kmv58dxlzyww
```

**Dogecoin (DOGE)**:

```
DAeWAroHCy8nXCoUsobderPRSNXNu1WY34
```

You can copy any of these addresses to your clipboard for easy transfer.

### Venmo

**Venmo ID**:

```
@Utah-DM
```

---

<a name="11-additional-services-offered"></a>
## 11. Additional Services Offered

At Temporal Labs LLC, I offer a diverse range of services to help you achieve your goals:

- **Tutoring**: Personalized tutoring in AI, programming, and technology to help you or your team acquire new skills and knowledge.
- **Development**: Custom software and hardware development tailored to your specific needs, leveraging the latest technologies.
- **Design**: Creative design services, including UI/UX design, graphic design, and product conceptualization.
- **Consulting**: Expert advice and strategic planning in AI integration, technological solutions, and innovation strategies.
- **Workshops and Training**: Interactive sessions designed to educate and inspire, suitable for businesses, educational institutions, and organizations.

Whether you're looking to enhance your team's capabilities, develop a new product, or navigate the complexities of modern technology, I'm here to assist you every step of the way.

---

<a name="12-attribution-and-courtesy-request"></a>
## 12. Attribution and Courtesy Request

If you create videos or content using prompts generated by the Temporal Prompt Engine, we kindly request that you include the phrase **"using Temporal Prompt Engine"** somewhere in your final release. This attribution helps bring awareness to localized AI development practices and supports the growth of our community.

Your acknowledgment is greatly appreciated and contributes to fostering a collaborative environment where creators can discover and benefit from innovative tools like the Temporal Prompt Engine.

---

<a name="13-contact"></a>
## 13. Contact

For questions, support, collaboration opportunities, or to discuss how we can work together, I'd love to hear from you:

- **Email**: [Sol@TemporalLab.com](mailto:Sol@TemporalLab.com)
- **Phone**: +1-385-222-9920

---

<a name="14-acknowledgments"></a>
## 14. Acknowledgments

The development of the Temporal Prompt Engine leverages a variety of open-source tools and libraries:

- **Git**: Version control system.
- **Python**: Programming language.
- **FFmpeg**: Multimedia framework.
- **Ollama**: AI-powered tools.
- **AudioLDM2**: Audio generation model.
- **ComfyUI**: User interface for AI-based video generation.
- **HuggingFace**: AI models and APIs.

A special thanks to the developers and communities behind these tools for making innovative projects like the Temporal Prompt Engine possible.

---

[Back to Top](#temporal-prompt-engine-intuitive-scene-builder-for-video-and-audio-generation)
