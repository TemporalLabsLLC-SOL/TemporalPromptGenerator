# Temporal Prompt Engine: Intuitive Scene Builder for Video and Audio Generation

---

### Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Features Overview](#features-overview)
5. [How to Use the Temporal Prompt Engine](#how-to-use-the-temporal-prompt-engine)
6. [Settings Guide](#settings-guide)
7. [Harnessing the Power of ComfyUI](#harnessing-the-power-of-comfyui)
8. [Join the Temporal Labs Journey](#join-the-temporal-labs-journey)
9. [API Key Setup](#api-key-setup)
10. [Troubleshooting](#troubleshooting)
11. [Donations and Support](#donations-and-support)
12. [Contact](#contact)

---

<a name="introduction"></a>

## 1. Introduction

Welcome to the **Temporal Prompt Engine**, your ultimate tool for crafting immersive video and audio experiences. This engine empowers you to generate high-quality prompts with unparalleled control over cinematic elements, all while being intuitive and accessible for both Windows and Linux users.

Imagine directing a scene set in a cyberpunk future, with neon-lit streets captured through the lens of a 1980s camera, and accompanied by a rich, layered soundscape that adapts dynamically to your visuals. With the Temporal Prompt Engine, this isn't just possible—it's straightforward and exhilarating.

Whether you're a filmmaker, game developer, digital artist, or an AI enthusiast, this engine is designed to unleash your creativity and bring your visions to life with precision and style.

---

<a name="installation"></a>

## 2. Installation

Setting up the Temporal Prompt Engine is simple and hassle-free, allowing you to focus on creation rather than configuration.

1. **Download the Application**:
   - Clone or download the repository to your preferred folder.

2. **Install Dependencies via `requirements.txt`**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Windows Users**:
   - Run the `setup.bat` file to automatically configure environment variables and install necessary dependencies.

4. **For Linux Users**:
   - Ensure Python 3.8+ is installed.
   - Follow the same installation commands as above.

5. **Troubleshooting**:
   - If you encounter any issues, run the `troubleshoot.py` script to diagnose and fix common problems.

---

<a name="setting-up-the-environment"></a>

## 3. Setting Up the Environment

No complicated configurations or pre-setups are required. The engine guides you through the initial setup, prompting you for:

- **OpenAI API Key**: Enables video prompt generation using GPT models.
- **ElevenLabs API Key**: Allows for dynamic audio soundscape creation.
- **Output Directory**: Where all your generated media will be saved.
- **ComfyUI Prompts Folder**: If using ComfyUI, link your prompts folder for seamless integration.

Simply run the application, and you'll be prompted to enter these details when necessary. There's no need to edit any files manually—the engine handles it all for you.

---

<a name="features-overview"></a>

## 4. Features Overview

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

<a name="how-to-use-the-temporal-prompt-engine"></a>

## 5. How to Use the Temporal Prompt Engine

### **Step-by-Step Guide**

1. **Run the Application**:
   ```bash
   python temporal_prompt_engine.py
   ```

2. **Enter Your Scene Concept**:
   - Input your creative idea or scene description (up to 450 characters).
   - **Example**: "A futuristic cityscape bustling with flying cars, towering skyscrapers adorned with holographic advertisements, under the glow of a setting sun."

3. **Configure Video Prompt Options**:
   - Click on **Video Prompt Options** to customize settings.
   - **Example**:
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
   - Notice how selecting the **1980s** decade filters the available cameras to those from that era, and choosing the **Anamorphic** lens influences the aspect ratio and cinematic feel of your scene.

4. **Generate Video Prompts**:
   - After setting your preferences, click **Generate Video Prompts**.
   - The engine crafts detailed prompts, ensuring consistent and reliable scene generation.
   - The main prompt you entered influences the end prompt lists, integrating with your selected settings to produce a cohesive and vivid description ready for video generation.

5. **Configure Audio Prompt Options**:
   - Click on **Audio Prompt Options** to tailor your soundscape.
   - **Example**:
     - **Exclude Music**: Enabled (focus on ambient sounds)
     - **Holiday Mode**: Disabled
     - **Specific Modes**: SoundScape Mode
     - **Layer Intensity**: High (for a rich, multi-layered soundscape)
   - The choices here will generate audio prompts that emphasize environmental sounds like the hum of flying cars and distant chatter, enhancing the immersive experience.

6. **Generate Audio Prompts**:
   - Click **Generate Audio Prompts** to create audio that adapts dynamically to your video scenes.

7. **Generate Sound Effects**:
   - Specify the duration of each sound effect.
   - Click **Generate Sound Effects** to produce layered audio, enriching your scenes.

8. **Harnessing the Power of ComfyUI**:
   - Use the included [ComfyUI Workflow](https://openart.ai/workflows/sol_temporallabsllc/temporal-labs-llc-presents-cogvideo5b-infinity-study-6-second-generations/Uh4tTacq0UiYHdMjgk3P) to generate your videos.
   - The engine seamlessly integrates with ComfyUI, allowing you to take the generated video prompts and produce stunning visuals with ease.

9. **Combine Video and Audio**:
   - Use the **COMBINE** feature to merge your video and audio into a seamless, immersive experience.

10. **Save and Export**:
    - All media and prompts are saved in your designated output directory for easy access and future use.

---

<a name="settings-guide"></a>

## 6. Settings Guide

Discover the full potential of the Temporal Prompt Engine by exploring its comprehensive settings. Each setting is designed to influence the final output significantly, and understanding how they interact will elevate your creations.

### **Video Prompt Settings**

1. **Theme**:

   - **Description**: Sets the overall mood and genre.
   - **Examples**:
     - *Adventure*: Embark on epic journeys.
     - *Horror*: Create chilling atmospheres.
     - *Sci-Fi*: Dive into futuristic worlds.
   - **Impact**: The theme guides the AI to incorporate genre-specific elements into your prompts, ensuring the scene aligns with your creative vision.

2. **Art Style**:

   - **Description**: Defines the visual aesthetics.
   - **Examples**:
     - *Cyberpunk*: Neon lights and dystopian settings.
     - *Impressionism*: Soft edges and vibrant colors.
     - *Noir*: High contrast and shadow play.
   - **Interaction with Theme**: Combining *Sci-Fi* theme with *Cyberpunk* art style intensifies the futuristic and edgy feel of your scene.

3. **Lighting**:

   - **Description**: Influences the scene's illumination and mood.
   - **Examples**:
     - *Golden Hour*: Warm, soft lighting during sunrise or sunset.
     - *Neon Lighting*: Vibrant and colorful urban settings.
     - *Low Key*: Dramatic shadows and contrasts.
   - **Impact on Mood**: Lighting settings dramatically affect the emotional tone of your scene, and when paired with the right art style, can create a compelling atmosphere.

4. **Framing**:

   - **Description**: Controls the composition and perspective.
   - **Examples**:
     - *Close-Up*: Focus on a subject's details.
     - *Bird's Eye View*: Aerial perspectives for grand scenes.
     - *Dutch Angle*: Tilted shots for dynamic effects.
   - **Interaction with Camera Movement**: Choosing a *Tracking Shot* with a *Wide Shot* framing can create a sense of movement and scale, perfect for action sequences.

5. **Camera Movement**:

   - **Description**: Adds dynamism to your scenes.
   - **Examples**:
     - *Steadicam*: Smooth, fluid motion.
     - *Handheld*: Adds realism with subtle shakes.
     - *Crane*: Sweeping shots from high angles.
   - **Impact on Viewer Experience**: Camera movement settings influence how viewers perceive motion within your scene, adding depth and engagement.

6. **Time of Day**:

   - **Description**: Sets the temporal context.
   - **Examples**:
     - *Dawn*: Fresh starts and new beginnings.
     - *Midnight*: Mysterious and quiet settings.
   - **Influence on Lighting**: Time of day interacts with your lighting choices, affecting color palettes and shadow lengths.

7. **Decade**:

   - **Description**: Choose an era to influence camera selection.
   - **Examples**:
     - *1960s*: Vintage vibes with cameras like the **Arriflex 35**.
     - *2020s*: Modern aesthetics with cameras like the **RED Komodo**.
   - **Impact on Resolution and Aspect Ratio**: Selecting an older decade may limit resolution options to match historical capabilities, enhancing authenticity.

8. **Camera**:

   - **Description**: Select specific camera models for authenticity.
   - **Examples**:
     - *ARRI Alexa Mini LF*: For high-end, cinematic quality.
     - *Bell & Howell 2709*: To recreate early Hollywood charm.
   - **Influence on Lens and Resolution**: Certain cameras are compatible with specific lenses and support particular resolutions, so your camera choice can affect these settings.

9. **Lens**:

   - **Description**: Shapes the visual characteristics.
   - **Examples**:
     - *Anamorphic*: Wide-screen cinematic feel.
     - *Macro*: Extreme close-ups with fine details.
     - *Fisheye*: Ultra-wide angles with a distorted edge.
   - **Impact on Framing and Aspect Ratio**: An *Anamorphic* lens widens the aspect ratio, while a *Macro* lens enhances close-up framing.

10. **Resolution**:

    - **Description**: Guides the AI towards desired visual clarity.
    - **Examples**:
      - *4K UHD*: Crisp and detailed imagery.
      - *Standard Definition*: For a retro look.
    - **Interaction with Camera**: Some cameras, especially vintage ones, may influence the maximum achievable resolution, adding to the scene's authenticity.

11. **Specific Modes**:

    - **No People Mode**: Focus on landscapes or objects.
    - **SoundScape Mode**: Emphasizes environmental sounds.
    - **Story Mode**: Ensures a cohesive narrative flow.
    - **Chaos Mode**: Introduces unpredictability for creative effects.
    - **Remix Mode**: Randomizes selected parameters for variety.
    - **Holiday Mode**: Adds festive or seasonal elements.
    - **Impact on Overall Output**: These modes can dramatically alter the feel and content of your scene, providing unique and engaging results.

### **Audio Prompt Settings**

1. **Exclude Music**:

   - **Description**: Focuses on ambient sounds without musical elements.
   - **When to Use**: Ideal for creating realistic environments where background music would be unnatural.

2. **Holiday Mode**:

   - **Description**: Infuses holiday-specific sounds.
   - **Examples**:
     - *Christmas*: Sleigh bells and gentle snowfall.
     - *Halloween*: Eerie whispers and rustling leaves.
   - **Enhancing the Scene**: Adds thematic layers to your audio, enriching the viewer's experience.

3. **Specific Modes**:

   - **Description**: Tailor audio to genres or themes.
   - **Examples**:
     - *Documentary Mode*: Natural and informative soundscapes.
     - *Action Mode*: High-energy sounds and effects.
   - **Alignment with Video**: Ensures that the audio complements the visual style and narrative.

4. **Layer Intensity**:

   - **Description**: Adjusts the complexity of the audio.
   - **Impact**:
     - *Low Intensity*: Minimalist soundscapes, focusing on key sounds.
     - *High Intensity*: Rich, multi-layered audio environments with ambient sounds, foreground effects, and background atmospheres.

5. **Chaos Mode**:

   - **Description**: Adds creative unpredictability to your soundscapes.
   - **When to Use**: Perfect for experimental projects or to inspire new ideas.

6. **Story Mode**:

   - **Description**: Ensures audio flows cohesively with the visual narrative.
   - **Benefit**: Creates a unified and immersive experience, crucial for storytelling.

---

<a name="harnessing-the-power-of-comfyui"></a>

## 7. Harnessing the Power of ComfyUI

The Temporal Prompt Engine is designed to work seamlessly with [ComfyUI](https://comfyui.com/), a powerful tool for AI-based video generation.

- **Integration**: After generating your video prompts, use the included [ComfyUI Workflow](https://openart.ai/workflows/sol_temporallabsllc/temporal-labs-llc-presents-cogvideo5b-infinity-study-6-second-generations/Uh4tTacq0UiYHdMjgk3P) to produce your videos.
- **Process**:
  - **Import Prompts**: The engine exports prompts in a format compatible with ComfyUI.
  - **Generate Videos**: Utilize ComfyUI's advanced capabilities to bring your scenes to life.
  - **Refinement**: Iterate on your prompts and settings based on the outputs, refining your vision.

By combining the strengths of the Temporal Prompt Engine and ComfyUI, you unlock a workflow that is both powerful and user-friendly, streamlining the path from concept to creation.

---

<a name="join-the-temporal-labs-journey"></a>

## 8. Join the Temporal Labs Journey

I am a lone futurist on a mission, dedicated to pushing the boundaries of what's possible in AI and technology. Temporal Labs LLC is not just about tools—it's about fostering a community of innovators, developers, artists, and visionaries.

### **Seeking Collaborators and Support**

- **Investors**: Partner with me to accelerate innovation and bring groundbreaking ideas to life.
- **Developers and Students**: Join the journey, contribute to cutting-edge projects, and grow alongside like-minded individuals.
- **Clients and Associates**: Let's work together to create custom solutions that meet your unique needs.
- **Supporters**: Your encouragement and support fuel the mission.

### **Beyond the Temporal Prompt Engine**

Temporal Labs LLC offers a range of AI and tech services across software and hardware. Whether it's developing custom AI models, consulting on technological strategies, or crafting innovative hardware solutions, I'm committed to delivering excellence.

---

<a name="api-key-setup"></a>

## 9. API Key Setup

Unlock the full potential of the Temporal Prompt Engine by setting up your API keys.

1. **OpenAI API Key**:

   - **Purpose**: Enables advanced video prompt generation.
   - **Setup**:
     - Sign up at [OpenAI API](https://beta.openai.com/signup/).
     - Enter your key when prompted by the application.

2. **ElevenLabs API Key**:

   - **Purpose**: Allows for dynamic audio creation.
   - **Setup**:
     - Register at [ElevenLabs](https://elevenlabs.io/).
     - Input your API key when the application requests it.

**Note**: Your API keys are stored securely, and you only need to enter them once.

---

<a name="troubleshooting"></a>

## 10. Troubleshooting

If you encounter any issues, here's how to resolve them:

1. **Run the Troubleshoot Script**:

   ```bash
   python troubleshoot.py
   ```

2. **Common Solutions**:

   - **Missing Dependencies**: Ensure all packages from `requirements.txt` are installed.
   - **API Key Errors**: Verify that your API keys are correct and active.
   - **Permission Issues**: Check that you have the necessary read/write permissions for your directories.
   - **Compatibility**: Ensure that your Python version is 3.8 or higher.

3. **Need Further Assistance?**

   - Feel free to reach out via the contact information below.

---

<a name="donations-and-support"></a>

## 11. Donations and Support

Your support helps continue the mission of innovation and the development of tools like the Temporal Prompt Engine. If you find this tool valuable, consider making a donation.

### **Crypto Donations**

- **Bitcoin (BTC)**:
  - Address: `1A1zp1eP5QGefi2DMPTfTL5SLmv7DivfNa`

- **Ethereum (ETH)**:
  - Address: `0x742d35Cc6634C0532925a3b844Bc454e4438f44e`

- **Litecoin (LTC)**:
  - Address: `ltc1qwlyjz8aymy9uagqhht5a4kaq06kmv58dxlzyww`

- **Solana (SOL)**:
  - Address: `FVPGxfGT7QWfQEWvXpFkwdgiiKFM3VdvzNG6mEmX8pgi`

- **Dogecoin (DOGE)**:
  - Address: `DAeWAroHCy8nXCoUsobderPRSNXNu1WY34`

*You can copy any of these addresses to your clipboard for easy transfer.*

### **Venmo**

- **Venmo ID**: `@Utah-DM`

---

<a name="contact"></a>

## 12. Contact

For questions, support, or to discuss collaboration opportunities, I'd love to hear from you:

- **Email**: Sol@TemporalLab.com
- **Phone**: +1-385-222-9920

---

### **Embark on the Temporal Labs Journey**

This is more than just a tool—it's a gateway to infinite creative possibilities. The Temporal Prompt Engine is crafted to inspire and empower, offering you a canvas limited only by your imagination. With its extensive features and intuitive design, creating visually stunning and sonically rich scenes has never been more accessible.

Join me in exploring the future of AI-driven media creation. Your support fuels innovation, and I'm excited to see what you'll create.

---

Visit us at [www.TemporalLab.com](http://www.TemporalLab.com)

---

*Temporal Labs LLC is committed to ethical AI development, focusing on open-source solutions and community-driven innovation. By choosing the Temporal Prompt Engine, you're not just using a tool—you're joining a movement towards responsible and groundbreaking technological advancement.*

---