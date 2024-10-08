# Temporal Prompt Engine: Local, Open-Soure, Intuitive, Cinematic Prompt Engine + Video and Audio Generation Suite for Nvidia GPUs

<a name="9-local-video-generation-using-cogvideo"></a>
## Local Video Generation Using CogVideo

You can generate videos locally using **CogVideo**. The setup for CogVideo must be done **outside** of the Temporal Prompt Engine's virtual environment, as it is a separate application.

### Installation Steps

1. **Clone the CogVideo Repository into the Root Directory of TemporalPromptEngine - DO NOT HAVE THE TEMPORALPROMPTENGINE VENV ACTIVE**:
   ```bash
   git clone https://github.com/THUDM/CogVideo
   ```

2. **Copy Utility Script**:
   - Copy `PromptList2MP4Utility.py` into the `CogVideo/inference/gradio_composite_demo` folder.

3. **Create the Video Generation Environment**:
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

