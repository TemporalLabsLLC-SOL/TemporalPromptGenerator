import os
import random
import time
import gc
from typing import Optional, List, Union
import itertools
import re
import logging

import torch
from transformers import AutoTokenizer, pipeline

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ExifTags, Image
import subprocess

# --------------------- Configuration ---------------------
TOKENIZER_NAME = "gpt2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# Default values for all parameters
DEFAULTS = {
    "model": "HYVideo-T/2-cfgdistill",
    "precision": "bf16",
    "vae_tiling": True,
    "flow_shift": "1",  # Changed to string to support ranges
    "flow_reverse": True,
    "flow_solver": "euler",
    "batch_size": "1",  # Changed to string to support ranges
    "infer_steps": "50",  # Changed to string to support ranges
    "save_path": "./results",  # This will be overridden by the prompt directory
    "save_path_suffix": "",
    "name_suffix": "",
    "num_videos": "1",  # Changed to string to support ranges
    "video_size": "1280x720 (720p)",
    "video_length": "129",  # Changed to string to support ranges
    "seed": "1990",  # Changed to string to support ranges
    "neg_prompt": "",
    "cfg_scale": "1",  # Changed to string to support ranges
    "embedded_cfg_scale": "6",  # Changed to string to support ranges
    "reproduce": False,
    "ulysses_degree": "1",  # Changed to string to support ranges
    "ring_degree": "1",  # Changed to string to support ranges
    "audio_prompt": "",
    "audio_neg_prompt": "",
    "mmaudio_steps": "50",  # Changed to string to support ranges
    "cfg_strength": "4.5"  # Changed to string to support ranges
}

FIELDS_INFO = {
    "model": "Model type/name used.",
    "precision": "Computation precision, e.g., bf16, fp16.",
    "vae_tiling": "Enable VAE tiling for large resolution.",
    "flow_shift": "Shift factor for flow. Accepts single value or range (e.g., '1-10' or '1-10:2').",
    "flow_reverse": "Reverse generation order.",
    "flow_solver": "Flow solver type, e.g., euler.",
    "batch_size": "Batch size for generation. Accepts single value or range.",
    "infer_steps": "Number of inference steps. Accepts single value or range.",
    "save_path": "Path to save generated video.",
    "save_path_suffix": "Suffix to add to save path.",
    "name_suffix": "Suffix to add to output filenames.",
    "num_videos": "Number of videos to generate. Accepts single value or range.",
    "video_size": "Video resolution from the predefined list.",
    "video_length": "Number of frames in the video. Accepts single value or range.",
    "seed": "Random seed for reproducibility. Accepts single value or range.",
    "neg_prompt": "Negative prompt to avoid certain details.",
    "cfg_scale": "CFG scale for guidance. Accepts single value or range.",
    "embedded_cfg_scale": "Embedded CFG scale. Accepts single value or range.",
    "reproduce": "Try to reproduce deterministic results.",
    "ulysses_degree": "Ulysses degree setting (integer). Accepts single value or range.",
    "ring_degree": "Ring degree setting (integer). Accepts single value or range.",
    "audio_prompt": "Audio prompt to include in MMAudio.",
    "audio_neg_prompt": "Negative audio prompt to exclude in MMAudio.",
    "mmaudio_steps": "Number of MMAudio steps. Accepts single value or range.",
    "cfg_strength": "MMAudio CFG Strength. Accepts single value or range."
}

VIDEO_RESOLUTIONS = [
    "1280x720 (720p) [OFFICIAL SUPPORT]",
    "960x544 (544p) [OFFICIAL SUPPORT]",
    "960x480 (960h)",
    "820x480 (DVD)",
    "580x486 (LaserDisc)",
    "576x480 (SVCD)",
    "486x320 (VHS)",
    "480x352 (Xvid)",
    "372x240 (SNES)",
    "352x240 (VCD)",
    "320x240 (QVGA)",
    "256x224 (NES)",
    "160x144 (GBC)",
    "96x65 (Early Color Nokia)",
    "84x48 (Early Nokia)"
]

# Minimal sets of RESOLUTIONS, GUIDING_PHRASES, CAMERA_TERMS
RESOLUTIONS = {
    "1960s": [
        "Standard Definition (SD) - Mature film era, color or B&W",
        "Television 480i - Common household TV format",
        "CinemaScope and Panavision - Widescreen with moderate sharpness",
        "Early 720p Equivalent (Film) - Some experimental sharper formats"
    ],
    "1980s": [
        "Standard Definition (SD) - VHS era softness",
        "VHS Resolution - Visible scan lines and tracking noise",
        "LaserDisc 480p - Slightly sharper than VHS",
        "IMAX Film - Specialty theaters",
        "Betacam SP - Professional broadcast video"
    ],
    "2020s": [
        "4K UHD - Mainstream high clarity",
        "8K UHD - Cutting-edge sharpness",
        "12K Cinema - Experimental ultra-resolution",
        "1440p QHD - Popular in gaming monitors",
        "5.3K - Specialized action camera use"
    ],
    "Future": [
        "8K UHD - Possibly mainstream soon",
        "12K+ Cinema - Ultra-futuristic clarity",
        "16K - Hypothetical extreme resolution",
        "32K - Theoretical next-gen"
    ],
    "Experimental and Proto (Pre-1900s)": [
        "Very Low Definition - Experimental resolution...",
        "Glass Plate Quality - Gelatin dry plate..."
    ],
    "1900s": [
        "Glass Plate Standard...",
        "Low Definition Film...",
        "Low-Res Panoramic..."
    ],
    "1910s": [
        "Standard Definition (SD)...",
        "Silent Film Standard...",
        "Plate Film Quality..."
    ],
    "1920s": [
        "Standard Definition (SD)...",
        "Silent Film Standard...",
        "Panoramic Film Quality..."
    ],
    "1930s": [
        "Standard Definition (SD)...",
        "Film Resolution...",
        "Enhanced Silent Film Quality..."
    ],
    "1940s": [
        "Standard Definition (SD)...",
        "Film Quality...",
        "Early Widescreen..."
    ],
    "1950s": [
        "Standard Definition (SD)...",
        "CinemaScope Widescreen...",
        "Technicolor Quality...",
        "TV Standard Definition (480i)"
    ],
    "1970s": [
        "Standard Definition (SD)...",
        "Television 480i...",
        "Panavision Quality Film...",
        "IMAX Film (Limited)"
    ],
    "1990s": [
        "480p DVD Quality...",
        "720p HD (Early)...",
        "1080i HD (Broadcast)...",
        "IMAX Film...",
        "MiniDV 480p..."
    ],
    "2000s": [
        "720p HD...",
        "1080p Full HD...",
        "2K Digital Cinema...",
        "4K (Cinema)...",
        "IMAX Digital..."
    ],
    "2010s": [
        "1080p Full HD...",
        "2K Digital...",
        "4K UHD...",
        "5K...",
        "8K UHD (Experimental)"
    ]
}

DECADES = sorted(RESOLUTIONS.keys())

GUIDING_PHRASES = {
    "1960s": {
        "start": "(Enter an era of steady, familiar definition...)",
        "end": "(...dissolves, leaving a warm retro subtlety.)"
    },
    "1980s": {
        "start": "(Step into the soft haze of analog signals...)",
        "end": "(...tape runs out, leaving faint static memories.)"
    },
    "2020s": {
        "start": "(Enter a realm of ultra-high clarity...)",
        "end": "(...concludes, crystal-clear in immense fidelity.)"
    },
    "Future": {
        "start": "(Glimpse a speculative horizon...)",
        "end": "(...leaving only the concept of boundless clarity.)"
    },
    "Experimental and Proto (Pre-1900s)": {
        "start": "(You enter a world of faint silhouettes...)",
        "end": "(...the scene dissolves into a memory...)"
    },
    "1900s": {
        "start": "(Step into a realm of flickering frames...)",
        "end": "(...grainy and spectral, an afterimage of a distant past.)"
    },
    "1910s": {
        "start": "(Behold the fragile clarity of early black-and-white...)",
        "end": "(...fading with subtle dust and scratches.)"
    },
    "1920s": {
        "start": "(Step into the silent frames of a monochrome hush...)",
        "end": "(...flicker lingering like a gentle whisper.)"
    },
    "1930s": {
        "start": "(Enter a cinematic world of early sound...)",
        "end": "(...the curtain falls softly into memory.)"
    },
    "1940s": {
        "start": "(Immerse yourself in a classic tableau...)",
        "end": "(...the vision recedes with a gentle hum.)"
    },
    "1950s": {
        "start": "(Step into a time of early, vibrant color...)",
        "end": "(...fading like old postcards in an attic.)"
    },
    "1970s": {
        "start": "(Immerse in richly hued frames, grain and color...)",
        "end": "(...dims slowly, resting quietly in storage.)"
    },
    "1990s": {
        "start": "(Enter an era of cleaner home video...)",
        "end": "(...ends with a gentle fade, like a DVD menu.)"
    },
    "2000s": {
        "start": "(Immerse in the crisp rise of HD...)",
        "end": "(...fades, leaving an impression of newfound clarity.)"
    },
    "2010s": {
        "start": "(Step into a sharper world of digital cinema...)",
        "end": "(...recedes with polished definition, carefully archived.)"
    }
}

CAMERA_TERMS = {
    "1960s": [
        "steady, reliable film definition with mild grain",
        "broadcast television softness in 480i signals",
        "colors more stable but not ultra-crisp",
        "occasional lens vignetting at corners",
        "moderately sharp, still subdued details"
    ],
    "1980s": [
        "VHS-era softness with visible scan lines",
        "muted colors and occasional tracking noise",
        "tape-based playback artifacts and flicker",
        "low-resolution outlines and gentle blur",
        "a familiar, home-video warmth"
    ],
    "2020s": [
        "4K and beyond hyper-detail rendering",
        "extremely refined textures and depth",
        "perfectly balanced colors and lighting",
        "no noticeable grain, crystal clarity",
        "immersive, almost tactile visual presence"
    ],
    "Future": [
        "beyond 8K resolution, hyper-real fidelity",
        "virtually no visible artifacts",
        "limitless detail at extreme magnification",
        "colors and contrasts beyond current tech",
        "a vision exceeding present imagination"
    ],
    "Experimental and Proto (Pre-1900s)": [
        "extremely long exposures and blurred subjects",
        "soft-focus forms barely etched onto glass",
        "grainy textures and nearly monochrome tints",
        "hand-prepared emulsions with uneven chemistry",
        "ghostly silhouettes and very low contrast"
    ],
    "1900s": [
        "hand-cranked frames with visible flicker",
        "high contrast black and white film stock",
        "uneven focus and vignetting at the edges",
        "soft edges and subtle lens distortion",
        "no synchronized sound, purely silent motion"
    ],
    "1910s": [
        "slightly steadier hand-cranked sequences",
        "dust speckles and faint scratches on film",
        "limited tonal range in grayscale imagery",
        "soft edges and subtle lens distortion",
        "no synchronized sound, purely silent motion"
    ],
    "1920s": [
        "improved silent frames with mild flicker",
        "hand-tinted scenes adding gentle color hints",
        "soft grain and simple lighting contrasts",
        "simple title cards bridging narrative gaps",
        "no deep focus, foreground and background blur"
    ],
    "1930s": [
        "slightly steadier film with early sound integration",
        "enhanced lighting contrasts and shadow details",
        "subtle lens flares from early lighting tech",
        "balanced exposure with better dynamic range",
        "clearer focus with minimal motion blur"
    ],
    "1940s": [
        "refined black and white or early muted color",
        "better exposure balancing highlights and shadows",
        "mild grain and smoother camera panning",
        "slight lens halation around bright points",
        "a stable, classic cinema aesthetic"
    ],
    "1950s": [
        "early vibrant color stock with moderate saturation",
        "CinemaScope-like wide frames yet soft detail",
        "gentle film grain and slight lens flares",
        "limited dynamic range in bright sunlight",
        "tones that evoke a pastel postcard feel"
    ],
    "1970s": [
        "rich color with noticeable natural grain",
        "warmer hues and slightly soft edges",
        "steady tripod shots with occasional zoom",
        "subtle lens flare under strong lights",
        "overall earthy, nostalgic palette"
    ],
    "1990s": [
        "cleaner 480p DVD-like definition",
        "slightly sharper colors than VHS",
        "still some softness but improved stability",
        "rudimentary digital attempts, less grain",
        "sound more consistent but not fully HD"
    ],
    "2000s": [
        "1080p HD clarity with crisp edges",
        "vibrant colors and balanced contrasts",
        "minimal grain, more lifelike tones",
        "smoother motion and stable frame rates",
        "a clear leap beyond earlier decades"
    ],
    "2010s": [
        "ultra-clear digital cinema visuals",
        "high dynamic range and vivid hues",
        "sharp focus revealing subtle textures",
        "steady, stabilized camera motions",
        "polished, film-like grading with digital ease"
    ]
}

def sanitize_filename(filename: str) -> str:
    keepcharacters = (" ", ".", "_", "-")
    sanitized = "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()
    sanitized = sanitized.replace(" ", "_")
    return sanitized

def parse_prompt_file(lines: list) -> list:
    prompts = []
    current_prompt = {}
    current_section = None
    for idx, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if stripped_line.startswith("positive:"):
            if "positive" in current_prompt:
                logging.warning(f"New 'positive:' found before completing previous prompt at line {idx}. Skipping previous prompt.")
                current_prompt = {}
            current_prompt["positive"] = stripped_line[len("positive:"):].strip()
            current_section = "positive"
        elif stripped_line.startswith("negative:"):
            if "positive" not in current_prompt:
                logging.warning(f"'negative:' section without a preceding 'positive:' at line {idx}. Skipping.")
                current_section = None
                continue
            current_prompt["negative"] = stripped_line[len("negative:"):].strip()
            current_section = "negative"
        elif set(stripped_line) == set("-"):
            if "positive" in current_prompt and "negative" in current_prompt:
                prompts.append(current_prompt)
            else:
                if "positive" in current_prompt:
                    logging.warning(f"'negative:' section missing. Skipping.")
            current_prompt = {}
            current_section = None
        else:
            if current_section and current_section in current_prompt:
                current_prompt[current_section] += " " + stripped_line
            else:
                logging.warning(f"Unrecognized line at {idx}: '{stripped_line}'. Skipping.")
    if "positive" in current_prompt and "negative" in current_prompt:
        prompts.append(current_prompt)
    elif "positive" in current_prompt:
        logging.warning("Last prompt missing 'negative:' section. Skipping.")
    return prompts

def select_prompt_file() -> Optional[str]:
    file_path = filedialog.askopenfilename(
        title="Select Prompt List File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not file_path:
        messagebox.showwarning("No File Selected", "No prompt list file was selected. Exiting.")
        return None
    return file_path

def create_srt_file(video_path: str, subtitle_text: str, duration: float):
    try:
        base, _ = os.path.splitext(video_path)
        srt_path = f"{base}.srt"
        start_time = "00:00:00,000"
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration - int(duration)) * 1000)
        end_time = f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

        srt_content = f"""1
{start_time} --> {end_time}
{subtitle_text}
"""
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        logging.info(f"SRT file saved to: {srt_path}")
    except Exception as e:
        logging.error(f"Error creating SRT file: {e}")
        messagebox.showerror("SRT Generation Error", f"Error creating SRT file:\n{e}")

def get_selected_args():
    dialog = ArgsDialog()
    args = dialog.result
    if not args:
        args = DEFAULTS.copy()
    else:
        for k, v in DEFAULTS.items():
            if k not in args:
                args[k] = v
    return args

def parse_range(input_str: str, param_type: type) -> List[Union[int, float]]:
    """
    Parses a string input that can be a single value or a range.
    Examples:
        '5' -> [5]
        '1-10' -> [1, 2, ..., 10]
        '1-10:2' -> [1, 3, 5, 7, 9]
    """
    input_str = input_str.strip()
    if '-' in input_str:
        try:
            range_part, *step_part = input_str.split(':')
            start, end = map(float, range_part.split('-'))
            step = float(step_part[0]) if step_part else 1.0
            if param_type == int:
                start, end, step = int(start), int(end), int(step)
            num_steps = int((end - start) / step) + 1
            return [param_type(start + step * i) for i in range(num_steps) if start + step * i <= end]
        except Exception as e:
            raise ValueError(f"Invalid range format: '{input_str}'. Use 'start-end' or 'start-end:step'. Error: {e}")
    else:
        try:
            return [param_type(input_str)]
        except ValueError as ve:
            raise ValueError(f"Invalid value: '{input_str}'. Expected a single {param_type.__name__} or a range.")

def get_decade_from_prompt(prompt: str) -> str:
    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2}|21\d{2})", prompt)
    if years:
        year = int(years[0])
        decade = (year // 10) * 10
        for d in DECADES:
            if d.endswith('s'):
                try:
                    d_year = int(d[:-1])
                    if d_year == decade:
                        return d
                except ValueError:
                    continue
        if decade < 1900:
            return "Experimental and Proto (Pre-1900s)"
        elif decade >= 2020:
            return "2020s"
        else:
            # Default to "1960s" if no exact match found
            return "1960s"
    else:
        return "1960s"

def inject_resolution_terms(prompt: str) -> str:
    decade = get_decade_from_prompt(prompt)
    if decade in RESOLUTIONS:
        resolution_option = random.choice(RESOLUTIONS[decade])
        return prompt + " " + resolution_option
    return prompt

def inject_camera_terms(prompt: str) -> str:
    decade = get_decade_from_prompt(prompt)
    if decade not in CAMERA_TERMS:
        decade = "1960s"
    start_phrase = GUIDING_PHRASES[decade]["start"]
    end_phrase = GUIDING_PHRASES[decade]["end"]
    terms = CAMERA_TERMS[decade]
    chosen_terms = " ".join(terms)
    return f"{start_phrase} {prompt} {chosen_terms} {end_phrase}"

class ArgsDialog:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.result = None
        self.create_dialog()

    def create_dialog(self):
        dialog = MainDialog(self.root)
        self.result = dialog.result

class MainDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Configure Video and Audio Generation")
        self.resizable(True, True)
        self.grab_set()
        self.result = None
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        apply_theme(self)  # Apply theme to the dialog
        self.wait_window()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Video Settings Tab
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video Settings")
        self.create_video_settings(video_frame)

        # Audio Settings Tab
        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio Settings")
        self.create_audio_settings(audio_frame)

        # Output Settings Tab
        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text="Output Settings")
        self.create_output_settings(output_frame)

        # Advanced Settings Tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced Settings")
        self.create_advanced_settings(advanced_frame)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side='right')

    def create_video_settings(self, frame):
        # Video Size
        ttk.Label(frame, text="Video Size:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.video_size = ttk.Combobox(frame, values=VIDEO_RESOLUTIONS, state="readonly")
        self.video_size.set(DEFAULTS["video_size"])
        self.video_size.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Video Length
        ttk.Label(frame, text="Video Length (frames):", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.video_length = ttk.Entry(frame)
        self.video_length.insert(0, DEFAULTS["video_length"])
        self.video_length.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=1, column=2, text="e.g., 100 or 100-200:10")

        # Number of Videos
        ttk.Label(frame, text="Number of Videos:", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.num_videos = ttk.Entry(frame)
        self.num_videos.insert(0, DEFAULTS["num_videos"])
        self.num_videos.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=2, column=2, text="e.g., 1 or 1-5")

        # Seed
        ttk.Label(frame, text="Random Seed:", anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.seed = ttk.Entry(frame)
        self.seed.insert(0, DEFAULTS["seed"])
        self.seed.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=3, column=2, text="e.g., 1990 or 1000-1005")

        # Negative Prompt
        ttk.Label(frame, text="Negative Prompt:", anchor='w').grid(row=4, column=0, sticky='nw', padx=5, pady=5)
        self.neg_prompt = tk.Text(frame, height=4, width=40, bg=COLOR_PALETTE["white"], fg=COLOR_PALETTE["dark_blue_black"])
        self.neg_prompt.insert(tk.END, DEFAULTS["neg_prompt"])
        self.neg_prompt.grid(row=4, column=1, sticky='ew', padx=5, pady=5)

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def create_audio_settings(self, frame):
        # Audio Prompt
        ttk.Label(frame, text="Audio Prompt:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.audio_prompt = ttk.Entry(frame, width=50)
        self.audio_prompt.insert(0, DEFAULTS["audio_prompt"])
        self.audio_prompt.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Audio Negative Prompt
        ttk.Label(frame, text="Audio Negative Prompt:", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.audio_neg_prompt = ttk.Entry(frame, width=50)
        self.audio_neg_prompt.insert(0, DEFAULTS["audio_neg_prompt"])
        self.audio_neg_prompt.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def create_output_settings(self, frame):
        # Save Path
        ttk.Label(frame, text="Save Path:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.save_path = ttk.Entry(frame, width=50)
        self.save_path.insert(0, DEFAULTS["save_path"])
        self.save_path.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_save_path).grid(row=0, column=2, sticky='w', padx=5, pady=5)

        # Save Path Suffix
        ttk.Label(frame, text="Save Path Suffix:", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.save_path_suffix = ttk.Entry(frame)
        self.save_path_suffix.insert(0, DEFAULTS["save_path_suffix"])
        self.save_path_suffix.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # Name Suffix
        ttk.Label(frame, text="Name Suffix:", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.name_suffix = ttk.Entry(frame)
        self.name_suffix.insert(0, DEFAULTS["name_suffix"])
        self.name_suffix.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def create_advanced_settings(self, frame):
        # Model
        ttk.Label(frame, text="Model:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.model = ttk.Entry(frame)
        self.model.insert(0, DEFAULTS["model"])
        self.model.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Precision
        ttk.Label(frame, text="Precision:", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.precision = ttk.Combobox(frame, values=["bf16", "fp16", "fp32"], state="readonly")
        self.precision.set(DEFAULTS["precision"])
        self.precision.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # CFG Scale
        ttk.Label(frame, text="CFG Scale:", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.cfg_scale = ttk.Entry(frame)
        self.cfg_scale.insert(0, DEFAULTS["cfg_scale"])
        self.cfg_scale.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=2, column=2, text="e.g., 1 or 1-5:0.5")

        # Embedded CFG Scale
        ttk.Label(frame, text="Embedded CFG Scale:", anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.embedded_cfg_scale = ttk.Entry(frame)
        self.embedded_cfg_scale.insert(0, DEFAULTS["embedded_cfg_scale"])
        self.embedded_cfg_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=3, column=2, text="e.g., 6 or 6-8:1")

        # Reproduce
        self.reproduce = tk.BooleanVar(value=DEFAULTS["reproduce"])
        ttk.Checkbutton(frame, text="Reproduce Results", variable=self.reproduce).grid(row=4, column=1, sticky='w', padx=5, pady=5)

        # Ulysses Degree
        ttk.Label(frame, text="Ulysses Degree (int):", anchor='w').grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.ulysses_degree = ttk.Entry(frame)
        self.ulysses_degree.insert(0, DEFAULTS["ulysses_degree"])
        self.ulysses_degree.grid(row=5, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=5, column=2, text="e.g., 1 or 1-3")

        # Ring Degree
        ttk.Label(frame, text="Ring Degree (int):", anchor='w').grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.ring_degree = ttk.Entry(frame)
        self.ring_degree.insert(0, DEFAULTS["ring_degree"])
        self.ring_degree.grid(row=6, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=6, column=2, text="e.g., 1 or 1-2")

        # MMAudio Steps
        ttk.Label(frame, text="MMAudio Steps:", anchor='w').grid(row=7, column=0, sticky='w', padx=5, pady=5)
        self.mmaudio_steps = ttk.Entry(frame)
        self.mmaudio_steps.insert(0, DEFAULTS["mmaudio_steps"])
        self.mmaudio_steps.grid(row=7, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=7, column=2, text="e.g., 50 or 50-100:10")

        # MMAudio CFG Strength
        ttk.Label(frame, text="MMAudio CFG Strength:", anchor='w').grid(row=8, column=0, sticky='w', padx=5, pady=5)
        self.cfg_strength = ttk.Entry(frame)
        self.cfg_strength.insert(0, DEFAULTS["cfg_strength"])
        self.cfg_strength.grid(row=8, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=8, column=2, text="e.g., 4.5 or 4.5-6.0:0.5")

        # VAE Tiling
        self.vae_tiling = tk.BooleanVar(value=DEFAULTS["vae_tiling"])
        ttk.Checkbutton(frame, text="Enable VAE Tiling", variable=self.vae_tiling).grid(row=9, column=1, sticky='w', padx=5, pady=5)

        # Flow Reverse
        self.flow_reverse = tk.BooleanVar(value=DEFAULTS["flow_reverse"])
        ttk.Checkbutton(frame, text="Enable Flow Reverse", variable=self.flow_reverse).grid(row=10, column=1, sticky='w', padx=5, pady=5)

        # Flow Solver
        ttk.Label(frame, text="Flow Solver:", anchor='w').grid(row=11, column=0, sticky='w', padx=5, pady=5)
        self.flow_solver = ttk.Combobox(frame, values=["euler", "other_solver"], state="readonly")
        self.flow_solver.set(DEFAULTS["flow_solver"])
        self.flow_solver.grid(row=11, column=1, sticky='ew', padx=5, pady=5)

        # Flow Shift
        ttk.Label(frame, text="Flow Shift:", anchor='w').grid(row=12, column=0, sticky='w', padx=5, pady=5)
        self.flow_shift = ttk.Entry(frame)
        self.flow_shift.insert(0, DEFAULTS["flow_shift"])
        self.flow_shift.grid(row=12, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=12, column=2, text="e.g., 7 or 7-10:1")

        # Batch Size
        ttk.Label(frame, text="Batch Size:", anchor='w').grid(row=13, column=0, sticky='w', padx=5, pady=5)
        self.batch_size = ttk.Entry(frame)
        self.batch_size.insert(0, DEFAULTS["batch_size"])
        self.batch_size.grid(row=13, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=13, column=2, text="e.g., 1 or 1-3")

        # Infer Steps
        ttk.Label(frame, text="Inference Steps:", anchor='w').grid(row=14, column=0, sticky='w', padx=5, pady=5)
        self.infer_steps = ttk.Entry(frame)
        self.infer_steps.insert(0, DEFAULTS["infer_steps"])
        self.infer_steps.grid(row=14, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=14, column=2, text="e.g., 50 or 50-100:10")

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def add_range_hint(self, frame, row, column, text):
        ttk.Label(frame, text=text, anchor='w', foreground=COLOR_PALETTE["light_gold"]).grid(row=row, column=column, sticky='w', padx=5, pady=5)

    def browse_save_path(self):
        path = filedialog.askdirectory(title="Select Save Path")
        if path:
            self.save_path.delete(0, tk.END)
            self.save_path.insert(0, path)

    def on_ok(self):
        try:
            # Validate and collect all inputs
            data = {}

            # Video Settings
            data["video_size"] = self.video_size.get()
            data["video_length"] = self.video_length.get()
            data["num_videos"] = self.num_videos.get()
            data["seed"] = self.seed.get()
            data["neg_prompt"] = self.neg_prompt.get("1.0", tk.END).strip()

            # Audio Settings
            data["audio_prompt"] = self.audio_prompt.get().strip()
            data["audio_neg_prompt"] = self.audio_neg_prompt.get().strip()

            # Output Settings
            data["save_path"] = self.save_path.get().strip()
            data["save_path_suffix"] = self.save_path_suffix.get().strip()
            data["name_suffix"] = self.name_suffix.get().strip()

            # Advanced Settings
            data["model"] = self.model.get().strip()
            data["precision"] = self.precision.get().strip()
            data["cfg_scale"] = self.cfg_scale.get().strip()
            data["embedded_cfg_scale"] = self.embedded_cfg_scale.get().strip()
            data["reproduce"] = self.reproduce.get()
            data["ulysses_degree"] = self.ulysses_degree.get().strip()
            data["ring_degree"] = self.ring_degree.get().strip()
            data["mmaudio_steps"] = self.mmaudio_steps.get().strip()
            data["cfg_strength"] = self.cfg_strength.get().strip()
            data["flow_shift"] = self.flow_shift.get().strip()
            data["batch_size"] = self.batch_size.get().strip()
            data["infer_steps"] = self.infer_steps.get().strip()
            data["flow_solver"] = self.flow_solver.get().strip()
            data["vae_tiling"] = self.vae_tiling.get()
            data["flow_reverse"] = self.flow_reverse.get()

            # Parse all range or single value inputs
            param_names = ["video_length", "num_videos", "seed", "cfg_scale", "embedded_cfg_scale",
                          "ulysses_degree", "ring_degree", "mmaudio_steps", "cfg_strength",
                          "flow_shift", "batch_size", "infer_steps"]
            parsed_args = {}
            for param in param_names:
                value = data[param]
                if param in ["cfg_scale", "cfg_strength"]:
                    parsed_args[param] = parse_range(value, float)
                else:
                    parsed_args[param] = parse_range(value, int)

            # Fixed parameters
            parsed_args["model"] = data["model"]
            parsed_args["precision"] = data["precision"]
            parsed_args["save_path"] = data["save_path"]
            parsed_args["save_path_suffix"] = data["save_path_suffix"]
            parsed_args["name_suffix"] = data["name_suffix"]
            parsed_args["video_size"] = data["video_size"]
            parsed_args["neg_prompt"] = data["neg_prompt"]
            parsed_args["audio_prompt"] = data["audio_prompt"]
            parsed_args["audio_neg_prompt"] = data["audio_neg_prompt"]
            parsed_args["reproduce"] = data["reproduce"]
            parsed_args["flow_reverse"] = data["flow_reverse"]
            parsed_args["flow_solver"] = data["flow_solver"]
            parsed_args["vae_tiling"] = data["vae_tiling"]

            self.result = parsed_args
            self.destroy()
        except ValueError as ve:
            logging.error(f"Invalid input: {ve}")
            messagebox.showerror("Invalid Input", f"Please ensure all inputs are correct.\nError: {ve}")

    def on_cancel(self):
        self.destroy()

def generate_video(args: dict, prompt: str, negative_prompt: str) -> str:
    """
    Generates a video using sample_video.py with the given prompts and arguments.
    Returns the path to the generated temporary video.
    """
    resolution_str = args["video_size"].split(" ")[0]
    width_str, height_str = resolution_str.split("x")
    width, height = int(width_str), int(height_str)

    enriched_prompt = inject_resolution_terms(prompt)
    enriched_prompt = inject_camera_terms(enriched_prompt)

    logging.debug(f"Enriched Positive Prompt: {enriched_prompt}")
    logging.debug(f"Negative Prompt: {negative_prompt}")

    # Use the passed 'negative_prompt' instead of 'args["neg_prompt"]'
    full_neg_prompt = (
        negative_prompt +
        ", Camera, Lens, Tripod, deformed limbs, Bad Focus, Bad Framing, Missing limbs, Glitching, Watermark, transition, transforming, transition swipe, swipe, scene swipe, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, mutation, disgusting, bad anatomy, bad proportions, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, hovering unattractive, unnatural pose, unsightly, written language, transition"
    )

    logging.info(f"Constructed full_neg_prompt: {full_neg_prompt}")

    # Define a temporary filename to prevent overwriting
    temp_video_name = f"temp_video_{random.randint(1000,9999)}.mp4"
    temp_video_path = os.path.join(args["save_path"], temp_video_name)

    cmd = [
        "python3", "sample_video.py",
        "--model", args["model"],
        "--precision", args["precision"],
        "--flow-shift", str(args["flow_shift"]),
        "--flow-solver", args["flow_solver"],
        "--batch-size", str(args["batch_size"]),
        "--infer-steps", str(args["infer_steps"]),
        "--save-path", args["save_path"],
        "--num-videos", "1",  # Generate one video at a time
        "--video-size", str(width), str(height),
        "--video-length", str(args["video_length"]),
        "--prompt", enriched_prompt,
        "--seed", str(args["seed"]),
        "--neg-prompt", full_neg_prompt,
        "--cfg-scale", str(args["cfg_scale"]),
        "--embedded-cfg-scale", str(args["embedded_cfg_scale"]),
        "--ulysses-degree", str(args["ulysses_degree"]),
        "--ring-degree", str(args["ring_degree"])
        # Removed '--output' argument
    ]

    if args["vae_tiling"]:
        cmd.append("--vae-tiling")
    if args["flow_reverse"]:
        cmd.append("--flow-reverse")
    if args["save_path_suffix"]:
        cmd.extend(["--save-path-suffix", args["save_path_suffix"]])
    if args["name_suffix"]:
        cmd.extend(["--name-suffix", args["name_suffix"]])
    if args["reproduce"]:
        cmd.append("--reproduce")

    logging.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, text=True)

    # After generation, identify the newly created video
    generated_files_after = set(os.listdir(args["save_path"]))
    # Assuming sample_video.py saves the video with a specific naming convention,
    # we'll find the latest .mp4 file in the save_path
    mp4_files = [f for f in generated_files_after if f.endswith(".mp4")]
    if not mp4_files:
        raise FileNotFoundError("No generated .mp4 file found.")

    # Get the latest modified .mp4 file
    latest_mp4 = max(
        mp4_files,
        key=lambda x: os.path.getmtime(os.path.join(args["save_path"], x))
    )
    latest_mp4_path = os.path.join(args["save_path"], latest_mp4)

    logging.info(f"Generated video at: {latest_mp4_path}")
    return latest_mp4_path

# --------------------- Theming Configuration ---------------------
# Define the color palette
COLOR_PALETTE = {
    "white": "#ffffff",
    "dark_blue_black": "#1a1a2e",
    "gold": "#ffab00",
    "dark_navy": "#040e24",
    "deep_gold": "#ffa200",
    "light_gold": "#ffc14e"
}

def apply_theme(root):
    style = ttk.Style()
    style.theme_use('default')

    # Configure styles
    style.configure('TFrame', background=COLOR_PALETTE["dark_blue_black"])
    style.configure('TLabel', background=COLOR_PALETTE["dark_blue_black"], foreground=COLOR_PALETTE["white"])
    style.configure('TButton', background=COLOR_PALETTE["gold"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.configure('TEntry', fieldbackground=COLOR_PALETTE["white"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.configure('TCombobox', fieldbackground=COLOR_PALETTE["white"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.configure('TNotebook', background=COLOR_PALETTE["dark_blue_black"])
    style.configure('TNotebook.Tab', background=COLOR_PALETTE["gold"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.map('TButton',
              background=[('active', COLOR_PALETTE["deep_gold"])],
              foreground=[('active', COLOR_PALETTE["dark_blue_black"])])
    style.configure("TText", background=COLOR_PALETTE["white"], foreground=COLOR_PALETTE["dark_blue_black"])

    # Set the root window background
    root.configure(background=COLOR_PALETTE["dark_blue_black"])

# --------------------- Dialog Classes ---------------------
class ArgsDialog:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.result = None
        self.create_dialog()

    def create_dialog(self):
        dialog = MainDialog(self.root)
        self.result = dialog.result

class MainDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Configure Video and Audio Generation")
        self.resizable(True, True)
        self.grab_set()
        self.result = None
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        apply_theme(self)  # Apply theme to the dialog
        self.wait_window()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Video Settings Tab
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video Settings")
        self.create_video_settings(video_frame)

        # Audio Settings Tab
        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio Settings")
        self.create_audio_settings(audio_frame)

        # Output Settings Tab
        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text="Output Settings")
        self.create_output_settings(output_frame)

        # Advanced Settings Tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced Settings")
        self.create_advanced_settings(advanced_frame)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side='right')

    def create_video_settings(self, frame):
        # Video Size
        ttk.Label(frame, text="Video Size:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.video_size = ttk.Combobox(frame, values=VIDEO_RESOLUTIONS, state="readonly")
        self.video_size.set(DEFAULTS["video_size"])
        self.video_size.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Video Length
        ttk.Label(frame, text="Video Length (frames):", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.video_length = ttk.Entry(frame)
        self.video_length.insert(0, DEFAULTS["video_length"])
        self.video_length.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=1, column=2, text="e.g., 100 or 100-200:10")

        # Number of Videos
        ttk.Label(frame, text="Number of Videos:", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.num_videos = ttk.Entry(frame)
        self.num_videos.insert(0, DEFAULTS["num_videos"])
        self.num_videos.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=2, column=2, text="e.g., 1 or 1-5")

        # Seed
        ttk.Label(frame, text="Random Seed:", anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.seed = ttk.Entry(frame)
        self.seed.insert(0, DEFAULTS["seed"])
        self.seed.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=3, column=2, text="e.g., 1990 or 1000-1005")

        # Negative Prompt
        ttk.Label(frame, text="Negative Prompt:", anchor='w').grid(row=4, column=0, sticky='nw', padx=5, pady=5)
        self.neg_prompt = tk.Text(frame, height=4, width=40, bg=COLOR_PALETTE["white"], fg=COLOR_PALETTE["dark_blue_black"])
        self.neg_prompt.insert(tk.END, DEFAULTS["neg_prompt"])
        self.neg_prompt.grid(row=4, column=1, sticky='ew', padx=5, pady=5)

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def create_audio_settings(self, frame):
        # Audio Prompt
        ttk.Label(frame, text="Audio Prompt:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.audio_prompt = ttk.Entry(frame, width=50)
        self.audio_prompt.insert(0, DEFAULTS["audio_prompt"])
        self.audio_prompt.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Audio Negative Prompt
        ttk.Label(frame, text="Audio Negative Prompt:", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.audio_neg_prompt = ttk.Entry(frame, width=50)
        self.audio_neg_prompt.insert(0, DEFAULTS["audio_neg_prompt"])
        self.audio_neg_prompt.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def create_output_settings(self, frame):
        # Save Path
        ttk.Label(frame, text="Save Path:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.save_path = ttk.Entry(frame, width=50)
        self.save_path.insert(0, DEFAULTS["save_path"])
        self.save_path.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_save_path).grid(row=0, column=2, sticky='w', padx=5, pady=5)

        # Save Path Suffix
        ttk.Label(frame, text="Save Path Suffix:", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.save_path_suffix = ttk.Entry(frame)
        self.save_path_suffix.insert(0, DEFAULTS["save_path_suffix"])
        self.save_path_suffix.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # Name Suffix
        ttk.Label(frame, text="Name Suffix:", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.name_suffix = ttk.Entry(frame)
        self.name_suffix.insert(0, DEFAULTS["name_suffix"])
        self.name_suffix.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def create_advanced_settings(self, frame):
        # Model
        ttk.Label(frame, text="Model:", anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.model = ttk.Entry(frame)
        self.model.insert(0, DEFAULTS["model"])
        self.model.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Precision
        ttk.Label(frame, text="Precision:", anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.precision = ttk.Combobox(frame, values=["bf16", "fp16", "fp32"], state="readonly")
        self.precision.set(DEFAULTS["precision"])
        self.precision.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        # CFG Scale
        ttk.Label(frame, text="CFG Scale:", anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.cfg_scale = ttk.Entry(frame)
        self.cfg_scale.insert(0, DEFAULTS["cfg_scale"])
        self.cfg_scale.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=2, column=2, text="e.g., 1 or 1-5:0.5")

        # Embedded CFG Scale
        ttk.Label(frame, text="Embedded CFG Scale:", anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.embedded_cfg_scale = ttk.Entry(frame)
        self.embedded_cfg_scale.insert(0, DEFAULTS["embedded_cfg_scale"])
        self.embedded_cfg_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=3, column=2, text="e.g., 6 or 6-8:1")

        # Reproduce
        self.reproduce = tk.BooleanVar(value=DEFAULTS["reproduce"])
        ttk.Checkbutton(frame, text="Reproduce Results", variable=self.reproduce).grid(row=4, column=1, sticky='w', padx=5, pady=5)

        # Ulysses Degree
        ttk.Label(frame, text="Ulysses Degree (int):", anchor='w').grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.ulysses_degree = ttk.Entry(frame)
        self.ulysses_degree.insert(0, DEFAULTS["ulysses_degree"])
        self.ulysses_degree.grid(row=5, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=5, column=2, text="e.g., 1 or 1-3")

        # Ring Degree
        ttk.Label(frame, text="Ring Degree (int):", anchor='w').grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.ring_degree = ttk.Entry(frame)
        self.ring_degree.insert(0, DEFAULTS["ring_degree"])
        self.ring_degree.grid(row=6, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=6, column=2, text="e.g., 1 or 1-2")

        # MMAudio Steps
        ttk.Label(frame, text="MMAudio Steps:", anchor='w').grid(row=7, column=0, sticky='w', padx=5, pady=5)
        self.mmaudio_steps = ttk.Entry(frame)
        self.mmaudio_steps.insert(0, DEFAULTS["mmaudio_steps"])
        self.mmaudio_steps.grid(row=7, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=7, column=2, text="e.g., 50 or 50-100:10")

        # MMAudio CFG Strength
        ttk.Label(frame, text="MMAudio CFG Strength:", anchor='w').grid(row=8, column=0, sticky='w', padx=5, pady=5)
        self.cfg_strength = ttk.Entry(frame)
        self.cfg_strength.insert(0, DEFAULTS["cfg_strength"])
        self.cfg_strength.grid(row=8, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=8, column=2, text="e.g., 4.5 or 4.5-6.0:0.5")

        # VAE Tiling
        self.vae_tiling = tk.BooleanVar(value=DEFAULTS["vae_tiling"])
        ttk.Checkbutton(frame, text="Enable VAE Tiling", variable=self.vae_tiling).grid(row=9, column=1, sticky='w', padx=5, pady=5)

        # Flow Reverse
        self.flow_reverse = tk.BooleanVar(value=DEFAULTS["flow_reverse"])
        ttk.Checkbutton(frame, text="Enable Flow Reverse", variable=self.flow_reverse).grid(row=10, column=1, sticky='w', padx=5, pady=5)

        # Flow Solver
        ttk.Label(frame, text="Flow Solver:", anchor='w').grid(row=11, column=0, sticky='w', padx=5, pady=5)
        self.flow_solver = ttk.Combobox(frame, values=["euler", "other_solver"], state="readonly")
        self.flow_solver.set(DEFAULTS["flow_solver"])
        self.flow_solver.grid(row=11, column=1, sticky='ew', padx=5, pady=5)

        # Flow Shift
        ttk.Label(frame, text="Flow Shift:", anchor='w').grid(row=12, column=0, sticky='w', padx=5, pady=5)
        self.flow_shift = ttk.Entry(frame)
        self.flow_shift.insert(0, DEFAULTS["flow_shift"])
        self.flow_shift.grid(row=12, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=12, column=2, text="e.g., 7 or 7-10:1")

        # Batch Size
        ttk.Label(frame, text="Batch Size:", anchor='w').grid(row=13, column=0, sticky='w', padx=5, pady=5)
        self.batch_size = ttk.Entry(frame)
        self.batch_size.insert(0, DEFAULTS["batch_size"])
        self.batch_size.grid(row=13, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=13, column=2, text="e.g., 1 or 1-3")

        # Infer Steps
        ttk.Label(frame, text="Inference Steps:", anchor='w').grid(row=14, column=0, sticky='w', padx=5, pady=5)
        self.infer_steps = ttk.Entry(frame)
        self.infer_steps.insert(0, DEFAULTS["infer_steps"])
        self.infer_steps.grid(row=14, column=1, sticky='ew', padx=5, pady=5)
        self.add_range_hint(frame, row=14, column=2, text="e.g., 50 or 50-100:10")

        # Grid configuration
        frame.columnconfigure(1, weight=1)

    def add_range_hint(self, frame, row, column, text):
        ttk.Label(frame, text=text, anchor='w', foreground=COLOR_PALETTE["light_gold"]).grid(row=row, column=column, sticky='w', padx=5, pady=5)

    def browse_save_path(self):
        path = filedialog.askdirectory(title="Select Save Path")
        if path:
            self.save_path.delete(0, tk.END)
            self.save_path.insert(0, path)

    def on_ok(self):
        try:
            # Validate and collect all inputs
            data = {}

            # Video Settings
            data["video_size"] = self.video_size.get()
            data["video_length"] = self.video_length.get()
            data["num_videos"] = self.num_videos.get()
            data["seed"] = self.seed.get()
            data["neg_prompt"] = self.neg_prompt.get("1.0", tk.END).strip()

            # Audio Settings
            data["audio_prompt"] = self.audio_prompt.get().strip()
            data["audio_neg_prompt"] = self.audio_neg_prompt.get().strip()

            # Output Settings
            data["save_path"] = self.save_path.get().strip()
            data["save_path_suffix"] = self.save_path_suffix.get().strip()
            data["name_suffix"] = self.name_suffix.get().strip()

            # Advanced Settings
            data["model"] = self.model.get().strip()
            data["precision"] = self.precision.get().strip()
            data["cfg_scale"] = self.cfg_scale.get().strip()
            data["embedded_cfg_scale"] = self.embedded_cfg_scale.get().strip()
            data["reproduce"] = self.reproduce.get()
            data["ulysses_degree"] = self.ulysses_degree.get().strip()
            data["ring_degree"] = self.ring_degree.get().strip()
            data["mmaudio_steps"] = self.mmaudio_steps.get().strip()
            data["cfg_strength"] = self.cfg_strength.get().strip()
            data["flow_shift"] = self.flow_shift.get().strip()
            data["batch_size"] = self.batch_size.get().strip()
            data["infer_steps"] = self.infer_steps.get().strip()
            data["flow_solver"] = self.flow_solver.get().strip()
            data["vae_tiling"] = self.vae_tiling.get()
            data["flow_reverse"] = self.flow_reverse.get()

            # Parse all range or single value inputs
            param_names = ["video_length", "num_videos", "seed", "cfg_scale", "embedded_cfg_scale",
                          "ulysses_degree", "ring_degree", "mmaudio_steps", "cfg_strength",
                          "flow_shift", "batch_size", "infer_steps"]
            parsed_args = {}
            for param in param_names:
                value = data[param]
                if param in ["cfg_scale", "cfg_strength"]:
                    parsed_args[param] = parse_range(value, float)
                else:
                    parsed_args[param] = parse_range(value, int)

            # Fixed parameters
            parsed_args["model"] = data["model"]
            parsed_args["precision"] = data["precision"]
            parsed_args["save_path"] = data["save_path"]
            parsed_args["save_path_suffix"] = data["save_path_suffix"]
            parsed_args["name_suffix"] = data["name_suffix"]
            parsed_args["video_size"] = data["video_size"]
            parsed_args["neg_prompt"] = data["neg_prompt"]
            parsed_args["audio_prompt"] = data["audio_prompt"]
            parsed_args["audio_neg_prompt"] = data["audio_neg_prompt"]
            parsed_args["reproduce"] = data["reproduce"]
            parsed_args["flow_reverse"] = data["flow_reverse"]
            parsed_args["flow_solver"] = data["flow_solver"]
            parsed_args["vae_tiling"] = data["vae_tiling"]

            self.result = parsed_args
            self.destroy()
        except ValueError as ve:
            logging.error(f"Invalid input: {ve}")
            messagebox.showerror("Invalid Input", f"Please ensure all inputs are correct.\nError: {ve}")

    def on_cancel(self):
        self.destroy()

def generate_video(args: dict, prompt: str, negative_prompt: str) -> str:
    """
    Generates a video using sample_video.py with the given prompts and arguments.
    Returns the path to the generated video.
    """
    resolution_str = args["video_size"].split(" ")[0]
    width_str, height_str = resolution_str.split("x")
    width, height = int(width_str), int(height_str)

    enriched_prompt = inject_resolution_terms(prompt)
    enriched_prompt = inject_camera_terms(enriched_prompt)

    logging.debug(f"Enriched Positive Prompt: {enriched_prompt}")
    logging.debug(f"Negative Prompt: {negative_prompt}")

    # Use the passed 'negative_prompt' instead of 'args["neg_prompt"]'
    full_neg_prompt = (
        negative_prompt +
        ", Camera, Lens, Tripod, deformed limbs, Bad Focus, Bad Framing, Missing limbs, Glitching, Watermark, Transforming, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language, transition"
    )

    logging.info(f"Constructed full_neg_prompt: {full_neg_prompt}")

    # Define a temporary filename to prevent overwriting
    temp_video_name = f"temp_video_{random.randint(1000,9999)}.mp4"
    temp_video_path = os.path.join(args["save_path"], temp_video_name)

    cmd = [
        "python3", "sample_video.py",
        "--model", args["model"],
        "--precision", args["precision"],
        "--flow-shift", str(args["flow_shift"]),
        "--flow-solver", args["flow_solver"],
        "--batch-size", str(args["batch_size"]),
        "--infer-steps", str(args["infer_steps"]),
        "--save-path", args["save_path"],
        "--num-videos", "1",  # Generate one video at a time
        "--video-size", str(width), str(height),
        "--video-length", str(args["video_length"]),
        "--prompt", enriched_prompt,
        "--seed", str(args["seed"]),
        "--neg-prompt", full_neg_prompt,
        "--cfg-scale", str(args["cfg_scale"]),
        "--embedded-cfg-scale", str(args["embedded_cfg_scale"]),
        "--ulysses-degree", str(args["ulysses_degree"]),
        "--ring-degree", str(args["ring_degree"])
        # Removed '--output' argument
    ]

    if args["vae_tiling"]:
        cmd.append("--vae-tiling")
    if args["flow_reverse"]:
        cmd.append("--flow-reverse")
    if args["save_path_suffix"]:
        cmd.extend(["--save-path-suffix", args["save_path_suffix"]])
    if args["name_suffix"]:
        cmd.extend(["--name-suffix", args["name_suffix"]])
    if args["reproduce"]:
        cmd.append("--reproduce")

    logging.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, text=True)

    # After generation, identify the newly created video
    generated_files_after = set(os.listdir(args["save_path"]))
    # Assuming sample_video.py saves the video with a specific naming convention,
    # we'll find the latest .mp4 file in the save_path
    mp4_files = [f for f in generated_files_after if f.endswith(".mp4")]
    if not mp4_files:
        raise FileNotFoundError("No generated .mp4 file found.")

    # Get the latest modified .mp4 file
    latest_mp4 = max(
        mp4_files,
        key=lambda x: os.path.getmtime(os.path.join(args["save_path"], x))
    )
    latest_mp4_path = os.path.join(args["save_path"], latest_mp4)

    logging.info(f"Generated video at: {latest_mp4_path}")
    return latest_mp4_path

def main():
    root = tk.Tk()
    apply_theme(root)  # Apply theme to the main window
    root.withdraw()

    prompt_file = select_prompt_file()
    if not prompt_file:
        root.destroy()
        return

    # Derive the directory from the prompt_file path
    prompt_dir = os.path.dirname(prompt_file)
    # Set save_path to the prompt directory automatically
    DEFAULTS["save_path"] = prompt_dir

    args = get_selected_args()
    args["save_path"] = prompt_dir

    logging.info(f"Configuration Arguments: {args}")

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = parse_prompt_file(lines)

    except Exception as e:
        logging.error(f"Error reading prompt files: {e}")
        messagebox.showerror("File Read Error", f"Error reading prompt files:\n{e}")
        root.destroy()
        return

    if not prompts:
        logging.info("No valid video prompts found in the selected file.")
        messagebox.showinfo("No Prompts", "No valid video prompts found.")
        root.destroy()
        return

    # Generate all combinations of parameter ranges
    param_names = ["video_length", "num_videos", "seed", "cfg_scale", "embedded_cfg_scale",
                  "ulysses_degree", "ring_degree", "mmaudio_steps", "cfg_strength",
                  "flow_shift", "batch_size", "infer_steps"]

    # Create a list of lists for each parameter
    param_values = [args[param] for param in param_names]

    # Compute the cartesian product of all parameter ranges
    try:
        all_param_combinations = list(itertools.product(*param_values))
    except Exception as e:
        logging.error(f"Error creating parameter combinations: {e}")
        messagebox.showerror("Parameter Combination Error", f"Error creating parameter combinations:\n{e}")
        root.destroy()
        return

    logging.info(f"Total parameter combinations: {len(all_param_combinations)}")

    # Limit the total number of videos to prevent excessive generation
    MAX_TOTAL_VIDEOS = 999999  # Adjust as needed
    total_prompts = len(prompts)
    total_combinations = len(all_param_combinations)
    total_videos = total_prompts * total_combinations

    if total_videos > MAX_TOTAL_VIDEOS:
        logging.warning(f"Total videos to generate ({total_videos}) exceed the maximum allowed ({MAX_TOTAL_VIDEOS}).")
        messagebox.showerror("Too Many Videos", f"Total videos to generate ({total_videos}) exceed the maximum allowed ({MAX_TOTAL_VIDEOS}). Please narrow down your ranges.")
        root.destroy()
        return

    logging.info(f"Using seed value(s): {args['seed']}")

    os.makedirs(args["save_path"], exist_ok=True)

    # Determine the starting video counter based on existing files
    existing_videos = [f for f in os.listdir(args["save_path"]) if f.startswith("video_") and f.endswith(".mp4")]
    video_counter = 1
    if existing_videos:
        existing_numbers = []
        for f in existing_videos:
            match = re.match(r"video_(\d+)_", f)
            if match:
                existing_numbers.append(int(match.group(1)))
        if existing_numbers:
            video_counter = max(existing_numbers) + 1

    for prompt_idx, prompt_data in enumerate(prompts, start=1):
        positive_prompt = prompt_data.get("positive")
        negative_prompt = prompt_data.get("negative")

        if not positive_prompt or not negative_prompt:
            logging.warning(f"Skipping prompt {prompt_idx}: Incomplete 'positive' or 'negative' sections in video prompts.")
            continue

        for combination in all_param_combinations:
            # Map parameter names to their current values
            current_args = args.copy()
            for i, param in enumerate(param_names):
                current_args[param] = combination[i]

            final_positive = positive_prompt
            final_negative = negative_prompt

            # Handle Audio Prompts
            audio_prompt = current_args.get("audio_prompt", "").strip()
            audio_neg_prompt = current_args.get("audio_neg_prompt", "").strip()

            logging.info(f"Prompt {prompt_idx}: Audio Prompt: '{audio_prompt}'")
            logging.info(f"Prompt {prompt_idx}: Audio Negative Prompt: '{audio_neg_prompt}'")

            five_word_summary = ' '.join(final_positive.split()[:5]) if final_positive else "summary"
            safe_summary = sanitize_filename(five_word_summary)[:20]
            if not safe_summary:
                safe_summary = f"summary_{video_counter}"

            # Infer decade from prompt
            decade = get_decade_from_prompt(final_positive)

            # Construct parameter suffix
            param_suffix = "_".join([f"{key}{value}" for key, value in zip(param_names, combination)])

            # Final filename as per required convention
            final_mp4_name = f"video_{video_counter}_{param_suffix}_Set_in_{decade}_shot_on.mp4"

            logging.info(f"\nGenerating video {video_counter}/{total_videos}:")
            logging.info(f"Video Positive Prompt: {final_positive}")
            logging.info(f"Video Negative Prompt: {final_negative}")
            logging.info(f"5-Word Summary: {five_word_summary}")
            logging.info(f"Final Intended Filename: {final_mp4_name}")

            try:
                # Generate video
                generated_video_path = generate_video(current_args, final_positive, final_negative)

                # Run MMAudio to add audio
                mmaudio_dir = "/home/solomon/TemporalPromptEngine/WINDOWS/VideoGeneratorUtilities/HunyuanVideo/MMAudio"
                original_dir = os.getcwd()
                try:
                    os.chdir(mmaudio_dir)
                    # Prepare prompts for MMAudio
                    # Use user-provided audio prompts directly
                    logging.info(f"MMAudio Prompt: '{audio_prompt}'")
                    logging.info(f"MMAudio Negative Prompt: '{audio_neg_prompt}'")

                    # Define output directory for MMAudio
                    audio_output_dir = os.path.join(current_args["save_path"], "audio_output")
                    os.makedirs(audio_output_dir, exist_ok=True)

                    cmd = [
                        "conda", "run", "-n", "mmaudio", "python", "demo.py",
                        "--duration", "8",
                        "--video", generated_video_path,
                        "--prompt", audio_prompt,
                        "--negative_prompt", audio_neg_prompt,
                        "--output", audio_output_dir,
                        "--cfg_strength", str(current_args["cfg_strength"]),
                        "--num_steps", str(current_args["mmaudio_steps"])
                    ]
                    logging.info(f"Executing MMAudio command: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"MMAudio command failed: {e}")
                    messagebox.showerror("MMAudio Error", f"Error during MMAudio processing:\n{e}")
                    # Remove generated video if MMAudio fails
                    os.remove(generated_video_path)
                    continue
                finally:
                    os.chdir(original_dir)

                # After MMAudio processing, retrieve the final video and audio
                final_video_candidates = [f for f in os.listdir(audio_output_dir) if f.endswith(".mp4")]
                final_audio_candidates = [f for f in os.listdir(audio_output_dir) if f.endswith(".flac")]

                if final_video_candidates:
                    # Assume the latest file is the one just created
                    final_video_candidates.sort(key=lambda x: os.path.getmtime(os.path.join(audio_output_dir, x)), reverse=True)
                    final_video = final_video_candidates[0]
                    final_video_path = os.path.join(audio_output_dir, final_video)
                else:
                    logging.error(f"No final video found in {audio_output_dir}")
                    messagebox.showerror("MMAudio Error", f"No final video found in {audio_output_dir}")
                    # Remove generated video
                    os.remove(generated_video_path)
                    continue

                # Rename the final video to the comprehensive name
                new_final_video_path = os.path.join(current_args["save_path"], final_mp4_name)
                os.rename(final_video_path, new_final_video_path)
                logging.info(f"Renamed final video to: {new_final_video_path}")

                # Create SRT file
                fps = 8
                duration_seconds = int(current_args["video_length"]) / fps
                create_srt_file(new_final_video_path, final_positive, duration_seconds)

                # Delete the generated video from sample_video.py if it's different
                if generated_video_path != final_video_path:
                    os.remove(generated_video_path)
                    logging.info(f"Deleted intermediate video: {generated_video_path}")

                # Clean up audio files if necessary
                # (Optional) If you don't need the audio separately, you can delete the .flac file
                if final_audio_candidates:
                    final_audio_candidates.sort(key=lambda x: os.path.getmtime(os.path.join(audio_output_dir, x)), reverse=True)
                    final_audio = final_audio_candidates[0]
                    final_audio_path = os.path.join(audio_output_dir, final_audio)
                    # Optionally rename or move the audio file as needed
                    # For this example, we'll keep it as is
                else:
                    logging.warning(f"No final audio found in {audio_output_dir}")

                # Create a text file with prompt info
                final_txt_path = os.path.join(current_args["save_path"], f"video_{video_counter}_info.txt")
                with open(final_txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(
                        f"Video Positive Prompt:\n{final_positive}\n\n"
                        f"Video Negative Prompt:\n{final_negative}\n\n"
                        f"Audio Prompt:\n{audio_prompt}\n\n"
                        f"Audio Negative Prompt:\n{audio_neg_prompt}\n\n"
                        f"Final Video Path:\n{new_final_video_path}\n"
                    )
                logging.info(f"Text info saved to: {final_txt_path}")

            except Exception as e:
                logging.error(f"Error generating video for prompt '{final_positive}': {e}")
                messagebox.showerror("Video Generation Error", f"Error generating video:\n{e}")
                continue

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            video_counter += 1

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logging.info("\nAll videos have been generated successfully.")
    messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")
    root.destroy()

if __name__ == "__main__":
    main()
