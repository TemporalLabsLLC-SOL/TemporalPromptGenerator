import os
import random
import time
import gc
import subprocess
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ExifTags, Image
from datetime import datetime
from pathlib import Path
from loguru import logger

# --------------------- Imported from previous code sections ---------------------
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

# --------------------- Configuration ---------------------

# Default parameters for new video generation approach (HunyuanVideo)
DEFAULT_INFER_STEPS = 50
DEFAULT_EMBEDDED_CFG_SCALE = 6.0
DEFAULT_FLOW_SHIFT = 7.0
DEFAULT_FLOW_REVERSE = True
DEFAULT_USE_CPU_OFFLOAD = True
DEFAULT_SEED = 1990

# Expanded resolution presets including "official" and additional cinematic/history-inspired ones
RESOLUTION_PRESETS = {
    "720p 9:16": ((720, 1280), 129, "9:16 (vertical)"),
    "720p 16:9": ((1280, 720), 129, "16:9 (widescreen)"),
    "720p 4:3": ((1104, 832), 129, "4:3"),
    "720p 3:4": ((832, 1104), 129, "3:4"),
    "720p 1:1": ((960, 960), 129, "1:1 (square)"),

    "540p 9:16": ((544, 960), 129, "9:16"),
    "540p 16:9": ((960, 544), 129, "16:9"),
    "540p 4:3": ((624, 832), 129, "4:3"),
    "540p 3:4": ((832, 624), 129, "3:4"),
    "540p 1:1": ((720, 720), 129, "1:1"),

    # Additional resolutions
    "480p 16:9": ((480, 854), 129, "16:9"),
    "1080p 16:9": ((1080, 1920), 129, "16:9"),
    "1080p 9:16": ((1080, 1920), 129, "9:16 vertical"),
    "1080p 4:3": ((1080, 1440), 129, "4:3"),
    "1080p 1:1": ((1080, 1080), 129, "1:1"),

    # 4K
    "4K 16:9": ((2160, 3840), 129, "16:9"),
    "4K 9:16": ((2160, 3840), 129, "9:16"),
    "4K 4:3": ((2160, 2880), 129, "4:3"),
    "4K 1:1": ((2160, 2160), 129, "1:1"),

    # Cinematic aspect ratios
    "1080p Cinemascope(2.39:1)": ((1080, 2582), 129, "2.39:1 Cinemascope"),
    "720p Cinemascope(2.39:1)": ((720, 1720), 129, "2.39:1 Cinemascope"),
}

# --------------------- Helper Functions ---------------------

def sanitize_filename(filename: str) -> str:
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

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
                print(f"Warning: New 'positive:' found before completing previous prompt at line {idx}.")
                current_prompt = {}
            current_prompt["positive"] = stripped_line[len("positive:"):].strip()
            current_section = "positive"
        elif stripped_line.startswith("negative:"):
            if "positive" not in current_prompt:
                print(f"Warning: 'negative:' section without a preceding 'positive:' at line {idx}. Skipping.")
                current_section = None
                continue
            current_prompt["negative"] = stripped_line[len("negative:"):].strip()
            current_section = "negative"
        elif set(stripped_line) == set("-"):
            if "positive" in current_prompt and "negative" in current_prompt:
                prompts.append(current_prompt)
            else:
                if "positive" in current_prompt:
                    print(f"Warning: 'negative:' section missing before line {idx}. Skipping.")
            current_prompt = {}
            current_section = None
        else:
            if current_section and current_section in current_prompt:
                current_prompt[current_section] += " " + stripped_line
            else:
                print(f"Warning: Unrecognized line at {idx}: '{stripped_line}'. Skipping.")

    if "positive" in current_prompt and "negative" in current_prompt:
        prompts.append(current_prompt)
    elif "positive" in current_prompt:
        print("Warning: Last prompt missing 'negative:' section. Skipping.")

    return prompts

def select_prompt_file() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
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
        print(f"SRT file saved to: {srt_path}")
    except Exception as e:
        print(f"Error creating SRT file: {e}")
        messagebox.showerror("SRT Generation Error", f"Error creating SRT file:\n{e}")

class ResolutionDialog(tk.Toplevel):
    def __init__(self, master, default_seed, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Select Video Settings")

        tk.Label(self, text="Select Resolution Preset:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.res_var = tk.StringVar(value="720p 9:16")
        presets = list(RESOLUTION_PRESETS.keys())
        self.res_combobox = ttk.Combobox(self, textvariable=self.res_var, values=presets, state="readonly")
        self.res_combobox.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="Inference Steps:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.infer_steps_var = tk.StringVar(value=str(DEFAULT_INFER_STEPS))
        tk.Entry(self, textvariable=self.infer_steps_var).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="Embedded CFG Scale:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.cfg_scale_var = tk.StringVar(value=str(DEFAULT_EMBEDDED_CFG_SCALE))
        tk.Entry(self, textvariable=self.cfg_scale_var).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="Flow Shift:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.flow_shift_var = tk.StringVar(value=str(DEFAULT_FLOW_SHIFT))
        tk.Entry(self, textvariable=self.flow_shift_var).grid(row=3, column=1, padx=5, pady=5)

        self.flow_reverse_var = tk.BooleanVar(value=DEFAULT_FLOW_REVERSE)
        tk.Checkbutton(self, text="Flow Reverse", variable=self.flow_reverse_var).grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.cpu_offload_var = tk.BooleanVar(value=DEFAULT_USE_CPU_OFFLOAD)
        tk.Checkbutton(self, text="Use CPU Offload", variable=self.cpu_offload_var).grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        tk.Label(self, text="Seed:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.seed_var = tk.StringVar(value=str(default_seed))
        tk.Entry(self, textvariable=self.seed_var).grid(row=6, column=1, padx=5, pady=5)
        self.randomize_button = tk.Button(self, text="Randomize", command=self.randomize_seed)
        self.randomize_button.grid(row=6, column=2, padx=5, pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=10)
        tk.Button(btn_frame, text="OK", command=self.on_ok).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side="right", padx=5)

        self.result = None

    def randomize_seed(self):
        random_seed = random.randint(0, 2**32 - 1)
        self.seed_var.set(str(random_seed))

    def on_ok(self):
        try:
            chosen_res = self.res_var.get()
            if chosen_res not in RESOLUTION_PRESETS:
                raise ValueError("Invalid resolution preset selected.")
            (h, w), vid_length, desc = RESOLUTION_PRESETS[chosen_res]

            infer_steps = int(self.infer_steps_var.get())
            cfg_scale = float(self.cfg_scale_var.get())
            flow_shift = float(self.flow_shift_var.get())
            flow_reverse = self.flow_reverse_var.get()
            use_cpu_offload = self.cpu_offload_var.get()
            seed = int(self.seed_var.get())

            self.result = {
                "video_size": (h, w),
                "video_length": vid_length,
                "infer_steps": infer_steps,
                "embedded_cfg_scale": cfg_scale,
                "flow_shift": flow_shift,
                "flow_reverse": flow_reverse,
                "use_cpu_offload": use_cpu_offload,
                "seed": seed
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("Input Error", f"Error parsing inputs:\n{e}")

    def on_cancel(self):
        self.result = None
        self.destroy()

def get_video_config(default_seed=DEFAULT_SEED):
    root = tk.Tk()
    root.withdraw()
    dialog = ResolutionDialog(root, default_seed)
    root.wait_window(dialog)
    root.destroy()
    return dialog.result

def main():
    global DEFAULT_SEED

    root = tk.Tk()
    root.withdraw()

    prompt_file = select_prompt_file()
    if not prompt_file:
        root.destroy()
        return

    video_config = get_video_config(DEFAULT_SEED)
    if not video_config:
        messagebox.showwarning("No Configuration", "No configuration provided. Exiting.")
        root.destroy()
        return

    DEFAULT_SEED = video_config["seed"]

    # Read and parse the prompts
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        prompts = parse_prompt_file(lines)
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        messagebox.showerror("File Read Error", f"Error reading prompt file:\n{e}")
        root.destroy()
        return

    if not prompts:
        print("No valid prompts found in the selected file.")
        messagebox.showinfo("No Prompts", "No valid prompts found in the selected file.")
        root.destroy()
        return

    output_dir = os.path.dirname(prompt_file)
    if not output_dir:
        output_dir = os.getcwd()

    # Extract parameters
    video_size = video_config["video_size"]
    video_length = video_config["video_length"]
    infer_steps = video_config["infer_steps"]
    embedded_cfg_scale = video_config["embedded_cfg_scale"]
    flow_shift = video_config["flow_shift"]
    flow_reverse = video_config["flow_reverse"]
    use_cpu_offload = video_config["use_cpu_offload"]
    seed = video_config["seed"]

    # -----------------------------------
    # Load HunyuanVideoSampler
    # The args can be constructed to point to model_base, etc.
    # Adjust these as per your environment or pass CLI arguments if parse_args is needed.
    # If you need to provide certain arguments (like model_base, save_path etc.) you may specify them here:
    class SimpleArgs:
        # Provide any necessary defaults here or integrate parse_args if you have a config file
        model_base = "ckpts"   # Adjust this path as needed
        save_path = output_dir
        save_path_suffix = ""
        prompt = ""  # Will be replaced per prompt below
        neg_prompt = None
        cfg_scale = 1.0
        num_videos = 1
        batch_size = 1

    args = SimpleArgs()

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        messagebox.showerror("Error", f"`models_root` not exists: {models_root_path}")
        root.destroy()
        return

    # Initialize sampler once
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # get updated args
    args = hunyuan_video_sampler.args

    for idx, prompt_data in enumerate(prompts, start=1):
        positive_prompt = prompt_data.get("positive")
        negative_prompt = prompt_data.get("negative")

        if not positive_prompt or not negative_prompt:
            print(f"Skipping prompt {idx}: Incomplete 'positive' or 'negative' sections.")
            continue

        five_word_summary = ' '.join(positive_prompt.split()[:5]) if positive_prompt else "summary"
        safe_summary = sanitize_filename(five_word_summary)[:20]
        if not safe_summary:
            safe_summary = f"summary_{idx}"

        print(f"\nGenerating video for prompt {idx}/{len(prompts)}:")
        print(f"Positive Prompt: {positive_prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"5-Word Summary: {five_word_summary}")

        video_output_dir = os.path.join(output_dir, f"Video_{idx}_{safe_summary}")
        os.makedirs(video_output_dir, exist_ok=True)

        try:
            # Generate the video frames using hunyuan_video_sampler
            outputs = hunyuan_video_sampler.predict(
                prompt=positive_prompt,
                height=video_size[0],
                width=video_size[1],
                video_length=video_length,
                seed=seed,
                negative_prompt=negative_prompt,
                infer_steps=infer_steps,
                guidance_scale=args.cfg_scale,   # You can adjust or add UI control for cfg_scale if needed
                num_videos_per_prompt=args.num_videos,
                flow_shift=flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=embedded_cfg_scale
            )
            samples = outputs['samples']

            # Save samples
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                video_filename = f"{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                out_path = os.path.join(video_output_dir, video_filename)
                save_videos_grid(sample, out_path, fps=24)
                logger.info(f'Sample save to: {out_path}')

                # Create SRT
                # Assume fps=8 as mentioned previously for durations
                fps = 8
                duration_seconds = video_length / fps
                create_srt_file(out_path, positive_prompt, duration_seconds)

        except Exception as e:
            print(f"Error generating video for prompt '{positive_prompt}': {e}")
            messagebox.showerror("Video Generation Error", f"Error generating video:\n{e}")
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\nAll videos have been generated successfully.")
    messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")
    root.destroy()

if __name__ == "__main__":
    main()
