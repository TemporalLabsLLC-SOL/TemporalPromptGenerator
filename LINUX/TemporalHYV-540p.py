import os
import random
import time
import gc
import subprocess
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import ExifTags, Image
from datetime import datetime
from pathlib import Path
from loguru import logger

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

# --------------------- Configuration ---------------------
DEFAULT_INFER_STEPS = 50
DEFAULT_EMBEDDED_CFG_SCALE = 6.0
DEFAULT_FLOW_SHIFT = 7.0
DEFAULT_FLOW_REVERSE = True
DEFAULT_USE_CPU_OFFLOAD = True
DEFAULT_SEED = 1990

# Official 540p resolution (for example: 540p 16:9)
VIDEO_SIZE = (960, 544)
VIDEO_LENGTH = 129
FPS = 8  # As previously assumed for duration calculation

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

def main():
    root = tk.Tk()
    root.withdraw()

    prompt_file = select_prompt_file()
    if not prompt_file:
        root.destroy()
        return

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

    # Prepare args for the model loading
    class SimpleArgs:
        model_base = "ckpts"   # Adjust if needed
        save_path = output_dir
        save_path_suffix = ""
        prompt = ""
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

    # Load model sampler
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    # Fixed parameters for official 540p
    video_size = VIDEO_SIZE
    video_length = VIDEO_LENGTH
    infer_steps = DEFAULT_INFER_STEPS
    embedded_cfg_scale = DEFAULT_EMBEDDED_CFG_SCALE
    flow_shift = DEFAULT_FLOW_SHIFT
    flow_reverse = DEFAULT_FLOW_REVERSE
    use_cpu_offload = DEFAULT_USE_CPU_OFFLOAD
    seed = DEFAULT_SEED

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
            outputs = hunyuan_video_sampler.predict(
                prompt=positive_prompt,
                height=video_size[0],
                width=video_size[1],
                video_length=video_length,
                seed=seed,
                negative_prompt=negative_prompt,
                infer_steps=infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=embedded_cfg_scale
            )
            samples = outputs['samples']

            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                video_filename = f"{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                out_path = os.path.join(video_output_dir, video_filename)
                save_videos_grid(sample, out_path, fps=24)
                logger.info(f'Sample save to: {out_path}')

                # Create SRT using fps=8 for duration
                duration_seconds = video_length / FPS
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
