import os
import random
import time
import gc
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import ExifTags, Image
import subprocess

# --------------------- Configuration ---------------------
# Default values for arguments (can be overridden by user input)
DEFAULT_VIDEO_SIZE = (720, 1280)   
DEFAULT_VIDEO_LENGTH = 129
DEFAULT_INFER_STEPS = 50
DEFAULT_EMBEDDED_CFG_SCALE = 6.0
DEFAULT_FLOW_SHIFT = 7.0
DEFAULT_FLOW_REVERSE = True
DEFAULT_USE_CPU_OFFLOAD = True
DEFAULT_SEED = 1990
DEFAULT_SAVE_PATH = "./results"

# Summarization settings
TOKENIZER_NAME = "gpt2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
POSITIVE_MAX_TOKENS = 210
NEGATIVE_MAX_TOKENS = 60
POSITIVE_MIN_TOKENS = 80
NEGATIVE_MIN_TOKENS = 30

# --------------------- Initialization ---------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    print(f"Error loading tokenizer '{TOKENIZER_NAME}': {e}")
    tokenizer = None

try:
    summarizer = pipeline(
        "summarization",
        model=SUMMARIZATION_MODEL,
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    print(f"Error loading summarization model '{SUMMARIZATION_MODEL}': {e}")
    summarizer = None

# --------------------- Helper Functions ---------------------
def summarize_text(text: str, max_tokens: int = 220, min_tokens: int = 30) -> str:
    if not tokenizer:
        print("Tokenizer not initialized. Returning original text.")
        return text

    try:
        summary = summarizer(
            text,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False,
            truncation=True,
        )[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text[:max_tokens]  # fallback truncate

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
                print(f"Warning: New 'positive:' found before completing previous prompt at line {idx}. Skipping previous prompt.")
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
                    print(f"Warning: 'negative:' section missing for prompt before line {idx}. Skipping.")
            current_prompt = {}
            current_section = None
        else:
            if current_section and current_section in current_prompt:
                current_prompt[current_section] += " " + stripped_line
            else:
                print(f"Warning: Unrecognized line format at line {idx}: '{stripped_line}'. Skipping.")

    # Handle last prompt if no trailing delimiter
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
        print(f"Error creating SRT file for video '{video_path}': {e}")
        messagebox.showerror("SRT Generation Error", f"Error creating SRT file for video '{video_path}':\n{e}")

def generate_video(
    prompt: str,
    negative_prompt: str,
    output_path: str,
    video_size=(720, 1280),
    video_length=129,
    infer_steps=50,
    embedded_cfg_scale=6.0,
    flow_shift=7.0,
    flow_reverse=True,
    use_cpu_offload=True,
    seed=None,
):
    """
    Generate video using sample_video.py with provided arguments.
    """
    full_prompt = prompt
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "python3", "sample_video.py",
        "--video-size", str(video_size[0]), str(video_size[1]),
        "--video-length", str(video_length),
        "--infer-steps", str(infer_steps),
        "--embedded-cfg-scale", str(embedded_cfg_scale),
        "--flow-shift", str(flow_shift),
        "--prompt", full_prompt,
        "--save-path", output_path
    ]

    if negative_prompt.strip():
        cmd.extend(["--neg-prompt", negative_prompt])

    if flow_reverse:
        cmd.append("--flow-reverse")
    if use_cpu_offload:
        cmd.append("--use-cpu-offload")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    subprocess.run(cmd, check=True, text=True)

    # Assume single .mp4 file is generated
    for filename in os.listdir(output_path):
        if filename.endswith(".mp4"):
            video_file_path = os.path.join(output_path, filename)
            fps = 8  # Assuming fps
            duration_seconds = video_length / fps
            create_srt_file(video_file_path, prompt, duration_seconds)
            break

class ArgDialog(simpledialog.Dialog):
    """
    Dialog to ask the user for various arguments. Provides a brief explanation
    of what each parameter does and its effect on the outcome.
    """
    def body(self, master):
        self.title("Input Generation Arguments")

        # Each label explains the parameter and its effect.
        # Prompt not asked here since it's from the prompt file.
        
        tk.Label(master, text="Video Width (default 720):\nAffects resolution and detail.", justify=tk.LEFT).grid(row=0, column=0, sticky="w")
        self.width_entry = tk.Entry(master)
        self.width_entry.insert(0, str(DEFAULT_VIDEO_SIZE[0]))
        self.width_entry.grid(row=0, column=1)

        tk.Label(master, text="Video Height (default 1280):\nAffects resolution and detail.", justify=tk.LEFT).grid(row=1, column=0, sticky="w")
        self.height_entry = tk.Entry(master)
        self.height_entry.insert(0, str(DEFAULT_VIDEO_SIZE[1]))
        self.height_entry.grid(row=1, column=1)

        tk.Label(master, text="Video Length (default 129 frames):\nNumber of frames. Longer = longer video.", justify=tk.LEFT).grid(row=2, column=0, sticky="w")
        self.length_entry = tk.Entry(master)
        self.length_entry.insert(0, str(DEFAULT_VIDEO_LENGTH))
        self.length_entry.grid(row=2, column=1)

        tk.Label(master, text="Inference Steps (default 50):\nMore steps can mean higher quality but longer generation time.", justify=tk.LEFT).grid(row=3, column=0, sticky="w")
        self.infer_steps_entry = tk.Entry(master)
        self.infer_steps_entry.insert(0, str(DEFAULT_INFER_STEPS))
        self.infer_steps_entry.grid(row=3, column=1)

        tk.Label(master, text="Embedded CFG Scale (default 6.0):\nHigher = more adherence to prompt, can reduce diversity.", justify=tk.LEFT).grid(row=4, column=0, sticky="w")
        self.cfg_scale_entry = tk.Entry(master)
        self.cfg_scale_entry.insert(0, str(DEFAULT_EMBEDDED_CFG_SCALE))
        self.cfg_scale_entry.grid(row=4, column=1)

        tk.Label(master, text="Flow Shift (default 7.0):\nAffects temporal consistency and motion smoothness.", justify=tk.LEFT).grid(row=5, column=0, sticky="w")
        self.flow_shift_entry = tk.Entry(master)
        self.flow_shift_entry.insert(0, str(DEFAULT_FLOW_SHIFT))
        self.flow_shift_entry.grid(row=5, column=1)

        tk.Label(master, text="Flow Reverse (default True):\nIf true, samples from end to start, can affect style.", justify=tk.LEFT).grid(row=6, column=0, sticky="w")
        self.flow_reverse_var = tk.BooleanVar(value=DEFAULT_FLOW_REVERSE)
        self.flow_reverse_check = tk.Checkbutton(master, text="Enable Flow Reverse", variable=self.flow_reverse_var)
        self.flow_reverse_check.grid(row=6, column=1, sticky="w")

        tk.Label(master, text="Use CPU Offload (default True):\nHelps memory usage, might be slower but necessary for high-res.", justify=tk.LEFT).grid(row=7, column=0, sticky="w")
        self.cpu_offload_var = tk.BooleanVar(value=DEFAULT_USE_CPU_OFFLOAD)
        self.cpu_offload_check = tk.Checkbutton(master, text="Enable CPU Offload", variable=self.cpu_offload_var)
        self.cpu_offload_check.grid(row=7, column=1, sticky="w")

        tk.Label(master, text="Seed (default 1990):\nControls randomness. Same seed = repeatable results.", justify=tk.LEFT).grid(row=8, column=0, sticky="w")
        self.seed_entry = tk.Entry(master)
        self.seed_entry.insert(0, str(DEFAULT_SEED))
        self.seed_entry.grid(row=8, column=1)

        tk.Label(master, text="Output Save Path (default ./results):\nDirectory to save the generated video.", justify=tk.LEFT).grid(row=9, column=0, sticky="w")
        self.save_path_entry = tk.Entry(master)
        self.save_path_entry.insert(0, DEFAULT_SAVE_PATH)
        self.save_path_entry.grid(row=9, column=1)

        return self.width_entry

    def apply(self):
        try:
            width = int(self.width_entry.get().strip())
            height = int(self.height_entry.get().strip())
            length = int(self.length_entry.get().strip())
            infer_steps = int(self.infer_steps_entry.get().strip())
            cfg_scale = float(self.cfg_scale_entry.get().strip())
            flow_shift = float(self.flow_shift_entry.get().strip())
            flow_reverse = self.flow_reverse_var.get()
            cpu_offload = self.cpu_offload_var.get()
            seed = int(self.seed_entry.get().strip())
            save_path = self.save_path_entry.get().strip()

            self.result = {
                "video_size": (width, height),
                "video_length": length,
                "infer_steps": infer_steps,
                "embedded_cfg_scale": cfg_scale,
                "flow_shift": flow_shift,
                "flow_reverse": flow_reverse,
                "use_cpu_offload": cpu_offload,
                "seed": seed,
                "save_path": save_path
            }
        except Exception as e:
            messagebox.showerror("Input Error", f"Error parsing input:\n{e}")
            self.result = None

def get_generation_args():
    root = tk.Tk()
    root.withdraw()
    dialog = ArgDialog(root)
    args = dialog.result
    root.destroy()

    if not args:
        # If user canceled or error occurred
        args = {
            "video_size": DEFAULT_VIDEO_SIZE,
            "video_length": DEFAULT_VIDEO_LENGTH,
            "infer_steps": DEFAULT_INFER_STEPS,
            "embedded_cfg_scale": DEFAULT_EMBEDDED_CFG_SCALE,
            "flow_shift": DEFAULT_FLOW_SHIFT,
            "flow_reverse": DEFAULT_FLOW_REVERSE,
            "use_cpu_offload": DEFAULT_USE_CPU_OFFLOAD,
            "seed": DEFAULT_SEED,
            "save_path": DEFAULT_SAVE_PATH
        }
    return args

def main():
    root = tk.Tk()
    root.withdraw()

    prompt_file = select_prompt_file()
    if not prompt_file:
        root.destroy()
        return

    # Get generation arguments via popup
    gen_args = get_generation_args()
    video_size = gen_args["video_size"]
    video_length = gen_args["video_length"]
    infer_steps = gen_args["infer_steps"]
    embedded_cfg_scale = gen_args["embedded_cfg_scale"]
    flow_shift = gen_args["flow_shift"]
    flow_reverse = gen_args["flow_reverse"]
    use_cpu_offload = gen_args["use_cpu_offload"]
    seed = gen_args["seed"]
    save_path = gen_args["save_path"]

    # Set the output directory relative to prompt file if needed
    output_dir = os.path.dirname(prompt_file)
    if not output_dir:
        output_dir = os.getcwd()

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

    print(f"Using seed value: {seed}")

    # Iterate through prompts
    for idx, prompt_data in enumerate(prompts, start=1):
        positive_prompt = prompt_data.get("positive")
        negative_prompt = prompt_data.get("negative")

        if not positive_prompt or not negative_prompt:
            print(f"Skipping prompt {idx}: Incomplete 'positive' or 'negative' sections.")
            continue

        summarized_positive = summarize_text(positive_prompt, max_tokens=POSITIVE_MAX_TOKENS, min_tokens=POSITIVE_MIN_TOKENS)
        summarized_negative = summarize_text(negative_prompt, max_tokens=NEGATIVE_MAX_TOKENS, min_tokens=NEGATIVE_MIN_TOKENS)

        if summarized_positive != positive_prompt:
            print("Positive prompt was too long and has been summarized.")
        if summarized_negative != negative_prompt:
            print("Negative prompt was too long and has been summarized.")

        five_word_summary = ' '.join(summarized_positive.split()[:5]) if summarized_positive else "summary"
        safe_summary = sanitize_filename(five_word_summary)[:20]
        if not safe_summary:
            safe_summary = f"summary_{idx}"

        print(f"\nGenerating video for prompt {idx}/{len(prompts)}:")
        print(f"Positive Prompt: {summarized_positive}")
        print(f"Negative Prompt: {summarized_negative}")
        print(f"5-Word Summary: {five_word_summary}")

        video_output_dir = os.path.join(output_dir, f"Video_{idx}_{safe_summary}")
        os.makedirs(video_output_dir, exist_ok=True)

        # Use the arguments gathered from the dialog
        try:
            generate_video(
                prompt=summarized_positive,
                negative_prompt=summarized_negative,
                output_path=video_output_dir,
                video_size=video_size,
                video_length=video_length,
                infer_steps=infer_steps,
                embedded_cfg_scale=embedded_cfg_scale,
                flow_shift=flow_shift,
                flow_reverse=flow_reverse,
                use_cpu_offload=use_cpu_offload,
                seed=seed,
            )
        except Exception as e:
            print(f"Error generating video for prompt '{summarized_positive}': {e}")
            messagebox.showerror("Video Generation Error", f"Error generating video for prompt:\n{e}")
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
