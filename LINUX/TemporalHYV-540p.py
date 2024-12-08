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

# Summarization settings
TOKENIZER_NAME = "gpt2"  # Tokenizer compatible with your summarization model
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Summarization model
POSITIVE_MAX_TOKENS = 210  # Max tokens for positive prompts
NEGATIVE_MAX_TOKENS = 60   # Max tokens for negative prompts
POSITIVE_MIN_TOKENS = 80   # Min tokens for positive prompts
NEGATIVE_MIN_TOKENS = 30   # Min tokens for negative prompts

# Default parameters for new video generation approach (HunyuanVideo)
VIDEO_SIZE = (720, 1280)   # recommended
VIDEO_LENGTH = 129
INFER_STEPS = 50
EMBEDDED_CFG_SCALE = 6.0
FLOW_SHIFT = 7.0
FLOW_REVERSE = True
USE_CPU_OFFLOAD = True
SEED = 1990  # Default seed value

# --------------------- Initialization ---------------------

# Initialize the tokenizer and summarization pipeline globally
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
    """
    Summarizes the input text to fit within the specified token limit.

    Parameters:
    - text (str): The input text to process.
    - max_tokens (int): The maximum number of tokens allowed.
    - min_tokens (int): The minimum number of tokens required.

    Returns:
    - str: The summarized or truncated text.
    """
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
        return text[:max_tokens]  # Fallback: truncate the text

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes the filename by removing or replacing illegal characters.

    Parameters:
    - filename (str): The filename to sanitize.

    Returns:
    - str: The sanitized filename.
    """
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def parse_prompt_file(lines: list) -> list:
    """
    Parses the prompt file lines into a list of prompt dictionaries.

    Parameters:
    - lines (list): List of lines from the prompt file.

    Returns:
    - list: A list of dictionaries with 'positive' and 'negative' prompts.
    """
    prompts = []
    current_prompt = {}
    current_section = None  # Can be 'positive', 'negative', or None

    for idx, line in enumerate(lines, start=1):
        stripped_line = line.strip()

        if not stripped_line:
            continue  # Skip empty lines

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
                # Append the line to the current section
                current_prompt[current_section] += " " + stripped_line
            else:
                print(f"Warning: Unrecognized line format at line {idx}: '{stripped_line}'. Skipping.")

    # Handle the last prompt if file doesn't end with delimiter
    if "positive" in current_prompt and "negative" in current_prompt:
        prompts.append(current_prompt)
    elif "positive" in current_prompt:
        print("Warning: Last prompt missing 'negative:' section. Skipping.")

    return prompts

def parse_range_list(s: str, type_func):
    """
    Parses a string containing numbers and ranges into a list of numbers.

    Parameters:
    - s (str): The input string.
    - type_func: The type to which the numbers are converted (int or float).

    Returns:
    - list: A list of numbers.
    """
    result = []
    items = [item.strip() for item in s.split(',') if item.strip()]
    for item in items:
        if '-' in item:
            # Range detected
            if ':' in item:
                # Range with step, format is start - end : step
                range_part, step_part = item.split(':')
                start_str, end_str = range_part.split('-')
                start = type_func(start_str.strip())
                end = type_func(end_str.strip())
                step = type_func(step_part.strip())
            else:
                # Range without step
                start_str, end_str = item.split('-')
                start = type_func(start_str.strip())
                end = type_func(end_str.strip())
                step = type_func(1) if type_func is int else type_func(0.1)
            if step == 0:
                raise ValueError(f"Step cannot be zero in '{item}'")
            elif (end - start) * step < 0:
                raise ValueError(f"Step does not move towards end in '{item}'")
            current = start
            values = []
            if step > 0:
                while current <= end + 1e-8:
                    values.append(current)
                    current += step
            else:
                while current >= end - 1e-8:
                    values.append(current)
                    current += step
            if type_func is float:
                values = [round(v, 8) for v in values]
            result.extend(values)
        else:
            value = type_func(item)
            result.append(value)
    return result

def get_parameters() -> Optional[tuple]:
    """
    Opens a popup dialog to get parameters like guidance scales and inference steps.
    Just keep this for compatibility, even though the new model doesn't use them directly now.
    Returns None, None if canceled.
    """
    root = tk.Tk()
    root.withdraw()

    guidance_scales_str = simpledialog.askstring(
        "Input Guidance Scales",
        "Enter Guidance Scales (separated by commas or ranges), not used now, press OK anyway:",
        parent=root
    )
    if guidance_scales_str is None:
        messagebox.showwarning("Input Cancelled", "No guidance scales were provided. Exiting.")
        root.destroy()
        return None, None

    inference_steps_str = simpledialog.askstring(
        "Input Inference Steps",
        "Enter Inference Steps (separated by commas or ranges), not used now, press OK anyway:",
        parent=root
    )
    if inference_steps_str is None:
        messagebox.showwarning("Input Cancelled", "No inference steps were provided. Exiting.")
        root.destroy()
        return None, None

    root.destroy()

    # Parse the input strings into lists of numbers
    try:
        guidance_scales = parse_range_list(guidance_scales_str, float)
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid guidance scales input:\n{e}")
        return None, None

    try:
        inference_steps = parse_range_list(inference_steps_str, int)
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid inference steps input:\n{e}")
        return None, None

    return guidance_scales, inference_steps

def get_seed() -> int:
    """
    Opens a popup dialog to get the seed value.
    Includes a secondary button to randomize the seed.

    Returns:
    - int: The seed value.
    """
    class SeedDialog(simpledialog.Dialog):
        def body(self, master):
            self.title("Input Seed Value")
            tk.Label(master, text="Enter Seed Value:").grid(row=0, column=0, padx=5, pady=5)
            self.seed_entry = tk.Entry(master)
            self.seed_entry.grid(row=0, column=1, padx=5, pady=5)
            self.seed_entry.insert(0, str(SEED))  # Pre-fill with default SEED value

            self.randomize_button = tk.Button(master, text="Randomize", command=self.randomize_seed)
            self.randomize_button.grid(row=0, column=2, padx=5, pady=5)

            return self.seed_entry  # initial focus

        def randomize_seed(self):
            random_seed = random.randint(0, 2**32 - 1)
            self.seed_entry.delete(0, tk.END)
            self.seed_entry.insert(0, str(random_seed))

        def apply(self):
            try:
                self.result = int(self.seed_entry.get())
            except ValueError:
                self.result = None
                messagebox.showerror("Input Error", "Please enter a valid integer for the seed.")

    root = tk.Tk()
    root.withdraw()

    dialog = SeedDialog(root)
    seed_value = dialog.result
    root.destroy()

    if seed_value is None:
        messagebox.showwarning("No Seed Provided", "No seed value was provided. Using default seed.")
        seed_value = SEED

    return seed_value

def select_prompt_file() -> Optional[str]:
    """
    Opens a file dialog to select a prompt list file.
    Returns the path to the selected file or None if canceled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Prompt List File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not file_path:
        messagebox.showwarning("No File Selected", "No prompt list file was selected. Exiting.")
        return None
    return file_path

def create_srt_file(video_path: str, subtitle_text: str, duration: float):
    """
    Creates an .srt subtitle file for the given video.

    Parameters:
    - video_path (str): The path to the video file.
    - subtitle_text (str): The text to be included in the subtitle.
    - duration (float): Duration of the video in seconds.
    """
    try:
        # Define SRT filename
        base, _ = os.path.splitext(video_path)
        srt_path = f"{base}.srt"

        # Define subtitle timing
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
    Generates a video using the new model (HunyuanVideo) and configuration.
    Uses the provided prompt and parameters to call 'sample_video.py' via subprocess.

    The provided `negative_prompt` is not directly used since the new script doesn't support it.
    If necessary, you could incorporate negative concepts into `prompt`.
    """
    full_prompt = prompt  # Not currently combining negative_prompt, as there's no direct support.

    os.makedirs(output_path, exist_ok=True)
    # We'll name the file "Video.mp4" inside output_path, or base it on prompt index if needed.
    # The `sample_video.py` script places the generated video into `save-path`.
    # We'll rely on it saving in `output_path`.

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

    if flow_reverse:
        cmd.append("--flow-reverse")
    if use_cpu_offload:
        cmd.append("--use-cpu-offload")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    subprocess.run(cmd, check=True, text=True)

    # The script presumably saves a file in `output_path`.
    # We'll assume it saves a .mp4 file. Let's find it and create an SRT.
    # We'll assume there's a single .mp4 file generated.
    for filename in os.listdir(output_path):
        if filename.endswith(".mp4"):
            video_file_path = os.path.join(output_path, filename)
            # Duration based on config: we know video_length=129 frames, fps is likely fixed.
            # Original instructions used fps=8, but we don't have that info here.
            # We'll guess fps ~8 (or you can define it if needed).
            # If fps isn't given, we can't accurately create the SRT timing.
            # We'll assume 8 fps for consistency:
            fps = 8
            duration_seconds = video_length / fps
            create_srt_file(video_file_path, prompt, duration_seconds)
            break

def main():
    global SEED  # Use the global SEED variable

    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Mode selection no longer relevant, just skip it
    # We'll just generate videos using the new model.

    prompt_file = select_prompt_file()
    if not prompt_file:
        root.destroy()
        return

    parameters = get_parameters()
    if not parameters:
        # User canceled or provided invalid input
        # These parameters are not needed for the new model, but we asked anyway.
        guidance_scales = [6.0]  # default
        inference_steps = [50]
    else:
        guidance_scales, inference_steps = parameters
        # Not used in the new model directly. Just keep defaults.

    SEED = get_seed()
    print(f"Using seed value: {SEED}")

    # Determine the directory of the prompt file for saving videos
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

    # Iterate through each prompt and generate videos
    for idx, prompt_data in enumerate(prompts, start=1):
        positive_prompt = prompt_data.get("positive")
        negative_prompt = prompt_data.get("negative")

        if not positive_prompt or not negative_prompt:
            print(f"Skipping prompt {idx}: Incomplete 'positive' or 'negative' sections.")
            continue

        # Summarize the positive and negative prompts separately
        summarized_positive = summarize_text(positive_prompt, max_tokens=POSITIVE_MAX_TOKENS, min_tokens=POSITIVE_MIN_TOKENS)
        summarized_negative = summarize_text(negative_prompt, max_tokens=NEGATIVE_MAX_TOKENS, min_tokens=NEGATIVE_MIN_TOKENS)

        if summarized_positive != positive_prompt:
            print("Positive prompt was too long and has been summarized.")
        if summarized_negative != negative_prompt:
            print("Negative prompt was too long and has been summarized.")

        # Generate a 5-word summary for the filename
        five_word_summary = ' '.join(summarized_positive.split()[:5]) if summarized_positive else "summary"
        safe_summary = sanitize_filename(five_word_summary)[:20]
        if not safe_summary:
            safe_summary = f"summary_{idx}"

        print(f"\nGenerating video for prompt {idx}/{len(prompts)}:")
        print(f"Positive Prompt: {summarized_positive}")
        print(f"Negative Prompt: {summarized_negative}")
        print(f"5-Word Summary: {five_word_summary}")

        # Create a sub-directory for each video to avoid overwriting
        video_output_dir = os.path.join(output_dir, f"Video_{idx}_{safe_summary}")
        os.makedirs(video_output_dir, exist_ok=True)

        # Generate the video using new script
        try:
            generate_video(
                prompt=summarized_positive,
                negative_prompt=summarized_negative,
                output_path=video_output_dir,
                video_size=VIDEO_SIZE,
                video_length=VIDEO_LENGTH,
                infer_steps=INFER_STEPS,
                embedded_cfg_scale=EMBEDDED_CFG_SCALE,
                flow_shift=FLOW_SHIFT,
                flow_reverse=FLOW_REVERSE,
                use_cpu_offload=USE_CPU_OFFLOAD,
                seed=SEED
            )
        except Exception as e:
            print(f"Error generating video for prompt '{summarized_positive}': {e}")
            messagebox.showerror("Video Generation Error", f"Error generating video for prompt:\n{e}")
            continue

        # Clear memory after each video generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Clear memory after all videos are generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\nAll videos have been generated successfully.")
    messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")
    root.destroy()

if __name__ == "__main__":
    main()
