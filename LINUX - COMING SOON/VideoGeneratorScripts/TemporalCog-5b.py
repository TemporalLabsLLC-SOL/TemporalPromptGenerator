import argparse
import os
import random
import time
import gc
from typing import Optional

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from transformers import AutoTokenizer, pipeline

# --------------------- Configuration ---------------------

# Summarization settings
TOKENIZER_NAME = "gpt2"  # Tokenizer compatible with your summarization model
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Summarization model
POSITIVE_MAX_TOKENS = 210  # Max tokens for positive prompts
NEGATIVE_MAX_TOKENS = 60   # Max tokens for negative prompts
POSITIVE_MIN_TOKENS = 80   # Min tokens for positive prompts
NEGATIVE_MIN_TOKENS = 30   # Min tokens for negative prompts

# Video generation settings
MODEL_PATH_5B = "THUDM/CogVideoX-5b"  # Pre-trained 5b model path
GENERATE_TYPE = "t2v"                   # Generation type: 't2v', 'i2v', 'v2v'
LORA_PATH = None                        # Path to LoRA weights if used
LORA_RANK = 128
GUIDANCE_SCALES = [7.0]                 # Default guidance scales (will be overridden by user input)
INFERENCE_STEPS = [10, 20, 40, 60, 80, 100]  # Default inference steps (will be overridden by user input)
SEED = 1990  # Default seed value
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# --------------------- Initialization ---------------------

# Initialize the tokenizer and summarization pipeline globally to avoid reloading for each prompt
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
    Truncates the input text to fit within the specified token limit without summarizing.

    Parameters:
    - text (str): The input text to process.
    - max_tokens (int): The maximum number of tokens allowed.

    Returns:
    - str: The text truncated to the token limit.
    """
    if not tokenizer:
        print("Tokenizer not initialized. Returning original text.")
        return text

    # Tokenize the input text
    tokens = tokenizer.encode(text, return_tensors="pt")
    num_tokens = tokens.shape[1]

    # If the input is within limits, return as is
    if num_tokens <= max_tokens:
        return text

    # Truncate the text to the maximum token limit
    truncated_text = tokenizer.decode(tokens[0, :max_tokens], skip_special_tokens=True)

    print("Prompt was too long and has been truncated to fit the model's token limit.")
    print(f"Original Prompt Length: {num_tokens} tokens")
    print(f"Truncated Prompt Length: {max_tokens} tokens")

    return truncated_text


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncates the input text to the specified token limit, ensuring that key elements remain intact.

    Parameters:
    - text (str): The text to truncate.
    - max_tokens (int): The token limit.

    Returns:
    - str: The truncated text adhering to the token limit.
    """
    # Tokenize text
    tokens = tokenizer.encode(text, return_tensors="pt")
    num_tokens = tokens.shape[1]

    if num_tokens <= max_tokens:
        return text  # No need to truncate

    # Split text into words and truncate based on token count
    words = text.split()

    # Token limit truncation (even if breaking sentences)
    truncated_words = []
    for word in words:
        truncated_words.append(word)
        truncated_text = ' '.join(truncated_words)
        if tokenizer.encode(truncated_text, return_tensors="pt").shape[1] > max_tokens:
            truncated_words.pop()  # Remove the last word that exceeded the limit
            break

    return ' '.join(truncated_words)

def create_five_word_summary(text: str) -> str:
    """
    Creates a 5-word summary of the given text.

    Parameters:
    - text (str): The text to summarize.

    Returns:
    - str: A 5-word summary.
    """
    if not summarizer:
        print("Summarizer not initialized. Returning first 5 words.")
        return ' '.join(text.split()[:5])

    try:
        # Summarize the text with a low max_length to aim for a short summary
        summary = summarizer(
            text,
            max_length=10,  # Adjust as needed
            min_length=5,
            do_sample=False,
            truncation=True,
        )[0]['summary_text']
        # Extract first 5 words
        five_word_summary = ' '.join(summary.split()[:5])
        return five_word_summary
    except Exception as e:
        print(f"Error during 5-word summarization: {e}")
        return ' '.join(text.split()[:5])

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
        # Convert duration to hours:minutes:seconds,milliseconds
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration - int(duration)) * 1000)
        end_time = f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

        # Create SRT content
        srt_content = f"""1
{start_time} --> {end_time}
{subtitle_text}
"""

        # Write to .srt file
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        print(f"SRT file saved to: {srt_path}")

    except Exception as e:
        print(f"Error creating SRT file for video '{video_path}': {e}")
        messagebox.showerror("SRT Generation Error", f"Error creating SRT file for video '{video_path}':\n{e}")

def select_prompt_file():
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

def parse_prompt_file(lines: list):
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
        print(f"Warning: Last prompt missing 'negative:' section. Skipping.")

    return prompts

def sanitize_filename(filename: str):
    """
    Sanitizes the filename by removing or replacing illegal characters.
    """
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

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
                # Range without step, default step is 1 for ints, 0.1 for floats
                start_str, end_str = item.split('-')
                start = type_func(start_str.strip())
                end = type_func(end_str.strip())
                step = type_func(1) if type_func is int else type_func(0.1)
            # Generate the range
            if step == 0:
                raise ValueError(f"Step cannot be zero in '{item}'")
            elif (end - start) * step < 0:
                raise ValueError(f"Step does not move towards end in '{item}'")
            current = start
            values = []
            if step > 0:
                while current <= end + 1e-8:  # Small epsilon to account for float errors
                    values.append(current)
                    current += step
            else:
                while current >= end - 1e-8:
                    values.append(current)
                    current += step  # step is negative
            if type_func is float:
                values = [round(v, 8) for v in values]
            result.extend(values)
        else:
            # Single value
            value = type_func(item)
            result.append(value)
    return result

def get_parameters():
    """
    Opens a popup dialog to get parameters like guidance scales and inference steps.
    Allows multiple values separated by commas or ranges in the format 'start - end : step'.

    Returns:
    - guidance_scales (list of float): List of guidance scales.
    - inference_steps (list of int): List of inference steps.
    """
    # Use tkinter to create a dialog
    import tkinter as tk
    from tkinter import simpledialog, messagebox

    root = tk.Tk()
    root.withdraw()

    # Ask for guidance scales
    guidance_scales_str = simpledialog.askstring(
        "Input Guidance Scales",
        "Enter Guidance Scales (separated by commas or ranges):\n"
        "Examples:\n"
        "- Individual values: 5.0, 7.5, 10.0\n"
        "- Range with step: 5.0 - 10.0 : 0.5",
        parent=root
    )
    if guidance_scales_str is None:
        messagebox.showwarning("Input Cancelled", "No guidance scales were provided. Exiting.")
        return None, None

    # Ask for inference steps
    inference_steps_str = simpledialog.askstring(
        "Input Inference Steps",
        "Enter Inference Steps (separated by commas or ranges):\n"
        "Examples:\n"
        "- Individual values: 10, 20, 30\n"
        "- Range with step: 10 - 30 : 5",
        parent=root
    )
    if inference_steps_str is None:
        messagebox.showwarning("Input Cancelled", "No inference steps were provided. Exiting.")
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

def get_seed():
    """
    Opens a popup dialog to get the seed value.
    Includes a secondary button to randomize the seed.
    """
    import tkinter as tk
    from tkinter import simpledialog, messagebox

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
            import random
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

# --------------------- Video Generation Function ---------------------

def generate_video(
    prompt: str,
    negative_prompt: str,
    generate_type: str,
    pipe,
    output_path: str,
    image_or_video_path: Optional[str] = None,
    num_inference_steps: int = 42,
    guidance_scale: float = 10.0,
    seed: Optional[int] = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The positive description of the video to be generated.
    - negative_prompt (str): The negative description to avoid in the video.
    - generate_type (str): The type of video generation ('t2v', 'i2v', 'v2v').
    - pipe: The pre-loaded pipeline object.
    - output_path (str): The path where the generated video will be saved.
    - image_or_video_path (str, optional): The path of the image or video to be used for 'i2v' or 'v2v'.
    - num_inference_steps (int): Number of steps for the inference process.
    - guidance_scale (float): The scale for classifier-free guidance.
    - seed (int, optional): The seed for reproducibility.
    """
    try:
        # Set random seed for reproducibility
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)

        # Enable inference mode to reduce memory usage
        with torch.inference_mode():
            # Generate the video frames based on the prompt
            if generate_type == "i2v":
                image = load_image(image=image_or_video_path)
                if image is None:
                    raise ValueError(f"Failed to load image from path: {image_or_video_path}")
                video_generate = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).frames[0]
            elif generate_type == "t2v":
                video_generate = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).frames[0]
            elif generate_type == "v2v":
                video = load_video(image_or_video_path)
                if video is None:
                    raise ValueError(f"Failed to load video from path: {image_or_video_path}")
                video_generate = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    video=video,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).frames[0]
            else:
                raise ValueError(f"Invalid generate_type: {generate_type}. Choose from 't2v', 'i2v', 'v2v'.")

            # Export the generated frames to a video file. fps must be 8 for original video.
            export_to_video(video_generate, output_path, fps=8)
            print(f"Video saved to: {output_path}")

            # Calculate video duration based on num_frames and fps
            num_frames = 49
            fps = 8
            duration_seconds = num_frames / fps

            # Create corresponding .srt file
            create_srt_file(output_path, prompt, duration_seconds)

    except Exception as e:
        print(f"Error generating video for prompt '{prompt}': {e}")
        messagebox.showerror("Video Generation Error", f"Error generating video for prompt:\n{e}")

    finally:
        # Delete variables to free up memory
        del video_generate
        if 'image' in locals():
            del image
        if 'video' in locals():
            del video
        # Clear memory to prevent GPU/CPU overload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# --------------------- Main Function ---------------------

def main():
    global SEED  # Use the global SEED variable

    # Select the prompt list file
    prompt_file = select_prompt_file()
    if not prompt_file:
        return

    # Get parameters from the user
    guidance_scales, inference_steps = get_parameters()
    if not guidance_scales or not inference_steps:
        # User canceled or provided invalid input
        messagebox.showwarning("No Parameters Provided", "No parameters were provided. Using default values.")
        guidance_scales = GUIDANCE_SCALES
        inference_steps = INFERENCE_STEPS

    # Get seed value from the user
    SEED = get_seed()
    print(f"Using seed value: {SEED}")

    # Determine the directory of the prompt file
    output_dir = os.path.dirname(prompt_file)

    # Read and parse the prompts
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Replace the existing line-by-line parsing with the new comprehensive parser
        prompts = parse_prompt_file(lines)

    except Exception as e:
        print(f"Error reading prompt file: {e}")
        messagebox.showerror("File Read Error", f"Error reading prompt file:\n{e}")
        return

    if not prompts:
        print("No valid prompts found in the selected file.")
        messagebox.showinfo("No Prompts", "No valid prompts found in the selected file.")
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
            print("Positive prompt was too long and has been summarized to fit the model's token limit.")
            if tokenizer:
                pos_length = len(tokenizer.encode(positive_prompt))
                summarized_pos_length = len(tokenizer.encode(summarized_positive))
                print(f"Original Positive Prompt Length: {pos_length} tokens")
                print(f"Summarized Positive Prompt Length: {summarized_pos_length} tokens")
            else:
                print("Tokenizer not available to count tokens.")

        if summarized_negative != negative_prompt:
            print("Negative prompt was too long and has been summarized to fit the model's token limit.")
            if tokenizer:
                neg_length = len(tokenizer.encode(negative_prompt))
                summarized_neg_length = len(tokenizer.encode(summarized_negative))
                print(f"Original Negative Prompt Length: {neg_length} tokens")
                print(f"Summarized Negative Prompt Length: {summarized_neg_length} tokens")
            else:
                print("Tokenizer not available to count tokens.")

        # Generate a 5-word summary for the filename
        five_word_summary = create_five_word_summary(summarized_positive) if summarized_positive else "summary"

        # Sanitize the 5-word summary for filename usage
        safe_summary = sanitize_filename(five_word_summary)[:20]  # Further limit to prevent filesystem issues
        if not safe_summary:
            safe_summary = f"summary_{idx}"

        print(f"\nGenerating videos for prompt {idx}/{len(prompts)}:")
        print(f"Positive Prompt: {summarized_positive}")
        print(f"Negative Prompt: {summarized_negative}")
        print(f"5-Word Summary: {five_word_summary}")
        print(f"Output Filename: video_{idx}_5b_[gs]gs_[steps]steps_{safe_summary}.mp4")

        # Iterate over each guidance scale and inference step to generate multiple videos per prompt
        for gs in guidance_scales:
            for steps in inference_steps:
                # Define output filename with the new naming convention
                output_filename = f"video_{idx}_5b_{gs}gs_{steps}steps_{safe_summary}.mp4"
                output_path = os.path.join(output_dir, output_filename)

                print(f"  - Generating video with guidance_scale={gs} and num_inference_steps={steps}")
                print(f"    Output: {output_path}")

                # Clear any residual memory before loading the model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Load the pipeline inside the loop to clear memory between generations
                try:
                    print(f"Loading model pipeline '{MODEL_PATH_5B}'...")
                    if GENERATE_TYPE == "i2v":
                        pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_PATH_5B, torch_dtype=DTYPE)
                    elif GENERATE_TYPE == "t2v":
                        pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH_5B, torch_dtype=DTYPE)
                    elif GENERATE_TYPE == "v2v":
                        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(MODEL_PATH_5B, torch_dtype=DTYPE)
                    else:
                        raise ValueError(f"Invalid GENERATE_TYPE: {GENERATE_TYPE}. Choose from 't2v', 'i2v', 'v2v'.")

                    # Apply LoRA weights if provided
                    if LORA_PATH:
                        if not os.path.isfile(LORA_PATH):
                            raise FileNotFoundError(f"LoRA weights file not found at: {LORA_PATH}")
                        pipe.load_lora_weights(
                            LORA_PATH,
                            weight_name="pytorch_lora_weights.safetensors",
                            adapter_name="lora_adapter",
                        )
                        pipe.fuse_lora(lora_scale=1 / LORA_RANK)

                    # Set Scheduler
                    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

                    # Device Handling
                    if torch.cuda.is_available():
                        pipe.to("cuda")
                    else:
                        pipe.to("cpu")

                    # Enable memory optimizations
                    pipe.enable_sequential_cpu_offload()
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
                    pipe.enable_attention_slicing("max")  # Enable attention slicing for lower memory usage

                    # Optional: Enable xformers for memory-efficient attention (if available)
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception as e:
                        print(f"Could not enable xformers memory efficient attention: {e}")

                except Exception as e:
                    print(f"Error loading the pipeline: {e}")
                    messagebox.showerror("Pipeline Load Error", f"Error loading the pipeline:\n{e}")
                    continue  # Skip to next iteration

                # Generate the video with the current guidance_scale and num_inference_steps
                generate_video(
                    prompt=summarized_positive,
                    negative_prompt=summarized_negative,
                    generate_type=GENERATE_TYPE,
                    pipe=pipe,
                    output_path=output_path,
                    image_or_video_path=None,  # Modify if using 'i2v' or 'v2v'
                    num_inference_steps=steps,
                    guidance_scale=gs,
                    seed=SEED,  # Use the seed value obtained from the user
                )

                # Delete the pipeline to free up memory
                del pipe

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

if __name__ == "__main__":
    main()
