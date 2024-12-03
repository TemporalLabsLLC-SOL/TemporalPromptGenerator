import argparse
import os
import random
import time
import gc
import json
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    DiffusionPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.utils.logging import get_logger

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from transformers import AutoTokenizer, pipeline

# --------------------- Configuration ---------------------

# Summarization settings
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Summarization model
POSITIVE_MAX_TOKENS = 210  # Max tokens for positive prompts
NEGATIVE_MAX_TOKENS = 60   # Max tokens for negative prompts
POSITIVE_MIN_TOKENS = 80   # Min tokens for positive prompts
NEGATIVE_MIN_TOKENS = 30   # Min tokens for negative prompts

# Video generation settings
MODEL_PATH_5B = "THUDM/CogVideoX-5b"  # Pre-trained 5b model path
GENERATE_TYPE = "i2v"                   # Generation type: 't2v', 'i2v', 'v2v'
LORA_PATH = None                        # Path to LoRA weights if used
LORA_RANK = 128
GUIDANCE_SCALES = [7.0]                 # Default guidance scales (will be overridden by user input)
INFERENCE_STEPS = [50]                  # Default inference steps (will be overridden by user input)
SEED = 1990  # Default seed value
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32  # Updated DTYPE

# Caption generation settings
CAPTION_GENERATOR_MODEL_ID = "EleutherAI/gpt-neo-1.3B"  # Updated model ID
CAPTION_GENERATOR_CACHE_DIR = None

# Image generation settings
IMAGE_GENERATOR_MODEL_ID = "black-forest-labs/FLUX.1-dev"
IMAGE_GENERATOR_CACHE_DIR = None
IMAGE_GENERATOR_NUM_INFERENCE_STEPS = 50

OUTPUT_DIR = "outputs"

torch.set_float32_matmul_precision("high")
logger = get_logger(__name__)

# --------------------- Initialization ---------------------

# Initialize the summarization pipeline globally to avoid reloading for each prompt
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
    - text (str): The input text to summarize.
    - max_tokens (int): The maximum number of tokens allowed.

    Returns:
    - str: The summarized text within the token limit.
    """
    if not summarizer:
        print("Summarizer not initialized. Returning original text.")
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
        return text

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncates the input text to the specified token limit.

    Parameters:
    - text (str): The text to truncate.
    - max_tokens (int): The token limit.

    Returns:
    - str: The truncated text adhering to the token limit.
    """
    # Tokenize text
    tokenizer = summarizer.tokenizer
    tokens = tokenizer.encode(text, return_tensors="pt")
    num_tokens = tokens.shape[1]

    if num_tokens <= max_tokens:
        return text  # No need to truncate

    # Truncate the text to the maximum token limit
    truncated_text = tokenizer.decode(tokens[0, :max_tokens], skip_special_tokens=True)

    print("Text was too long and has been truncated to fit the token limit.")
    print(f"Original Length: {num_tokens} tokens")
    print(f"Truncated Length: {max_tokens} tokens")

    return truncated_text

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

def sanitize_filename(filename: str):
    """
    Sanitizes the filename by removing or replacing illegal characters.
    """
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def get_parameters():
    """
    Opens a popup dialog to get parameters like guidance scales, inference steps, and seed.
    Allows multiple values separated by commas.

    Returns:
    - guidance_scales (list of float): List of guidance scales.
    - inference_steps (list of int): List of inference steps.
    - seed (int): The seed value.
    """
    # Use tkinter to create a dialog
    root = tk.Tk()
    root.withdraw()

    # Ask for guidance scales
    guidance_scales_str = simpledialog.askstring(
        "Input Guidance Scales",
        "Enter Guidance Scales (separated by commas):\nExample: 5.0, 7.5, 10.0",
        parent=root
    )
    if guidance_scales_str is None:
        messagebox.showwarning("Input Cancelled", "No guidance scales were provided. Exiting.")
        return None, None, None

    # Ask for inference steps
    inference_steps_str = simpledialog.askstring(
        "Input Inference Steps",
        "Enter Inference Steps (separated by commas):\nExample: 10, 20, 30",
        parent=root
    )
    if inference_steps_str is None:
        messagebox.showwarning("Input Cancelled", "No inference steps were provided. Exiting.")
        return None, None, None

    # Ask for seed value
    seed_value = get_seed()

    root.destroy()

    # Parse the input strings into lists of numbers
    try:
        guidance_scales = [float(s.strip()) for s in guidance_scales_str.split(",") if s.strip()]
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid guidance scales input:\n{e}")
        return None, None, None

    try:
        inference_steps = [int(s.strip()) for s in inference_steps_str.split(",") if s.strip()]
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid inference steps input:\n{e}")
        return None, None, None

    return guidance_scales, inference_steps, seed_value

def get_seed():
    """
    Opens a popup dialog to get the seed value.
    Includes a secondary button to randomize the seed.
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

def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

# --------------------- Main Function ---------------------

def main():
    global SEED  # Use the global SEED variable

    # Create output directory
    output_dir = pathlib.Path(OUTPUT_DIR)
    os.makedirs(output_dir.as_posix(), exist_ok=True)

    # Get parameters from the user
    guidance_scales, inference_steps, SEED = get_parameters()
    if not guidance_scales or not inference_steps:
        # User canceled or provided invalid input
        messagebox.showwarning("No Parameters Provided", "No parameters were provided. Using default values.")
        guidance_scales = GUIDANCE_SCALES
        inference_steps = INFERENCE_STEPS

    print(f"Using seed value: {SEED}")

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    reset_memory()

    # Initialize caption generator
    print("Initializing caption generator...")
    try:
        caption_tokenizer = AutoTokenizer.from_pretrained(
            CAPTION_GENERATOR_MODEL_ID
        )
        caption_generator = pipeline(
            "text-generation",
            model=CAPTION_GENERATOR_MODEL_ID,
            device_map="auto",
            model_kwargs={
                "cache_dir": CAPTION_GENERATOR_CACHE_DIR,
                "torch_dtype": torch.float16,
            },
            tokenizer=caption_tokenizer
        )
    except Exception as e:
        print(f"Error initializing caption generator: {e}")
        messagebox.showerror("Initialization Error", f"Error initializing caption generator:\n{e}")
        return

    # Generate captions
    num_videos = simpledialog.askinteger("Number of Videos", "Enter the number of videos to generate:", initialvalue=5)
    if not num_videos:
        messagebox.showwarning("Input Cancelled", "No number of videos was provided. Exiting.")
        return

    captions = []
    for i in range(num_videos):
        num_words = random.choice([50, 75, 100])
        user_prompt = f"Could you generate a prompt for a video generation model?\nPlease limit the prompt to [{num_words}] words."

        # Generate caption
        output = caption_generator(
            user_prompt,
            max_new_tokens=226,
            do_sample=True,
            temperature=0.7,
        )
        caption = output[0]["generated_text"]

        # Optionally process the output to remove the prompt text if needed
        # For example, remove the prompt from the generated text
        if caption.startswith(user_prompt):
            caption = caption[len(user_prompt):].strip()

        captions.append(caption)
        logger.info(f"Generated caption: {caption}")

    # Save captions to file
    with open(output_dir / "captions.json", "w") as file:
        json.dump(captions, file)

    del caption_generator
    reset_memory()

    # Initialize image generator
    print("Initializing image generator...")
    try:
        image_generator = DiffusionPipeline.from_pretrained(
            IMAGE_GENERATOR_MODEL_ID,
            cache_dir=IMAGE_GENERATOR_CACHE_DIR,
            torch_dtype=torch.float16  # Updated DTYPE
        )
        image_generator.to("cuda")

        # Optional: Enable compilation and VAE tiling if supported
        if hasattr(image_generator.vae, 'enable_tiling'):
            image_generator.vae.enable_tiling()
    except Exception as e:
        print(f"Error initializing image generator: {e}")
        messagebox.showerror("Initialization Error", f"Error initializing image generator:\n{e}")
        return

    # Generate images
    images = []
    for index, caption in enumerate(captions):
        try:
            image = image_generator(
                prompt=caption,
                height=480,
                width=720,
                num_inference_steps=IMAGE_GENERATOR_NUM_INFERENCE_STEPS,
                guidance_scale=3.5,
            ).images[0]
            filename = sanitize_filename(caption[:25])
            image.save(output_dir / f"{index}_{filename}.png")
            images.append(image)
        except Exception as e:
            print(f"Error generating image for caption '{caption}': {e}")
            messagebox.showerror("Image Generation Error", f"Error generating image for caption:\n{e}")
            continue

    del image_generator
    reset_memory()

    # Initialize video generator
    print(f"\nLoading model pipeline '{MODEL_PATH_5B}'...")
    try:
        video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_PATH_5B, torch_dtype=DTYPE).to("cuda")
        video_generator.scheduler = CogVideoXDPMScheduler.from_config(
            video_generator.scheduler.config,
            timestep_spacing="trailing")

        # Optional: Enable VAE tiling if supported
        if hasattr(video_generator.vae, 'enable_tiling'):
            video_generator.vae.enable_tiling()

        # Device Handling
        if torch.cuda.is_available():
            video_generator.to("cuda")
        else:
            video_generator.to("cpu")

        # Enable memory optimizations
        video_generator.enable_sequential_cpu_offload()
        video_generator.vae.enable_slicing()
        if hasattr(video_generator.vae, 'enable_tiling'):
            video_generator.vae.enable_tiling()
        video_generator.enable_attention_slicing("max")  # Enable attention slicing for lower memory usage

        # Optional: Enable xformers for memory-efficient attention (if available)
        try:
            video_generator.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers memory efficient attention: {e}")

    except Exception as e:
        print(f"Error loading the pipeline: {e}")
        messagebox.showerror("Pipeline Load Error", f"Error loading the pipeline:\n{e}")
        return

    # Generate videos
    for index, (caption, image) in enumerate(zip(captions, images)):
        summarized_caption = summarize_text(caption, max_tokens=POSITIVE_MAX_TOKENS, min_tokens=POSITIVE_MIN_TOKENS)
        negative_prompt = ""  # You can define a negative prompt if needed

        # Generate a 5-word summary for the filename
        five_word_summary = create_five_word_summary(summarized_caption) if summarized_caption else "summary"

        # Sanitize the 5-word summary for filename usage
        safe_summary = sanitize_filename(five_word_summary)[:20]  # Further limit to prevent filesystem issues
        if not safe_summary:
            safe_summary = f"summary_{index}"

        print(f"\nGenerating videos for caption {index+1}/{len(captions)}:")
        print(f"Caption: {summarized_caption}")
        print(f"5-Word Summary: {five_word_summary}")
        print(f"Output Filename: video_{index}_{safe_summary}.mp4")

        # Iterate over each guidance scale and inference step to generate multiple videos per prompt
        for gs in guidance_scales:
            for steps in inference_steps:
                # Define output filename
                output_filename = f"video_{index}_{gs}gs_{steps}steps_{safe_summary}.mp4"
                output_path = os.path.join(output_dir, output_filename)

                print(f"  - Generating video with guidance_scale={gs} and num_inference_steps={steps}")
                print(f"    Output: {output_path}")

                try:
                    generator = torch.Generator().manual_seed(SEED)

                    video = video_generator(
                        image=image,
                        prompt=summarized_caption,
                        height=480,
                        width=720,
                        num_frames=49,
                        num_inference_steps=steps,
                        guidance_scale=gs,
                        use_dynamic_cfg=True,
                        generator=generator,
                    ).frames[0]

                    # Export to video
                    export_to_video(video, output_path, fps=8)

                    # Calculate video duration
                    num_frames = 49
                    fps = 8
                    duration_seconds = num_frames / fps

                    # Create corresponding .srt file
                    create_srt_file(output_path, summarized_caption, duration_seconds)
                except Exception as e:
                    print(f"Error generating video for caption '{caption}': {e}")
                    messagebox.showerror("Video Generation Error", f"Error generating video for caption:\n{e}")
                    continue

                # Clear memory after each video generation
                reset_memory()

    # Clear memory after all videos are generated
    reset_memory()

    print("\nAll videos have been generated successfully.")
    messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")

if __name__ == "__main__":
    main()
