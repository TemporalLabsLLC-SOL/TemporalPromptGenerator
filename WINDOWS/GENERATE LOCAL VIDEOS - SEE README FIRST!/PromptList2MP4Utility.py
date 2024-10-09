"""
This script demonstrates how to generate videos using the CogVideoX model with the Hugging Face `diffusers` pipeline.
It supports generating multiple videos based on a list of prompts provided in a text file.

Prompt List Format:
Each prompt should consist of a 'positive:' and 'negative:' section, separated by a delimiter line of dashes.

Example:
positive: [Your positive prompt here]
negative: [Your negative prompt here]
--------------------
positive: [Next positive prompt]
negative: [Next negative prompt]
--------------------
"""

import argparse
import os
import random
import time
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
from tkinter import filedialog, messagebox


def generate_video(
    prompt: str,
    generate_type: str,
    model_path: str,
    output_path: str,
    image_or_video_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - generate_type (str): The type of video generation ('t2v', 'i2v', 'v2v').
    - model_path (str): The path of the pre-trained model to be used.
    - output_path (str): The path where the generated video will be saved.
    - image_or_video_path (str, optional): The path of the image or video to be used as input for 'i2v' or 'v2v'.
    - lora_path (str, optional): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - num_inference_steps (int): Number of steps for the inference process.
    - guidance_scale (float): The scale for classifier-free guidance.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    try:
        # Initialize the pipeline based on the generation type
        if generate_type == "i2v":
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
            image = load_image(image=image_or_video_path)
            if image is None:
                raise ValueError(f"Failed to load image from path: {image_or_video_path}")
        elif generate_type == "t2v":
            pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
            image = None
            video = None
        elif generate_type == "v2v":
            pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
            video = load_video(image_or_video_path)
            if video is None:
                raise ValueError(f"Failed to load video from path: {image_or_video_path}")
        else:
            raise ValueError(f"Invalid generate_type: {generate_type}. Choose from 't2v', 'i2v', 'v2v'.")

        # Apply LoRA weights if provided
        if lora_path:
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(f"LoRA weights file not found at: {lora_path}")
            pipe.load_lora_weights(
                lora_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="lora_adapter",
            )
            pipe.fuse_lora(lora_scale=1 / lora_rank)

        # Set Scheduler
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        # Device Handling
        if torch.cuda.is_available():
            pipe.to("cuda")
        else:
            pipe.to("cpu")

        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # Set random seed for reproducibility
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

        # Generate the video frames based on the prompt
        if generate_type == "i2v":
            video_generate = pipe(
                prompt=prompt,
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
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        else:  # v2v
            video_generate = pipe(
                prompt=prompt,
                video=video,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]

        # Export the generated frames to a video file. fps must be 8 for original video.
        export_to_video(video_generate, output_path, fps=8)
        print(f"Video saved to: {output_path}")

    except Exception as e:
        print(f"Error generating video for prompt '{prompt}': {e}")
        messagebox.showerror("Video Generation Error", f"Error generating video for prompt:\n{e}")


def select_prompt_file():
    """
    Opens a file dialog to select a prompt list file.
    Returns the path to the selected file or None if canceled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Prompt List File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    if not file_path:
        messagebox.showwarning("No File Selected", "No prompt list file was selected. Exiting.")
        return None
    return file_path


def parse_prompt_line(line: str):
    """
    Parses a single line from the prompt list file.
    Expected format:
    positive: [Your positive prompt]
    negative: [Your negative prompt]
    --------------------

    Returns a dictionary with keys: 'positive' or 'negative', or None for delimiter lines.
    """
    if line.startswith("positive:"):
        positive_prompt = line[len("positive:"):].strip()
        return {"positive": positive_prompt}
    elif line.startswith("negative:"):
        negative_prompt = line[len("negative:"):].strip()
        return {"negative": negative_prompt}
    elif set(line.strip()) == set("-"):
        return {"delimiter": True}
    else:
        # Unrecognized line
        return {"unrecognized": True}


def sanitize_filename(filename: str):
    """
    Sanitizes the filename by removing or replacing illegal characters.
    """
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()


def main():
    # Select the prompt list file
    prompt_file = select_prompt_file()
    if not prompt_file:
        return

    # Determine the directory of the prompt file
    output_dir = os.path.dirname(prompt_file)

    # Read and parse the prompts
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        prompts = []
        current_prompt = {}
        for idx, line in enumerate(lines, start=1):
            parsed = parse_prompt_line(line)
            if "unrecognized" in parsed:
                print(f"Warning: Unrecognized line format at line {idx}: '{line.strip()}'. Skipping.")
                continue
            if "delimiter" in parsed:
                if "positive" in current_prompt and "negative" in current_prompt:
                    prompts.append(current_prompt)
                    current_prompt = {}
                else:
                    if "positive" in current_prompt:
                        print(f"Warning: 'negative:' section missing for prompt at line {idx}. Skipping.")
                    current_prompt = {}
                continue
            if "positive" in parsed:
                if "positive" in current_prompt:
                    print(f"Warning: New 'positive:' found before completing previous prompt at line {idx}. Skipping previous prompt.")
                current_prompt["positive"] = parsed["positive"]
            elif "negative" in parsed:
                if "positive" not in current_prompt:
                    print(f"Warning: 'negative:' section without a preceding 'positive:' at line {idx}. Skipping.")
                    continue
                current_prompt["negative"] = parsed["negative"]

        # Handle last prompt if missing delimiter
        if "positive" in current_prompt and "negative" in current_prompt:
            prompts.append(current_prompt)
        elif "positive" in current_prompt:
            print(f"Warning: Last prompt missing 'negative:' section. Skipping.")

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

        # Combine positive and negative prompts as per the original generate_video function
        combined_prompt = f"positive: {positive_prompt}\nnegative: {negative_prompt}"

        # Define output path
        safe_prompt = sanitize_filename(positive_prompt)[:50]  # Limit length to prevent filesystem issues
        if not safe_prompt:
            safe_prompt = f"video_{idx}"
        output_filename = f"video_{idx}_{safe_prompt}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\nGenerating video {idx}/{len(prompts)}:")
        print(f"Prompt: {positive_prompt}")
        print(f"Output: {output_path}")

        # Generate the video - CUSTOMIZE OUTPUT SETTINGS
        generate_video(
            prompt=combined_prompt,
            generate_type="t2v",  # Assuming all prompts are text-to-video
            model_path="THUDM/CogVideoX-5b",  # Modify if different models are used
            output_path=output_path,
            image_or_video_path=None,  # Not needed for 't2v'
            lora_path=None,  # Modify if using LoRA weights
            lora_rank=128,
            num_inference_steps=48,
            guidance_scale=4.0,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            seed=random.randint(0, 1000000),
        )

    print("\nAll videos have been generated successfully.")
    messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")


if __name__ == "__main__":
    main()
