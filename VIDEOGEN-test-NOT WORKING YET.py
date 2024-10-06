import json
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image

from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

# Low GPU memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# Config and model path
model_name = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
sampler_name = "DDIM_Origin"

# Load pretrained model if need
transformer_path = None
vae_path = None
lora_path = None

# Other params
sample_size = [384, 672]
fps = 8

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype = torch.bfloat16

# Function to get the prompt file from user
def get_prompt_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_selected = filedialog.askopenfilename(title="Select Prompt List File", filetypes=[("Text files", "*.txt")])
    return file_selected

# Read prompts from text file
def load_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        prompt_blocks = content.strip().split('--------------------')
        prompts = []
        for block in prompt_blocks:
            positive_prompt = None
            negative_prompt = None
            for line in block.strip().split('\n'):
                if line.startswith("positive:"):
                    positive_prompt = line[len("positive:"):].strip()
                elif line.startswith("negative:"):
                    negative_prompt = line[len("negative:"):].strip()
            if positive_prompt:
                prompts.append((positive_prompt, negative_prompt))
        return prompts

# Initialize models and pipelines
transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    model_name,
    subfolder="transformer",
).to(weight_dtype)

vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name,
    subfolder="vae"
).to(weight_dtype)

text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)

Choosen_Scheduler = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]

scheduler = Choosen_Scheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)

if transformer.config.in_channels != vae.config.latent_channels:
    pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
else:
    pipeline = CogVideoX_Fun_Pipeline.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )

if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

def generate_videos_from_prompts(prompts, save_path):
    generator = torch.Generator(device="cuda").manual_seed(43)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for index, (prompt, negative_prompt) in enumerate(prompts):
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=49,  # Assuming video_length is fixed for all prompts
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=6.0,
                num_inference_steps=50,
            ).videos

        video_path = os.path.join(save_path, f"video_{index + 1:08d}.mp4")
        save_videos_grid(sample, video_path, fps=fps)
        print(f"Saved video {index + 1} at {video_path}")

# Main script execution
if __name__ == "__main__":
    prompt_file_path = get_prompt_file()
    if prompt_file_path:
        prompts = load_prompts_from_file(prompt_file_path)
        save_directory = os.path.join(os.path.dirname(prompt_file_path), "videos")
        generate_videos_from_prompts(prompts, save_directory)
    else:
        print("No prompt file selected. Exiting...")