"""
Batch Text-to-Video Generator using CogVideoX and Diffusers with GUI Options

This script provides a graphical user interface (GUI) to select a list of video prompts from a text file
and generate corresponding videos using the CogVideoX model. Users can adjust various generation
parameters before initiating the video creation process.

Features:
- Select a prompt list file (.txt) with 'positive:' and 'negative:' sections.
- Adjust generation parameters:
    - Number of Inference Steps
    - Guidance Scale (CFG)
    - Video Length (seconds)
    - Resolution
    - Seed (for reproducibility)
- Generate videos sequentially based on the prompts.
- Export generated videos to the same directory as the prompt list file.
- Real-time progress updates within the GUI.

Prompt List Format:
Each prompt block should contain 'positive:' and 'negative:' sections, separated by a line of dashes.

Example:
positive: [Your positive prompt here]
negative: [Your negative prompt here]
--------------------
positive: [Next positive prompt]
negative: [Next negative prompt]
--------------------
"""

import os
import random
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video


def generate_video(
    prompt: str,
    generate_type: str,
    model_path: str,
    output_path: str,
    image_or_video_path: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The combined 'positive' and 'negative' prompt.
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
                num_frames=int(video_length_sec * fps),
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        elif generate_type == "t2v":
            video_generate = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=int(video_length_sec * fps),
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
        export_to_video(video_generate, output_path, fps=fps)
        return f"Video saved to: {output_path}"

    except Exception as e:
        return f"Error generating video for prompt: {e}"


def parse_prompt_list(file_path: str):
    """
    Parses the prompt list file and extracts positive and negative prompts.

    Parameters:
    - file_path (str): Path to the prompt list file.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries with 'positive' and 'negative' keys.
    """
    prompts = []
    current_prompt = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments
            if line.startswith("positive:"):
                if "positive" in current_prompt:
                    print(f"Warning: Found 'positive:' at line {line_num} without a preceding 'negative:'. Overwriting previous 'positive:'.")
                current_prompt["positive"] = line[len("positive:"):].strip()
            elif line.startswith("negative:"):
                if "positive" not in current_prompt:
                    print(f"Warning: Found 'negative:' at line {line_num} without a preceding 'positive:'. Skipping.")
                    continue
                current_prompt["negative"] = line[len("negative:"):].strip()
                prompts.append(current_prompt)
                current_prompt = {}
            elif set(line) == set("-"):
                continue  # Delimiter line
            else:
                print(f"Warning: Unrecognized line format at line {line_num}: '{line}'. Skipping.")
    return prompts


def sanitize_filename(filename: str):
    """
    Sanitizes the filename by removing or replacing illegal characters.

    Parameters:
    - filename (str): Original filename.

    Returns:
    - str: Sanitized filename.
    """
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()


class VideoGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Batch Text-to-Video Generator")

        # Initialize variables
        self.prompt_file = tk.StringVar()
        self.inference_steps = tk.IntVar(value=50)
        self.guidance_scale = tk.DoubleVar(value=6.0)
        self.video_length = tk.DoubleVar(value=6.0)  # in seconds
        self.resolution = tk.StringVar(value="4K UHD (3840x2160)")
        self.seed = tk.StringVar(value="Random")
        self.status_text = tk.StringVar(value="Select a prompt file and set parameters.")

        # Define resolution options
        self.resolution_options = [
            "720p (1280x720)",
            "1080p (1920x1080)",
            "4K UHD (3840x2160)",
            "8K UHD (7680x4320)",
        ]
        self.fps = 8  # Fixed fps as per original script

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Prompt File Selection
        prompt_frame = tk.Frame(self.master)
        prompt_frame.pack(pady=10, padx=10, fill=tk.X)

        prompt_label = tk.Label(prompt_frame, text="Prompt List File:")
        prompt_label.pack(side=tk.LEFT)

        prompt_entry = tk.Entry(prompt_frame, textvariable=self.prompt_file, width=50)
        prompt_entry.pack(side=tk.LEFT, padx=5)

        browse_button = tk.Button(prompt_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.LEFT)

        # Generation Parameters
        params_frame = tk.LabelFrame(self.master, text="Generation Parameters")
        params_frame.pack(pady=10, padx=10, fill=tk.X)

        # Number of Inference Steps
        steps_label = tk.Label(params_frame, text="Number of Inference Steps:")
        steps_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        steps_entry = tk.Entry(params_frame, textvariable=self.inference_steps)
        steps_entry.grid(row=0, column=1, padx=5, pady=5)

        # Guidance Scale (CFG)
        cfg_label = tk.Label(params_frame, text="Guidance Scale (CFG):")
        cfg_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        cfg_entry = tk.Entry(params_frame, textvariable=self.guidance_scale)
        cfg_entry.grid(row=1, column=1, padx=5, pady=5)

        # Video Length
        length_label = tk.Label(params_frame, text="Video Length (seconds):")
        length_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        length_entry = tk.Entry(params_frame, textvariable=self.video_length)
        length_entry.grid(row=2, column=1, padx=5, pady=5)

        # Resolution
        resolution_label = tk.Label(params_frame, text="Resolution:")
        resolution_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        resolution_menu = tk.OptionMenu(params_frame, self.resolution, *self.resolution_options)
        resolution_menu.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Seed
        seed_label = tk.Label(params_frame, text="Seed:")
        seed_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        seed_entry = tk.Entry(params_frame, textvariable=self.seed)
        seed_entry.grid(row=4, column=1, padx=5, pady=5)

        # Configure grid weights
        params_frame.columnconfigure(1, weight=1)

        # Generate Button
        generate_button = tk.Button(self.master, text="Generate Videos", command=self.start_generation_thread)
        generate_button.pack(pady=10)

        # Status and Log
        status_frame = tk.Frame(self.master)
        status_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        status_label = tk.Label(status_frame, text="Status:")
        status_label.pack(anchor=tk.W)

        self.log_area = scrolledtext.ScrolledText(status_frame, height=10, state='disabled')
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Prompt List File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if file_path:
            self.prompt_file.set(file_path)
            self.log(f"Selected prompt file: {file_path}")

    def log(self, message: str):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def start_generation_thread(self):
        # Disable the Generate button to prevent multiple clicks
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget('text') == 'Generate Videos':
                widget.config(state='disabled')
        # Start the generation in a separate thread
        thread = threading.Thread(target=self.generate_videos)
        thread.start()

    def generate_videos(self):
        prompt_path = self.prompt_file.get()
        if not prompt_path or not os.path.isfile(prompt_path):
            self.log("Error: No valid prompt file selected.")
            messagebox.showerror("File Error", "Please select a valid prompt list file.")
            self.enable_generate_button()
            return

        # Read and parse prompts
        self.log("Parsing prompt list...")
        prompts = parse_prompt_list(prompt_path)
        if not prompts:
            self.log("Error: No valid prompts found.")
            messagebox.showerror("Parsing Error", "No valid prompts found in the selected file.")
            self.enable_generate_button()
            return

        # Retrieve generation parameters
        num_steps = self.inference_steps.get()
        guidance_scale = self.guidance_scale.get()
        video_length_sec = self.video_length.get()
        resolution = self.resolution.get()
        seed_input = self.seed.get()

        # Set FPS based on resolution if needed, or keep fixed
        global fps
        fps = 8  # Fixed FPS as per original script

        # Determine resolution dimensions
        resolution_dict = {
            "720p (1280x720)": (1280, 720),
            "1080p (1920x1080)": (1920, 1080),
            "4K UHD (3840x2160)": (3840, 2160),
            "8K UHD (7680x4320)": (7680, 4320),
        }
        resolution_dim = resolution_dict.get(resolution, (3840, 2160))  # Default to 4K UHD

        # Handle seed
        if seed_input.lower() == "random":
            seed = random.randint(0, 1000000)
        else:
            try:
                seed = int(seed_input)
            except ValueError:
                self.log("Invalid seed input. Using random seed.")
                seed = random.randint(0, 1000000)

        # Determine output directory (same as prompt file directory)
        output_dir = os.path.dirname(prompt_path)
        if not os.path.isdir(output_dir):
            self.log(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Iterate through each prompt and generate videos
        total = len(prompts)
        for idx, prompt_data in enumerate(prompts, start=1):
            positive_prompt = prompt_data.get("positive")
            negative_prompt = prompt_data.get("negative")

            if not positive_prompt or not negative_prompt:
                self.log(f"Skipping prompt {idx}: Incomplete 'positive' or 'negative' sections.")
                continue

            # Combine positive and negative prompts
            combined_prompt = f"positive: {positive_prompt}\nnegative: {negative_prompt}"

            # Define output path
            safe_prompt = sanitize_filename(positive_prompt)[:50]  # Limit length to prevent filesystem issues
            if not safe_prompt:
                safe_prompt = f"video_{idx}"
            output_filename = f"video_{idx}_{safe_prompt}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            self.log(f"Generating video {idx}/{total}:")
            self.log(f"Prompt: {positive_prompt}")
            self.log(f"Output: {output_path}")

            # Generate the video
            result = generate_video(
                prompt=combined_prompt,
                generate_type="t2v",  # Fixed as per user request
                model_path="THUDM/CogVideoX-5b",  # Modify if different models are used
                output_path=output_path,
                image_or_video_path=None,  # Not needed for 't2v'
                lora_path=None,  # Modify if using LoRA weights
                lora_rank=128,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                seed=seed,
            )

            self.log(result)

        self.log("\nAll videos have been generated successfully.")
        messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")
        self.enable_generate_button()

    def enable_generate_button(self):
        # Re-enable the Generate button after generation is complete
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget('text') == 'Generate Videos':
                widget.config(state='normal')



def parse_prompt_list(file_path: str):
    """
    Parses the prompt list file and extracts positive and negative prompts.

    Parameters:
    - file_path (str): Path to the prompt list file.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries with 'positive' and 'negative' keys.
    """
    prompts = []
    current_prompt = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments
            if line.startswith("positive:"):
                if "positive" in current_prompt:
                    print(f"Warning: Found 'positive:' at line {line_num} without a preceding 'negative:'. Overwriting previous 'positive:'.")
                current_prompt["positive"] = line[len("positive:"):].strip()
            elif line.startswith("negative:"):
                if "positive" not in current_prompt:
                    print(f"Warning: Found 'negative:' at line {line_num} without a preceding 'positive:'. Skipping.")
                    continue
                current_prompt["negative"] = line[len("negative:"):].strip()
                prompts.append(current_prompt)
                current_prompt = {}
            elif set(line) == set("-"):
                continue  # Delimiter line
            else:
                print(f"Warning: Unrecognized line format at line {line_num}: '{line}'. Skipping.")
    return prompts


def sanitize_filename(filename: str):
    """
    Sanitizes the filename by removing or replacing illegal characters.

    Parameters:
    - filename (str): Original filename.

    Returns:
    - str: Sanitized filename.
    """
    keepcharacters = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()


def generate_video(
    prompt: str,
    generate_type: str,
    model_path: str,
    output_path: str,
    image_or_video_path: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The combined 'positive' and 'negative' prompt.
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

        # Determine the number of frames based on video length and fps
        num_frames = int(video_length_sec * fps)

        # Generate the video frames based on the prompt
        if generate_type == "i2v":
            video_generate = pipe(
                prompt=prompt,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        elif generate_type == "t2v":
            video_generate = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
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
        export_to_video(video_generate, output_path, fps=fps)
        return f"Video saved to: {output_path}"

    except Exception as e:
        return f"Error generating video for prompt: {e}"


def main():
    # Initialize the main window
    root = tk.Tk()
    app = VideoGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
