"""
Batch Text-to-Video Generator using CogVideoX and Diffusers with GUI Options

This script provides a graphical user interface (GUI) to select a list of video prompts from a text file
and generate corresponding videos using the CogVideoX model.

Features:
- Select a prompt list file (.txt) with 'positive:' and 'negative:' sections.
- Choose the type of video generation:
    - Text-to-Video (t2v)
    - Image-to-Video (i2v)
    - Video-to-Video (v2v)
- Select the model version (2b or 5b).
- Input image or video paths based on generation type.
- Adjust generation parameters:
    - Number of Inference Steps
    - Guidance Scale (CFG)
    - Video Length (seconds)
    - Resolution
    - Data Type (float16 or bfloat16)
    - Scheduler Selection
    - Seed (for reproducibility)
    - LoRA Weights (optional)
- Generate videos sequentially based on the prompts.
- Export generated videos to the same directory as the prompt list file.
- Real-time progress updates and logging within the GUI.
- Progress bar to indicate the generation process status.

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
import platform
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
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
    video_length_sec: float = 6.0,
    fps: int = 8,
    scheduler_type: str = "dpm",  # 'dpm' or 'ddim'
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
    - video_length_sec (float): Desired length of the video in seconds.
    - fps (int): Frames per second for the output video.
    - scheduler_type (str): Type of scheduler to use ('dpm' or 'ddim').
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
        if scheduler_type.lower() == "dpm":
            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        elif scheduler_type.lower() == "ddim":
            pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        else:
            raise ValueError(f"Invalid scheduler_type: {scheduler_type}. Choose from 'dpm' or 'ddim'.")

        # Device Handling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)

        # Conditionally apply optimizations based on OS
        if platform.system() != "Windows":
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            print("Applied optimizations for non-Windows OS.")
        else:
            print("Skipped optimizations for Windows OS.")

        # Set random seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)

        # Determine the number of frames based on video length and fps
        num_frames = int(video_length_sec * fps)

        # Generate the video frames based on the prompt
        if generate_type == "i2v":
            video_generate = pipe(
                prompt=prompt,
                image=image,  # The path of the image to be used as the background of the video
                num_videos_per_prompt=1,  # Number of videos to generate per prompt
                num_inference_steps=num_inference_steps,  # Number of inference steps
                num_frames=num_frames,  # Number of frames to generate
                use_dynamic_cfg=True,  # Used for DPM Scheduler
                guidance_scale=guidance_scale,
                generator=generator,  # Set the seed for reproducibility
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
                video=video,  # The path of the video to be used as the background of the video
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,  # Set the seed for reproducibility
            ).frames[0]
        # Export the generated frames to a video file. fps must be set as per requirement.
        export_to_video(video_generate, output_path, fps=fps)
        return f"Video saved to: {output_path}"

    except Exception as e:
        # Handle exceptions and provide informative error messages
        error_message = f"Error during video generation: {e}"
        print(error_message)
        return error_message


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
        self.generate_type = tk.StringVar(value="t2v")
        self.model_selection = tk.StringVar(value="CogVideoX-5b")  # New variable for model selection
        self.inference_steps = tk.IntVar(value=50)
        self.guidance_scale = tk.DoubleVar(value=6.0)
        self.video_length = tk.DoubleVar(value=6.0)  # in seconds
        self.resolution = tk.StringVar(value="4K UHD (3840x2160)")
        self.data_type = tk.StringVar(value="bfloat16")
        self.scheduler_type = tk.StringVar(value="dpm")  # 'dpm' or 'ddim'
        self.seed = tk.StringVar(value="Random")
        self.lora_path = tk.StringVar()
        self.lora_rank = tk.IntVar(value=128)
        self.image_or_video_path = tk.StringVar()  # Added missing variable
        self.status_text = tk.StringVar(value="Select a prompt file and set parameters.")

        # Define resolution options
        self.resolution_options = [
            "480p (720x480)",
        ]

        # Define data type options
        self.data_type_options = [
            "float16",
            "bfloat16",
        ]

        # Define scheduler options
        self.scheduler_options = [
            "dpm",
            "ddim",
        ]

        # Define model options
        self.model_options = [
            "CogVideoX-2b",
            "CogVideoX-5b",
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

        # Generation Type Selection
        type_frame = tk.Frame(self.master)
        type_frame.pack(pady=5, padx=10, fill=tk.X)

        type_label = tk.Label(type_frame, text="Generation Type:")
        type_label.pack(side=tk.LEFT)

        type_menu = tk.OptionMenu(type_frame, self.generate_type, "t2v", "i2v", "v2v", command=self.update_generate_type)
        type_menu.pack(side=tk.LEFT, padx=5)

        # Model Selection
        model_frame = tk.Frame(self.master)
        model_frame.pack(pady=5, padx=10, fill=tk.X)

        model_label = tk.Label(model_frame, text="Model Selection:")
        model_label.pack(side=tk.LEFT)

        model_menu = tk.OptionMenu(model_frame, self.model_selection, *self.model_options)
        model_menu.pack(side=tk.LEFT, padx=5)

        # Image or Video Path (Dynamic)
        self.media_frame = tk.Frame(self.master)
        self.media_frame.pack(pady=5, padx=10, fill=tk.X)
        self.media_label = tk.Label(self.media_frame, text="Image/Video Path:")
        self.media_entry = tk.Entry(self.media_frame, textvariable=self.image_or_video_path, width=50)
        self.media_browse_button = tk.Button(self.media_frame, text="Browse", command=self.browse_media)

        # Initially hide media selection
        self.media_label.pack_forget()
        self.media_entry.pack_forget()
        self.media_browse_button.pack_forget()

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

        # Data Type
        data_type_label = tk.Label(params_frame, text="Data Type:")
        data_type_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        data_type_menu = tk.OptionMenu(params_frame, self.data_type, *self.data_type_options)
        data_type_menu.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        # Scheduler Selection
        scheduler_label = tk.Label(params_frame, text="Scheduler:")
        scheduler_label.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        scheduler_menu = tk.OptionMenu(params_frame, self.scheduler_type, *self.scheduler_options)
        scheduler_menu.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        # Seed
        seed_label = tk.Label(params_frame, text="Seed:")
        seed_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        seed_entry = tk.Entry(params_frame, textvariable=self.seed)
        seed_entry.grid(row=6, column=1, padx=5, pady=5)

        # LoRA Weights
        lora_frame = tk.Frame(params_frame)
        lora_frame.grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        lora_label = tk.Label(lora_frame, text="LoRA Weights Path:")
        lora_label.pack(side=tk.LEFT)

        lora_entry = tk.Entry(lora_frame, textvariable=self.lora_path, width=40)
        lora_entry.pack(side=tk.LEFT, padx=5)

        lora_browse_button = tk.Button(lora_frame, text="Browse", command=self.browse_lora)
        lora_browse_button.pack(side=tk.LEFT)

        # Configure grid weights
        for i in range(8):
            params_frame.rowconfigure(i, weight=1)
        params_frame.columnconfigure(1, weight=1)

        # Generate Button
        generate_button = tk.Button(self.master, text="Generate Videos", command=self.start_generation_thread)
        generate_button.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.master, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5, padx=10)

        # Status and Log
        status_frame = tk.Frame(self.master)
        status_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        status_label = tk.Label(status_frame, text="Status:")
        status_label.pack(anchor=tk.W)

        self.log_area = scrolledtext.ScrolledText(status_frame, height=15, state='disabled')
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Prompt List File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if file_path:
            self.prompt_file.set(file_path)
            self.log(f"Selected prompt file: {file_path}")

    def browse_media(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")],
        )
        if file_path:
            self.image_or_video_path.set(file_path)
            self.log(f"Selected media file: {file_path}")

    def browse_lora(self):
        file_path = filedialog.askopenfilename(
            title="Select LoRA Weights File",
            filetypes=[("SafeTensors Files", "*.safetensors"), ("All Files", "*.*")],
        )
        if file_path:
            self.lora_path.set(file_path)
            self.log(f"Selected LoRA weights file: {file_path}")

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
        generate_type = self.generate_type.get()
        model_selected = self.model_selection.get()
        image_or_video_path = self.image_or_video_path.get()
        lora_path = self.lora_path.get() if generate_type in ["i2v", "v2v"] else None
        lora_rank = self.lora_rank.get()
        data_type = self.data_type.get()
        scheduler_type = self.scheduler_type.get()

        # Map model selection to actual model paths
        model_map = {
            "CogVideoX-2b": "THUDM/CogVideoX-2b",
            "CogVideoX-5b": "THUDM/CogVideoX-5b",
        }
        model_path = model_map.get(model_selected)
        if not model_path:
            self.log(f"Error: Invalid model selected: {model_selected}")
            messagebox.showerror("Model Error", "Please select a valid model.")
            self.enable_generate_button()
            return

        if not prompt_path or not os.path.isfile(prompt_path):
            self.log("Error: No valid prompt file selected.")
            messagebox.showerror("File Error", "Please select a valid prompt list file.")
            self.enable_generate_button()
            return

        if generate_type in ["i2v", "v2v"] and not image_or_video_path:
            self.log(f"Error: {generate_type.upper()} requires an image or video path.")
            messagebox.showerror("Input Error", f"{generate_type.upper()} requires an image or video path.")
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
        fps = self.fps  # Fixed FPS as per original script

        # Determine resolution dimensions
        resolution_dict = {
            "480p (720x480)": (720,480),
        }
        resolution_dim = resolution_dict.get(resolution, (720, 480))  # Default to 4K UHD

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

        # Update progress bar
        self.progress['maximum'] = len(prompts)
        self.progress['value'] = 0

        # Iterate through each prompt and generate videos
        total = len(prompts)
        for idx, prompt_data in enumerate(prompts, start=1):
            positive_prompt = prompt_data.get("positive")
            negative_prompt = prompt_data.get("negative")

            if not positive_prompt or not negative_prompt:
                self.log(f"Skipping prompt {idx}: Incomplete 'positive' or 'negative' sections.")
                self.progress['value'] += 1
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
            self.log(f"Output: {output_path}")

            # Generate the video
            result = generate_video(
                prompt=combined_prompt,
                generate_type=generate_type,
                model_path=model_path,  # Use the selected model path
                output_path=output_path,
                image_or_video_path=image_or_video_path if generate_type in ["i2v", "v2v"] else None,
                lora_path=lora_path,
                lora_rank=lora_rank,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                dtype=torch.float16 if data_type == "float16" else torch.bfloat16,
                seed=seed,
                video_length_sec=video_length_sec,
                fps=fps,
                scheduler_type=scheduler_type.lower(),
            )

            self.log(result)
            self.progress['value'] += 1

        self.log("\nAll videos have been generated successfully.")
        messagebox.showinfo("Generation Complete", "All videos have been generated successfully.")
        self.enable_generate_button()

    def enable_generate_button(self):
        # Re-enable the Generate button after generation is complete
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget('text') == 'Generate Videos':
                widget.config(state='normal')

    def update_generate_type(self, selected_type):
        if selected_type in ["i2v", "v2v"]:
            self.media_label.pack(side=tk.LEFT)
            self.media_entry.pack(side=tk.LEFT, padx=5)
            self.media_browse_button.pack(side=tk.LEFT)
        else:
            self.media_label.pack_forget()
            self.media_entry.pack_forget()
            self.media_browse_button.pack_forget()


def main():
    # Initialize the main window
    root = tk.Tk()
    app = VideoGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
