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
TOKENIZER_NAME = "gpt2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
POSITIVE_MAX_TOKENS = 210
NEGATIVE_MAX_TOKENS = 60
POSITIVE_MIN_TOKENS = 80
NEGATIVE_MIN_TOKENS = 30

# Default namespace values:
DEFAULTS = {
    "model": "HYVideo-T/2-cfgdistill",
    "latent_channels": 16,
    "precision": "bf16",
    "rope_theta": 256,
    "vae": "884-16c-hy",
    "vae_precision": "fp16",
    "vae_tiling": True,
    "text_encoder": "llm",
    "text_encoder_precision": "fp16",
    "text_states_dim": 4096,
    "text_len": 256,
    "tokenizer": "llm",
    "prompt_template": "dit-llm-encode",
    "prompt_template_video": "dit-llm-encode-video",
    "hidden_state_skip_layer": 2,
    "apply_final_norm": False,
    "text_encoder_2": "clipL",
    "text_encoder_precision_2": "fp16",
    "text_states_dim_2": 768,
    "tokenizer_2": "clipL",
    "text_len_2": 77,
    "denoise_type": "flow",
    "flow_shift": 7.0,
    "flow_reverse": True,
    "flow_solver": "euler",
    "use_linear_quadratic_schedule": False,
    "linear_schedule_end": 25,
    "model_base": "ckpts",
    "dit_weight": "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    "model_resolution": "540p",
    "load_key": "module",
    "use_cpu_offload": True,
    "batch_size": 1,
    "infer_steps": 50,
    "disable_autocast": False,
    "save_path": "/home/solomon/Downloads/Video_1_The vintage Thanksgi",
    "save_path_suffix": "",
    "name_suffix": "",
    "num_videos": 1,
    "video_size": [720, 480],
    "video_length": 129,
    "prompt": "The vintage Thanksgiving Home Videos for a Utah Valley Family are shot on a PROFESSIONAL - RED - V-Raptor ST (2021) camera in 8K resolution. The frame is shot from a slightly low angle, emphasizing the grandeur of the family's gathering. Soft, natural light pours in through large windows, casting a warm glow on happy family faces.",
    "seed_type": "auto",
    "seed": 1990,
    "neg_prompt": "The scene features blurry background figures due to improper camera composition. A distracting background of holiday decorations and cluttered tables take center stage, overwhelming the viewer's attention.",
    "cfg_scale": 1.0,
    "embedded_cfg_scale": 10.0,
    "reproduce": False,
    "ulysses_degree": 1,
    "ring_degree": 1
}


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
                    print(f"Warning: 'negative:' section missing. Skipping.")
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

def generate_video(args: dict, prompt: str, negative_prompt: str, output_path: str):
    """
    Generates video using provided arguments.
    Maps the arguments to the sample_video.py command line as needed.
    Modify sample_video.py to accept all these arguments if required.
    """

    # Convert video_size to two separate values
    vs = args["video_size"]
    width, height = vs if isinstance(vs, (list, tuple)) else DEFAULTS["video_size"]

    cmd = [
        "python3", "sample_video.py",
        "--model", args["model"],
        "--latent-channels", str(args["latent_channels"]),
        "--precision", args["precision"],
        "--rope-theta", str(args["rope_theta"]),
        "--vae", args["vae"],
        "--vae-precision", args["vae_precision"],
        "--text-encoder", args["text_encoder"],
        "--text-encoder-precision", args["text_encoder_precision"],
        "--text-states-dim", str(args["text_states_dim"]),
        "--text-len", str(args["text_len"]),
        "--tokenizer", args["tokenizer"],
        "--prompt-template", args["prompt_template"],
        "--prompt-template-video", args["prompt_template_video"],
        "--hidden-state-skip-layer", str(args["hidden_state_skip_layer"]),
        "--text-encoder-2", args["text_encoder_2"],
        "--text-encoder-precision-2", args["text_encoder_precision_2"],
        "--text-states-dim-2", str(args["text_states_dim_2"]),
        "--tokenizer-2", args["tokenizer_2"],
        "--text-len-2", str(args["text_len_2"]),
        "--denoise-type", args["denoise_type"],
        "--flow-shift", str(args["flow_shift"]),
        "--flow-solver", args["flow_solver"],
        "--model-base", args["model_base"],
        "--dit-weight", args["dit_weight"],
        "--model-resolution", args["model_resolution"],
        "--load-key", args["load_key"],
        "--batch-size", str(args["batch_size"]),
        "--infer-steps", str(args["infer_steps"]),
        "--save-path", output_path,
        "--num-videos", str(args["num_videos"]),
        "--video-size", str(width), str(height),
        "--video-length", str(args["video_length"]),
        "--prompt", prompt,
        "--seed-type", args["seed_type"],
        "--seed", str(args["seed"]),
        "--cfg-scale", str(args["cfg_scale"]),
        "--embedded-cfg-scale", str(args["embedded_cfg_scale"]),
        "--ulysses-degree", str(args["ulysses_degree"]),
        "--ring-degree", str(args["ring_degree"])
    ]

    # Boolean flags
    if args["vae_tiling"]:
        cmd.append("--vae-tiling")
    if args["apply_final_norm"]:
        cmd.append("--apply-final-norm")
    if args["flow_reverse"]:
        cmd.append("--flow-reverse")
    if args["use_linear_quadratic_schedule"]:
        cmd.append("--use-linear-quadratic-schedule")
    if args["use_cpu_offload"]:
        cmd.append("--use-cpu-offload")
    if args["disable_autocast"]:
        cmd.append("--disable-autocast")
    if args["reproduce"]:
        cmd.append("--reproduce")

    # If negative prompt is provided
    if negative_prompt.strip():
        cmd.extend(["--neg-prompt", negative_prompt])

    if args["save_path_suffix"]:
        cmd.extend(["--save-path-suffix", args["save_path_suffix"]])
    if args["name_suffix"]:
        cmd.extend(["--name-suffix", args["name_suffix"]])

    # Run the command
    subprocess.run(cmd, check=True, text=True)

    # Assume single .mp4 file is generated
    for filename in os.listdir(output_path):
        if filename.endswith(".mp4"):
            video_file_path = os.path.join(output_path, filename)
            fps = 8  # hypothetical fps
            duration_seconds = args["video_length"] / fps
            create_srt_file(video_file_path, prompt, duration_seconds)
            break

class FullArgDialog(simpledialog.Dialog):
    """
    Dialog for all parameters in the namespace.
    """

    fields_info = {
        "model": "Model type/name used.",
        "latent_channels": "Number of latent channels.",
        "precision": "Computation precision, e.g., bf16, fp16.",
        "rope_theta": "ROPE theta parameter.",
        "vae": "VAE model name/path.",
        "vae_precision": "VAE precision.",
        "vae_tiling": "Enable VAE tiling for large resolutions.",
        "text_encoder": "Text encoder model name.",
        "text_encoder_precision": "Text encoder precision.",
        "text_states_dim": "Dimension of text states.",
        "text_len": "Text length token limit.",
        "tokenizer": "Tokenizer name.",
        "prompt_template": "Prompt template for encoding.",
        "prompt_template_video": "Prompt template for video encoding.",
        "hidden_state_skip_layer": "Hidden state skip layer in model.",
        "apply_final_norm": "Apply final normalization in model.",
        "text_encoder_2": "Secondary text encoder.",
        "text_encoder_precision_2": "Secondary text encoder precision.",
        "text_states_dim_2": "Dimension of second text states.",
        "tokenizer_2": "Secondary tokenizer.",
        "text_len_2": "Text length for second tokenizer.",
        "denoise_type": "Denoising technique (flow, etc.).",
        "flow_shift": "Shift factor for flow.",
        "flow_reverse": "Reverse generation order.",
        "flow_solver": "Flow solver type, e.g., euler.",
        "use_linear_quadratic_schedule": "Use linear-quadratic schedule.",
        "linear_schedule_end": "End step for linear schedule.",
        "model_base": "Base directory for model weights.",
        "dit_weight": "Path to dit weight file.",
        "model_resolution": "Model resolution setting (e.g. 540p).",
        "load_key": "Key for loading model states.",
        "use_cpu_offload": "Offload model to CPU for memory savings.",
        "batch_size": "Batch size for generation.",
        "infer_steps": "Number of inference steps.",
        "disable_autocast": "Disable autocast for precision.",
        "save_path": "Path to save generated video.",
        "save_path_suffix": "Suffix to add to save path.",
        "name_suffix": "Suffix to add to output filenames.",
        "num_videos": "Number of videos to generate.",
        "video_size": "Video size (width, height).",
        "video_length": "Number of frames in the video.",
        "prompt": "Main positive text prompt.",
        "seed_type": "Seed type (auto/manual).",
        "seed": "Random seed for reproducibility.",
        "neg_prompt": "Negative prompt to avoid certain details.",
        "cfg_scale": "CFG scale for guidance.",
        "embedded_cfg_scale": "Embedded CFG scale.",
        "reproduce": "Try to reproduce deterministic results.",
        "ulysses_degree": "Ulysses degree setting.",
        "ring_degree": "Ring degree setting."
    }

    def body(self, master):
        self.title("Configure All Arguments")

        self.entries = {}
        row = 0

        for key, default_val in DEFAULTS.items():
            desc = self.fields_info.get(key, "No description provided.")
            label_txt = f"{key} (default: {default_val})\n{desc}"
            tk.Label(master, text=label_txt, justify=tk.LEFT).grid(row=row, column=0, sticky="w")

            if isinstance(default_val, bool):
                var = tk.BooleanVar(value=default_val)
                chk = tk.Checkbutton(master, variable=var)
                chk.grid(row=row, column=1, sticky="w")
                self.entries[key] = var
            else:
                entry = tk.Entry(master)
                if isinstance(default_val, list):
                    entry.insert(0, " ".join(map(str, default_val)))
                else:
                    entry.insert(0, str(default_val))
                entry.grid(row=row, column=1, sticky="w")
                self.entries[key] = entry
            row += 1

        return list(self.entries.values())[0]

    def apply(self):
        result = {}
        for key, widget in self.entries.items():
            if isinstance(widget, tk.BooleanVar):
                val = widget.get()
            else:
                val_str = widget.get().strip()
                # Attempt to parse values
                default_val = DEFAULTS[key]

                if isinstance(default_val, bool):
                    val = (val_str.lower() in ("true", "1", "yes"))
                elif isinstance(default_val, int):
                    val = int(val_str)
                elif isinstance(default_val, float):
                    val = float(val_str)
                elif isinstance(default_val, list):
                    # Expecting space-separated values
                    parts = val_str.split()
                    val = [int(p) if p.isdigit() else float(p) if p.replace('.','',1).isdigit() else p for p in parts]
                else:
                    # string or other
                    val = val_str

                # Convert special cases:
                if key == "flow_reverse" or key == "use_cpu_offload" or key == "apply_final_norm" or key == "disable_autocast" or key == "reproduce" or key == "use_linear_quadratic_schedule" or key == "vae_tiling":
                    # if originally bool, already handled above, but just in case:
                    val = True if val_str.lower() in ("true", "1", "yes") else False

                result[key] = val

        self.result = result


def get_all_args():
    root = tk.Tk()
    root.withdraw()
    dialog = FullArgDialog(root)
    args = dialog.result
    root.destroy()

    if not args:
        args = DEFAULTS
    return args


def main():
    root = tk.Tk()
    root.withdraw()

    prompt_file = select_prompt_file()
    if not prompt_file:
        root.destroy()
        return

    args = get_all_args()

    # The prompt file parsing remains the same
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

    print(f"Using seed value: {args['seed']}")

    output_dir = os.path.dirname(prompt_file)
    if not output_dir:
        output_dir = os.getcwd()

    for idx, prompt_data in enumerate(prompts, start=1):
        positive_prompt = prompt_data.get("positive")
        negative_prompt = prompt_data.get("negative")

        if not positive_prompt or not negative_prompt:
            print(f"Skipping prompt {idx}: Incomplete 'positive' or 'negative' sections.")
            continue

        summarized_positive = summarize_text(
            positive_prompt,
            max_tokens=POSITIVE_MAX_TOKENS,
            min_tokens=POSITIVE_MIN_TOKENS
        )
        summarized_negative = summarize_text(
            negative_prompt,
            max_tokens=NEGATIVE_MAX_TOKENS,
            min_tokens=NEGATIVE_MIN_TOKENS
        )

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

        try:
            generate_video(args, summarized_positive, summarized_negative, video_output_dir)
        except Exception as e:
            print(f"Error generating video for prompt '{summarized_positive}': {e}")
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
