import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import sys
import shlex
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("audioldm2_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, default_options):
        super().__init__(parent)
        self.title("Audioldm2 Configuration")
        self.parent = parent
        self.result = None
        self.create_widgets(default_options)
        self.grab_set()
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def create_widgets(self, options):
        padding = {'padx': 10, 'pady': 5}

        # Create a frame for all widgets
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Mode Selection
        ttk.Label(frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, **padding)
        self.mode_var = tk.StringVar(value=options.get('mode', 'generation'))
        mode_combo = ttk.Combobox(frame, textvariable=self.mode_var, state="readonly")
        mode_combo['values'] = ['generation', 'sr_inpainting']
        mode_combo.grid(row=0, column=1, **padding)
        mode_combo.bind("<<ComboboxSelected>>", self.toggle_inpainting)

        # Original Audio File (only for sr_inpainting)
        self.audio_file_label = ttk.Label(frame, text="Original Audio File:")
        self.audio_file_entry = ttk.Entry(frame, width=40)
        self.audio_file_button = ttk.Button(frame, text="Browse", command=self.browse_audio_file)

        if options.get('mode') == 'sr_inpainting':
            self.audio_file_label.grid(row=1, column=0, sticky=tk.W, **padding)
            self.audio_file_entry.grid(row=1, column=1, sticky=tk.W, **padding)
            self.audio_file_button.grid(row=1, column=2, sticky=tk.W, **padding)

        # Model Selection
        ttk.Label(frame, text="Model:").grid(row=2, column=0, sticky=tk.W, **padding)
        self.model_var = tk.StringVar(value=options.get('model_name', 'audioldm2-full'))
        model_combo = ttk.Combobox(frame, textvariable=self.model_var, state="readonly")
        model_combo['values'] = [
            'audioldm2-full',
            'audioldm2-music-665k',
            'audioldm2-full-large-1150k',
            'audioldm2-speech-ljspeech',
            'audioldm2-speech-gigaspeech'
        ]
        model_combo.grid(row=2, column=1, columnspan=2, sticky=tk.W, **padding)

        # Device Selection
        ttk.Label(frame, text="Device:").grid(row=3, column=0, sticky=tk.W, **padding)
        self.device_var = tk.StringVar(value=options.get('device', 'auto'))
        device_combo = ttk.Combobox(frame, textvariable=self.device_var, state="readonly")
        device_combo['values'] = ['cpu', 'cuda', 'mps', 'auto']
        device_combo.grid(row=3, column=1, columnspan=2, sticky=tk.W, **padding)

        # Batch Size
        ttk.Label(frame, text="Batch Size:").grid(row=4, column=0, sticky=tk.W, **padding)
        self.batchsize_var = tk.IntVar(value=options.get('batchsize', 1))
        batchsize_spin = ttk.Spinbox(frame, from_=1, to=100, textvariable=self.batchsize_var, width=5)
        batchsize_spin.grid(row=4, column=1, sticky=tk.W, **padding)

        # DDIM Steps
        ttk.Label(frame, text="DDIM Steps:").grid(row=5, column=0, sticky=tk.W, **padding)
        self.ddim_steps_var = tk.IntVar(value=options.get('ddim_steps', 30))
        ddim_steps_spin = ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.ddim_steps_var, width=5)
        ddim_steps_spin.grid(row=5, column=1, sticky=tk.W, **padding)

        # Duration
        ttk.Label(frame, text="Duration (seconds):").grid(row=6, column=0, sticky=tk.W, **padding)
        self.duration_var = tk.DoubleVar(value=options.get('duration', 6.0))
        duration_spin = ttk.Spinbox(frame, from_=0.1, to=600, increment=0.1, textvariable=self.duration_var, width=7)
        duration_spin.grid(row=6, column=1, sticky=tk.W, **padding)

        # Guidance Scale
        ttk.Label(frame, text="Guidance Scale:").grid(row=7, column=0, sticky=tk.W, **padding)
        self.guidance_scale_var = tk.DoubleVar(value=options.get('guidance_scale', 7.5))
        guidance_scale_spin = ttk.Spinbox(frame, from_=0.0, to=100.0, increment=0.1, textvariable=self.guidance_scale_var, width=7)
        guidance_scale_spin.grid(row=7, column=1, sticky=tk.W, **padding)

        # Number of Candidates
        ttk.Label(frame, text="Candidates per Text:").grid(row=8, column=0, sticky=tk.W, **padding)
        self.n_candidate_var = tk.IntVar(value=options.get('n_candidate_gen_per_text', 3))
        n_candidate_spin = ttk.Spinbox(frame, from_=1, to=100, textvariable=self.n_candidate_var, width=5)
        n_candidate_spin.grid(row=8, column=1, sticky=tk.W, **padding)

        # Seed
        ttk.Label(frame, text="Seed:").grid(row=9, column=0, sticky=tk.W, **padding)
        self.seed_var = tk.StringVar(value=options.get('seed', '12345'))
        seed_entry = ttk.Entry(frame, textvariable=self.seed_var)
        seed_entry.grid(row=9, column=1, sticky=tk.W, **padding)

        # Transcription (Optional)
        ttk.Label(frame, text="Transcription (Optional):").grid(row=10, column=0, sticky=tk.W, **padding)
        self.transcription_var = tk.StringVar(value=options.get('transcription', ''))
        transcription_entry = ttk.Entry(frame, textvariable=self.transcription_var, width=40)
        transcription_entry.grid(row=10, column=1, columnspan=2, sticky=tk.W, **padding)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=11, column=0, columnspan=3, pady=10)
        ok_button = ttk.Button(button_frame, text="OK", command=self.on_ok)
        ok_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=5)

    def toggle_inpainting(self, event):
        if self.mode_var.get() == 'sr_inpainting':
            self.audio_file_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
            self.audio_file_entry.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
            self.audio_file_button.grid(row=1, column=2, sticky=tk.W, padx=10, pady=5)
        else:
            self.audio_file_label.grid_remove()
            self.audio_file_entry.grid_remove()
            self.audio_file_button.grid_remove()

    def browse_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select the original audio file for inpainting",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        if file_path:
            self.audio_file_entry.delete(0, tk.END)
            self.audio_file_entry.insert(0, file_path)

    def on_ok(self):
        # Validate inputs
        if self.mode_var.get() == 'sr_inpainting' and not self.audio_file_entry.get():
            messagebox.showerror("Input Error", "Please select an original audio file for inpainting.")
            return
        if self.seed_var.get():
            try:
                int(self.seed_var.get())
            except ValueError:
                messagebox.showerror("Input Error", "Seed must be an integer.")
                return
        self.result = {
            'mode': self.mode_var.get(),
            'file_path': self.audio_file_entry.get() if self.mode_var.get() == 'sr_inpainting' else None,
            'model_name': self.model_var.get(),
            'device': self.device_var.get(),
            'batchsize': self.batchsize_var.get(),
            'ddim_steps': self.ddim_steps_var.get(),
            'duration': self.duration_var.get(),
            'guidance_scale': self.guidance_scale_var.get(),
            'n_candidate_gen_per_text': self.n_candidate_var.get(),
            'seed': int(self.seed_var.get()) if self.seed_var.get() else None,
            'transcription': self.transcription_var.get()
        }
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

def select_file(prompt="Select the prompt list file"):
    file_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    return file_path

def parse_prompts(file_path):
    prompts = []
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        sections = content.split('--------------------')
        for section in sections:
            lines = section.strip().split('\n')
            positive = ""
            negative = ""
            for line in lines:
                if line.lower().startswith('positive:'):
                    positive = line[len('positive:'):].strip()
                elif line.lower().startswith('negative:'):
                    negative = line[len('negative:'):].strip()
            if positive:
                prompts.append({'positive': positive, 'negative': negative})
        return prompts
    except Exception as e:
        messagebox.showerror("Error", f"Failed to parse prompt list: {e}")
        logging.error(f"Failed to parse prompt list: {e}")
        sys.exit(1)

def sanitize_filename(name):
    # Remove or replace characters that are invalid in filenames
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def main():
    root = tk.Tk()
    root.withdraw()

    # Step 1: Select the prompt list file
    prompt_list_path = select_file("Select the audio prompt list (.txt) file")
    if not prompt_list_path:
        messagebox.showinfo("Info", "No file selected. Exiting.")
        sys.exit(0)

    # Step 2: Select the save directory
    default_save_dir = os.path.dirname(os.path.abspath(prompt_list_path))
    save_dir = filedialog.askdirectory(
        title="Select the directory to save outputs",
        initialdir=default_save_dir
    )
    if not save_dir:
        messagebox.showinfo("Info", "No save directory selected. Exiting.")
        sys.exit(0)

    # Step 3: Gather options via single popup
    default_options = {
        'mode': 'generation',
        'model_name': 'audioldm2-full',
        'device': 'auto',
        'batchsize': 1,
        'ddim_steps': 30,
        'duration': 6.0,
        'guidance_scale': 7.5,
        'n_candidate_gen_per_text': 3,
        'seed': '12345',
        'transcription': ''
    }

    # Initialize and display the settings dialog
    settings_dialog = SettingsDialog(root, default_options)
    root.wait_window(settings_dialog)

    options = settings_dialog.result
    if options is None:
        messagebox.showinfo("Info", "Operation cancelled. Exiting.")
        sys.exit(0)

    # Step 4: Parse the prompt list
    prompts = parse_prompts(prompt_list_path)
    if not prompts:
        messagebox.showerror("Error", "No valid prompts found in the prompt list.")
        sys.exit(1)

    # Step 5: Execute audioldm2 for each prompt
    for idx, prompt in enumerate(prompts, start=1):
        positive_prompt = prompt['positive']
        negative_prompt = prompt.get('negative', '').strip()

        # Split the positive prompt into individual sounds (assuming separated by commas)
        sounds = [sound.strip() for sound in positive_prompt.split(',') if sound.strip()]
        if not sounds:
            logging.warning(f"No valid sounds found in prompt {idx}. Skipping.")
            continue

        # Create Video_X directory
        video_save_path = os.path.join(save_dir, f"Video_{idx}")
        os.makedirs(video_save_path, exist_ok=True)

        logging.info(f"Processing Prompt {idx}: {positive_prompt}")
        logging.info(f"Number of sounds to generate: {len(sounds)}")

        for sound_idx, sound_text in enumerate(sounds, start=1):
            # Sanitize sound text for filename
            sanitized_sound = sanitize_filename(sound_text)
            desired_output_filename = f"audio_{sound_idx}_{sanitized_sound}.wav"
            final_output_path = os.path.join(video_save_path, desired_output_filename)

            # Prepare the audioldm2 command
            cmd = [
                "audioldm2",
                "--mode", options['mode'],
                "--text", sound_text,
                "--save_path", video_save_path,  # Save to directory, not file
                "--model_name", options['model_name'],
                "--device", options['device'],
                "--batchsize", str(options['batchsize']),
                "--ddim_steps", str(options['ddim_steps']),
                "--duration", str(options['duration']),
                "--guidance_scale", str(options['guidance_scale']),
                "--n_candidate_gen_per_text", str(options['n_candidate_gen_per_text'])
            ]

            # Add seed if provided
            if options['seed'] is not None:
                cmd += ["--seed", str(options['seed'])]

            # Add transcription if provided
            if options.get('transcription'):
                cmd += ["--transcription", options['transcription']]

            # Add negative prompt if audioldm2 supports it and it's valid
            if negative_prompt:
                # Ensure negative_prompt does not contain only punctuation or invalid values
                if re.search(r'\w', negative_prompt):
                    cmd += ["--negative", negative_prompt]
                else:
                    logging.warning(f"Invalid negative_prompt for Video {idx}, Sound {sound_idx}: '{negative_prompt}'. Skipping '--negative'.")

            # Display the command being run
            command_str = ' '.join(shlex.quote(arg) for arg in cmd)
            logging.info(f"Running command for Video {idx}, Sound {sound_idx}: {command_str}")

            try:
                # Run the audioldm2 command
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                logging.info(f"Video {idx}, Sound {sound_idx} processed successfully.")

                # Locate the latest directory created by audioldm2 within video_save_path
                subdirs = [d for d in os.listdir(video_save_path) if os.path.isdir(os.path.join(video_save_path, d))]
                if not subdirs:
                    raise FileNotFoundError("No subdirectories found after running audioldm2.")
                latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(video_save_path, d)))
                generated_audio_dir = os.path.join(video_save_path, latest_subdir)

                # Find the generated audio file
                generated_files = [f for f in os.listdir(generated_audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
                if not generated_files:
                    raise FileNotFoundError("No audio files found in the generated directory.")
                generated_file = os.path.join(generated_audio_dir, generated_files[0])

                # If the generated file is MP3, convert to WAV
                if generated_file.lower().endswith('.mp3'):
                    wav_file = os.path.splitext(generated_file)[0] + '.wav'
                    convert_cmd = ["ffmpeg", "-y", "-i", generated_file, wav_file]
                    logging.info(f"Converting {generated_file} to WAV format.")
                    subprocess.run(convert_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    os.remove(generated_file)  # Remove the original MP3 file
                    generated_file = wav_file
                    logging.info(f"Converted to {wav_file} and removed the MP3 file.")

                # Rename and move the generated file to the desired output path
                os.rename(generated_file, final_output_path)
                logging.info(f"Saved audio to {final_output_path}")

                # Optionally, remove the timestamped subdirectory if empty
                try:
                    os.rmdir(generated_audio_dir)
                    logging.info(f"Removed empty directory {generated_audio_dir}")
                except OSError:
                    logging.warning(f"Directory {generated_audio_dir} is not empty and cannot be removed.")

            except subprocess.CalledProcessError as e:
                logging.error(f"Error processing Video {idx}, Sound {sound_idx}: {e.stderr}")
                messagebox.showerror(
                    "Error",
                    f"Failed to process Video {idx}, Sound {sound_idx}.\nError: {e.stderr}"
                )
                # Continue with the next sound
                continue
            except Exception as ex:
                logging.error(f"Unexpected error processing Video {idx}, Sound {sound_idx}: {ex}")
                messagebox.showerror(
                    "Error",
                    f"An unexpected error occurred while processing Video {idx}, Sound {sound_idx}.\nError: {ex}"
                )
                continue

    messagebox.showinfo("Success", "All sounds have been processed and saved as WAV files.")
    sys.exit(0)

if __name__ == "__main__":
    main()

