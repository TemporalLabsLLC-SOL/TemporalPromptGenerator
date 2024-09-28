import os
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
from dotenv import load_dotenv, set_key
from huggingface_hub import InferenceClient
import requests
from pydub import AudioSegment

# Load environment variables from .env file if it exists
load_dotenv()

# Constants
ENV_FILE = '.env'
TOKEN_ENV_VAR = 'HUGGINGFACE_TOKEN'
MODEL_REPO = "m-a-p/AudioLDM2"  # Replace with the correct model repository if different

def get_huggingface_token():
    """
    Retrieves the Hugging Face token from the environment.
    If not found, prompts the user to enter it and saves it to the .env file.
    """
    token = os.getenv(TOKEN_ENV_VAR)
    if not token:
        # Initialize Tkinter
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Prompt for the token
        token = simpledialog.askstring("Hugging Face Token",
                                       "Enter your Hugging Face API Token:",
                                       show='*')
        if not token:
            messagebox.showerror("Error", "Hugging Face token is required to proceed.")
            sys.exit(1)

        # Save the token to .env
        try:
            set_key(ENV_FILE, TOKEN_ENV_VAR, token)
            messagebox.showinfo("Success", "Hugging Face token saved successfully!")
            print("Hugging Face token saved to .env")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save token to .env: {e}")
            sys.exit(1)
        finally:
            root.destroy()
    else:
        print("Hugging Face token loaded from .env")
    return token

def generate_sound(model_repo, prompt, token):
    """
    Generates sound using the specified model and prompt.
    """
    try:
        # Initialize the InferenceClient
        client = InferenceClient(repo_id=model_repo, token=token)
        print(f"InferenceClient initialized for model: {model_repo}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize Inference Client: {e}")
        sys.exit(1)

    try:
        # Generate audio using the model
        print(f"Generating sound with prompt: {prompt}")
        
        # The method name 'text_to_audio' is assumed. Replace it with the correct method if different.
        # You might need to refer to the model's documentation for the exact method.
        audio_bytes = client.text_to_audio(prompt)
        print("Sound generation successful.")

        # Save the audio data to a file
        sanitized_prompt = ''.join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"{model_repo.split('/')[-1]}_{sanitized_prompt.replace(' ', '_')}.mp3"
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        print(f"Sound effect saved as {filename}")
        messagebox.showinfo("Success", f"Sound effect saved as {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate sound: {e}")
        sys.exit(1)

def main():
    print("Launching the AudioLDM2 Sound Generator...")
    token = get_huggingface_token()
    prompt = get_text_prompt()
    generate_sound(MODEL_REPO, prompt, token)

def get_text_prompt():
    """
    Prompts the user to enter a text prompt for sound generation.
    """
    root = tk.Tk()
    root.withdraw()
    prompt = None
    while not prompt:
        prompt = simpledialog.askstring("Input", "Enter a description for the sound effect:")
        if prompt is None:
            messagebox.showerror("Error", "A text prompt is required.")
            sys.exit(1)
        prompt = prompt.strip()
        if not prompt:
            messagebox.showwarning("Empty Input", "Please enter a non-empty prompt.")
            prompt = None
    root.destroy()
    print(f"Prompt entered: {prompt}")
    return prompt

if __name__ == "__main__":
    main()
