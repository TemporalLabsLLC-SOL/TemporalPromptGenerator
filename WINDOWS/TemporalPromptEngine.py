import os
import sys
import datetime
import json
import subprocess
from dotenv import load_dotenv
import openai
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Menu, ttk
from tkinter.scrolledtext import ScrolledText
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips
from pydub import AudioSegment
from PIL import Image, ImageTk
import threading
import requests
from io import BytesIO
import pyperclip
import webbrowser
import re
import random
import platform
import socket
import time
import scipy
import torch
from diffusers import AudioLDM2Pipeline
from pathlib import Path
import shutil
from moviepy.video import fx as vfx



# Load environment variables from .env file
load_dotenv()

# Initialize global variables
# Initialize global variables
OUTPUT_DIRECTORY = os.path.join(os.getcwd(), "prompts_output")
COMFYUI_PROMPTS_FOLDER = os.getenv("COMFYUI_PROMPTS_FOLDER")
MAX_CHAR_LIMIT = 400  # Maximum characters allowed for prompts
MAX_PROMPTS = 250  # Maximum number of prompt sets allowed
DEFAULT_PROMPTS = 5  # Default number of prompt sets
LAST_USED_DIRECTORY = os.getenv("LAST_USED_DIRECTORY") or os.getcwd()
REQUIRED_MODEL = "llama3.2"  # Ensure you use the local model name
FORMATTED_PROMPTS_FILENAME = "formatted_prompts.txt"  # Ensure this is defined before use
RAW_PROMPTS_FILENAME = "RAW_prompts.txt"  # Ensure this is defined before use
# Define OLLAMA_API_URL globally so it can be used elsewhere
ollama_port = 11434  # Set to your known port
global OLLAMA_API_URL
OLLAMA_API_URL = f"http://localhost:{ollama_port}"  # Removed '/api' if unnecessary
repo_id = "cvssp/audioldm2-large"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


SETTINGS_FILE = "settings.json"
def ensure_ollama_server_and_model(model_name="llama3.2"):
    start_or_find_ollama_server()
    ensure_model_available(model_name)
# Save settings to .env file for persistence
def start_or_find_ollama_server():
    ollama_port = find_ollama_port()
    if ollama_port:
        print(f"Found Ollama server running on port {ollama_port}.")
        global OLLAMA_API_URL
        OLLAMA_API_URL = f"http://localhost:{ollama_port}/api/generate"
    else:
        ollama_port = find_available_port()
        start_ollama_server(ollama_port)
        wait_for_ollama_server()

def detect_gpu():
    """
    Detects if a GPU is available using torch and returns a boolean value.
    If a GPU is found, it returns True; otherwise, it returns False.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU detected.")
            return True
        else:
            print("No GPU detected.")
            return False
    except ImportError:
        print("Torch is not installed, assuming no GPU.")
        return False
        
def clean_prompt_text(raw_prompts):
    lines = raw_prompts.splitlines()
    cleaned_prompts = []
    positive_prompt = ""
    negative_prompt = ""

    for line in lines:
        line = line.strip()
        if line.startswith("positive:"):
            positive_prompt = line.replace("positive:", "positive:").strip()
        elif line.startswith("negative:"):
            negative_prompt = line.strip()
        if positive_prompt and negative_prompt:
            cleaned_prompts.append(f"{positive_prompt}\n{negative_prompt}")
            positive_prompt, negative_prompt = "", ""
    return "\n--------------------\n".join(cleaned_prompts)
def remove_unwanted_headers(cleaned_prompts):
    formatted_prompts = []
    lines = cleaned_prompts.splitlines()
    for line in lines:
        if not line.startswith("Title") and not line.startswith("Options") and not line.startswith("Theme"):
            formatted_prompts.append(line)
    return "\n".join(formatted_prompts)
# Utility function to save data to a file with error handling
def save_to_file(data, file_path):
    """
    Save data to a file with error handling.

    Args:
        data (str): The data to be saved.
        file_path (str): The path to the file.
    """
    if not file_path:
        print(f"Error: Invalid file path {file_path}")
        return

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}: {e}")

def run_audioldm2(prompt_text, output_filename, index):
    try:
        # Construct the AudioLDM2 command
        command = [
            sys.executable, '-m', 'audioldm2',
            '-t', prompt_text,
            '-s', output_filename,
            '--model_name', model_name,
            '--device', device,
            '--ddim_steps', str(ddim_steps),
            '--guidance_scale', str(guidance_scale),
            '--duration', str(duration),
            '--n_candidate_gen_per_text', str(n_candidate_gen_per_text),
            '--seed', str(seed),
            '--mode', 'generation'
        ]

        print(f"[AudioLDM2] Executing command: {' '.join(command)}")
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy()  # Inherit environment variables
        )
        print(f"[AudioLDM2] Generated sound effect for prompt {index}: {prompt_text}")
    except subprocess.CalledProcessError as e:
        print(f"[AudioLDM2 Error] Failed to generate sound effect for prompt {index}: {e.stderr}")
        messagebox.showerror("AudioLDM2 Error", f"Failed to generate sound effect for prompt {index}.\nError: {e.stderr}")
    except Exception as e:
        print(f"[AudioLDM2 Error] Unexpected error for prompt {index}: {e}")
        messagebox.showerror("AudioLDM2 Error", f"Failed to generate sound effect for prompt {index}.\nError: {e}")
    finally:
        # Update progress bar
        progress_bar['value'] = index
        progress_window.update_idletasks()

        
def handle_multiple_json_objects(response_text):
    try:
        json_objects = response_text.splitlines()
        for json_object in json_objects:
            try:
                data = json.loads(json_object)
                # Process only if 'response' field exists
                if "response" in data:
                    print("Parsed JSON object:", data["response"])
                    # Further processing if needed
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON object: {e}")
    except Exception as e:
        print(f"General error processing response: {e}")


def find_available_port(start_port=11435, max_attempts=10):
    """
    Find an available port starting from the given start_port.
    Tries up to max_attempts ports.
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    raise Exception("No available ports found.")

def check_if_port_in_use(port):
    """Checks if the given port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('localhost', port))
        return result == 0
        
def find_ollama_port():
    """
    Check if Ollama is running on the default port.
    If not, return the port where Ollama is running or None if not found.
    """
    for port in range(11420, 11435):  # Scan potential ports
        if check_if_port_in_use(port):
            return port
    return None
        
def detect_gpu():
    """
    Detects if a GPU is available using torch and returns a boolean value.
    If a GPU is found, it returns True; otherwise, it returns False.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU detected.")
            return True
        else:
            print("No GPU detected.")
            return False
    except ImportError:
        print("Torch is not installed, assuming no GPU.")
        return False

def disable_gpu_option_if_no_gpu(gpu_var, device_combobox):
    """
    Disables the GPU option if no GPU is detected. Otherwise, it automatically selects GPU.
    Args:
        gpu_var: The tkinter variable for the device selection.
        device_combobox: The tkinter combobox for selecting the device (CPU/GPU).
    """
    if detect_gpu():
        gpu_var.set("cuda")  # Auto-select GPU
        device_combobox.config(state="readonly")  # Lock the selection to GPU
    else:
        gpu_var.set("cpu")
        device_combobox.set("cpu")
        device_combobox.config(state="disabled")  # Disable GPU option if not available

def save_settings():
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}\n")
        f.write(f"COMFYUI_PROMPTS_FOLDER={COMFYUI_PROMPTS_FOLDER}\n")
        f.write(f"LAST_USED_DIRECTORY={LAST_USED_DIRECTORY}\n")
    print("Settings have been saved to .env file.")

def save_prompt_text(filename, prompt_text):
    # Ensure the directory exists
    if not self.output_folder:
        self.set_output_directory()
    if not self.output_folder:
        return

    text_save_path = os.path.join(self.output_folder, "Prompts")
    os.makedirs(text_save_path, exist_ok=True)  # Ensure the directory is created

    # Combine the chosen directory with the filename
    full_save_path = os.path.join(text_save_path, filename)

    # Save the prompt text to the selected directory
    with open(full_save_path, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"Prompt text saved to {full_save_path}")

    
def format_prompt(prompt_data):
    """
    Formats a single prompt set into the final prompt list format.
    """
    # Extracting key details from the prompt data
    title = prompt_data.get('Title', 'Untitled')
    description = prompt_data.get('Description', 'No description provided.')
    time_of_day = prompt_data.get('Time of day', '00:00')
    theme = prompt_data.get('Theme', 'Adventure')
    lighting = prompt_data.get('Lighting', 'Natural Light')
    shot_composition = prompt_data.get('Shot composition', 'Rule of Thirds')
    camera_movement = prompt_data.get('Camera movement', 'Pan')
    art_style = prompt_data.get('Art style', 'Realism')
    lens = prompt_data.get('Lens', 'Wide Angle')

    # Constructing positive prompt
    positive_prompt = (
        f"Create a {art_style.lower()} {theme.lower()} video capturing {description.lower()}. "
        f"Adopt the {shot_composition.lower()} to highlight the journey's thrill. "
        f"The {lighting.lower()} from the {time_of_day.lower()} should guide their paths. "
        f"Use a {lens.lower()} lens for a broad view and {camera_movement.lower()} "
        "to provide a sweeping view of the landscape."
    )

    # Constructing negative prompt
    negative_prompt = (
        f"Avoid unnatural lighting or closed shots which may distort the {theme.lower()} theme. "
        f"Do not ignore the {shot_composition.lower()} which may unbalance the shot composition, "
        f"and refrain from static or zoomed-in views that fail to capture the whole scene."
    )

    # Return the formatted prompt
    return f"positive: {positive_prompt}\nnegative: {negative_prompt}\n--------------------"
    return f"positive: {positive_prompt}\nnegative: {negative_prompt}\n--------------------"

# Function to format the raw prompts into the required structure

def format_prompts_for_output(raw_prompts):
    """
    Takes the raw prompts and formats them into the required positive/negative structure.
    """
    formatted_prompts = ""
    for index, prompt in enumerate(raw_prompts, start=1):
        # Ensure that each prompt is a dictionary or structured data
        if isinstance(prompt, dict):
            positive_prompt = f"positive: {prompt.get('positive', 'N/A')}"
            negative_prompt = f"negative: {prompt.get('negative', 'N/A')}"
        else:
            # If prompt is not a dict, handle it as a string or simple type
            positive_prompt = f"positive: {prompt}"
            negative_prompt = "negative: N/A"

        separator = "--------------------"
        formatted_prompts += positive_prompt + negative_prompt + separator
    return formatted_prompts


def format_and_overwrite_prompts_file(file_path):
    """
    Formats the raw prompts from a JSON file and overwrites the file with formatted prompts.
    
    Args:
        file_path (str): The full path to the raw prompts JSON file.
    """
    try:
        # Check if file_path is a directory
        if os.path.isdir(file_path):
            error_message = f"Expected a file path but received a directory: {file_path}"
            print(error_message)
            messagebox.showerror("Invalid File Path", error_message)
            return
        
        # Ensure the file exists
        if not os.path.exists(file_path):
            error_message = f"The file does not exist: {file_path}"
            print(error_message)
            messagebox.showerror("File Not Found", error_message)
            return
        
        # Read the raw prompts from the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        
        formatted_prompts = []
        for prompt_data in raw_data:
            formatted_prompts.append(format_prompt(prompt_data))
        
        # Join all formatted prompts with a divider
        formatted_prompts_str = "\n--------------------\n".join(formatted_prompts)
        
        # Save the formatted prompts back to the same file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(formatted_prompts_str)
        
        print(f"Formatted prompts successfully saved to {file_path}")
        messagebox.showinfo("Formatting Successful", f"Formatted prompts have been saved to:\n{file_path}")
    
    except PermissionError:
        error_message = f"Permission denied: Unable to write to {file_path}.\nPlease check your file permissions."
        print(error_message)
        messagebox.showerror("Permission Denied", error_message)
    
    except FileNotFoundError:
        error_message = f"File not found: The file does not exist at {file_path}."
        print(error_message)
        messagebox.showerror("File Not Found", error_message)
    
    except json.JSONDecodeError as e:
        error_message = f"JSON Decode Error: {e}"
        print(error_message)
        messagebox.showerror("JSON Error", error_message)
    
    except Exception as e:
        error_message = f"An error occurred while formatting prompts: {e}"
        print(error_message)
        messagebox.showerror("Formatting Error", error_message)


def save_raw_prompts(raw_prompts, file_path):
    """
    Saves raw prompt data to a specified file in JSON format.

    Args:
        raw_prompts (list): A list of dictionaries containing raw prompt data.
        file_path (str): The file path where the prompts will be saved.
    """
    try:
        # Check if file_path is a directory
        if os.path.isdir(file_path):
            # Assign a default filename if none is provided
            file_path = os.path.join(file_path, RAW_PROMPTS_FILENAME)
            print(f"No filename provided. Using default filename: {RAW_PROMPTS_FILENAME}")

        # Ensure the output directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured the directory exists: {directory}")

        # Save the raw prompts as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(raw_prompts, f, indent=4)
        
        print(f"Raw prompts successfully saved to {file_path}")
        messagebox.showinfo("Save Successful", f"Raw prompts have been saved to:\n{file_path}")
    
    except PermissionError:
        error_message = f"Permission denied: Unable to write to {file_path}.\nPlease check your directory permissions."
        print(error_message)
        messagebox.showerror("Permission Denied", error_message)
    
    except FileNotFoundError:
        error_message = f"File not found: The directory does not exist for {file_path}."
        print(error_message)
        messagebox.showerror("File Not Found", error_message)
    
    except Exception as e:
        error_message = f"Failed to save raw prompts to {file_path}: {e}"
        print(error_message)
        messagebox.showerror("Save Error", error_message)

    # Example usage
    raw_prompt_data = [
        {
            "Title": "Midnight Expedition",
            "Description": "a group of explorers on a test mission, navigating through a dense forest at midnight in the 1900s.",
            "Time of day": "00:00 - Midnight",
            "Theme": "Adventure",
            "Lighting": "Natural Light",
            "Shot composition": "Rule of Thirds",
            "Camera movement": "Pan",
            "Art style": "Realism",
            "Lens": "Wide Angle"
        },
        # Add more prompts here
    ]

    # Define the output directory and file name separately
    output_directory = os.path.join(OUTPUT_DIRECTORY, "raw_prompts")
    file_name = RAW_PROMPTS_FILENAME  # "RAW_prompts.json"
    raw_prompts_file_path = os.path.join(output_directory, file_name)

    # Step 1: Save raw prompts to file
    save_raw_prompts(raw_prompt_data, raw_prompts_file_path)

    # Step 2: Format the prompts and overwrite the file
    format_and_overwrite_prompts_file(raw_prompts_file_path)


def format_prompts_to_file(prompts_data, OUTPUT_DIRECTORY):
    """
    Formats the video prompts from raw data and writes them into the final prompt list format.
    """
    try:
        formatted_prompts = []
        for prompt_data in prompts_data:  # Assume prompts_data is a list of dictionaries
            formatted_prompts.append(format_prompt(prompt_data))
        
        # Writing the formatted prompts to the file
        with open(OUTPUT_DIRECTORY, 'w') as output_file:
            output_file.write("\n".join(formatted_prompts))

        print(f"Formatted prompts saved to {OUTPUT_DIRECTORY}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def format_prompts_from_raw_text(raw_text):
    """
    Takes the raw response text and formats it into positive/negative structure.
    """
    formatted_prompts = ""
    prompt_sections = raw_text.split("")

    for section in prompt_sections:
        if section.strip() == "": 
            continue
        
        # Extract prompt content and format into positive/negative
        positive_prompt = f"positive: {section}"
        negative_prompt = "negative: "
        separator = "--------------------"
        
        formatted_prompts += positive_prompt + negative_prompt + separator

    return formatted_prompts


def format_and_save_prompts(raw_prompt_data, output_directory=OUTPUT_DIRECTORY, file_name=FORMATTED_PROMPTS_FILENAME):
    """
    Formats the prompts and saves them into a file.

    Args:
        raw_prompt_data (list): A list of dictionaries with raw prompt data.
        output_directory (str): The directory where the file will be saved.
        file_name (str): The name of the file.
    """
    if not raw_prompt_data:
        print("Error: No raw prompts to format.")
        return

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    formatted_prompts = []

    # Format each prompt in the list
    for prompt_data in raw_prompt_data:
        formatted_prompt = format_prompt(prompt_data)  # Assuming format_prompt is defined
        formatted_prompts.append(formatted_prompt)

    # Join all formatted prompts with a divider
    formatted_prompts_str = "\n--------------------\n".join(formatted_prompts)

    # Construct full file path
    file_path = os.path.join(output_directory, file_name)

    # Save the formatted prompts to file
    save_to_file(formatted_prompts_str, file_path)  # Assuming save_to_file is defined  
def extract_video_prompts(api_response):
    # This splits prompts by "**" and works from the JSON's response text
    response_text = api_response["response"]
    prompt_list = response_text.split("**")[1:]  # Skip initial part, start from numbered prompts

    # Extract and format video prompts
    video_prompts = []
    for i in range(0, len(prompt_list), 2):
        title = prompt_list[i].strip()
        details = prompt_list[i + 1].strip()
        full_prompt = f"{title}\n{details}"
        formatted_prompt = format_prompt(full_prompt)
        video_prompts.append(formatted_prompt)
    
    return video_prompts
    
def extract_audio_prompts(api_response):
    # This is a placeholder implementation; customize as per your API structure
    audio_prompts = ["Craft an immersive soundscape for each adventure prompt."]
    formatted_audio_prompts = [format_prompt(prompt, is_audio=True) for prompt in audio_prompts]
    return formatted_audio_prompts
    
def save_prompts_to_file(video_prompts, audio_prompts, video_file_path, audio_file_path):
    try:
        # Get the directory paths from the file paths
        video_dir = os.path.dirname(video_file_path)
        audio_dir = os.path.dirname(audio_file_path)

        # Check if the directories exist, if not, create them
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        # Writing the video prompts to the video file
        with open(video_file_path, 'w') as video_file:
            video_file.writelines(video_prompts)
        
        # Writing the audio prompts to the audio file
        with open(audio_file_path, 'w') as audio_file:
            audio_file.writelines(audio_prompts)

        print(f"Prompts saved to {video_file_path} and {audio_file_path} successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def load_prompt_file(file_path):
    """
    Load a text file with proper encoding handling.
    """
    try:
        # Try reading with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            # Fallback to ISO-8859-1 encoding if UTF-8 fails
            with open(file_path, "r", encoding="ISO-8859-1") as file:
                return file.read()
        except UnicodeDecodeError as e:
            print(f"Error: Unable to decode file {file_path}: {e}")
            messagebox.showerror("Encoding Error", f"Failed to decode file: {file_path}.")
            return None

def set_output_directory():
    self.output_folder = filedialog.askdirectory(title="Select Output Directory")
    if not self.output_folder:
        messagebox.showwarning("Directory Selection", "No directory selected. Please select a valid directory.")
        return
    print(f"Output directory set to: {self.output_folder}")


# Function to set the ComfyUI INSPIRE prompts folder
def set_comfyui_prompts_folder():
    global COMFYUI_PROMPTS_FOLDER
    directory = filedialog.askdirectory(title="Select ComfyUI INSPIRE Prompts Folder")
    if directory:
        COMFYUI_PROMPTS_FOLDER = directory
        save_settings()
    else:
        messagebox.showwarning("ComfyUI Prompts Folder", "No directory selected.")

# Function to generate video or audio prompts with strict format compliance using OpenAI

def generate_prompts(input_concept, prompt_type, video_options, model_name=REQUIRED_MODEL):
    """
    Generate refined, natural language prompts for video or audio generation models, incorporating all selected options.
    Each prompt will maintain the format of positive and negative aspects while being cohesive and detailed.

    Parameters:
    - input_concept: The central idea or theme for the prompts.
    - prompt_type: Either 'video' or 'audio'.
    - video_options: Dictionary of video options (e.g., decade, camera type, lighting, etc.)
    - model_name: The model used for generating prompts (default is REQUIRED_MODEL).
    """

    if not wait_for_ollama_server():
        print("Ollama server is not responding.")
        return None

    # Ensure all options are treated as strings
    decade = str(video_options.get('decade', 'unspecified era'))
    theme = str(video_options.get('theme', 'general theme'))
    art_style = str(video_options.get('art_style', 'standard art style'))
    lighting = str(video_options.get('lighting', 'natural lighting'))
    framing = str(video_options.get('framing', 'balanced framing'))
    camera_movement = str(video_options.get('camera_movement', 'steady camera'))
    shot_composition = str(video_options.get('shot_composition', 'simple composition'))
    time_of_day = str(video_options.get('time_of_day', 'daytime'))
    camera = str(video_options.get('camera', 'a cinematic camera'))
    lens = str(video_options.get('lens', 'a regular lens'))
    resolution = str(video_options.get('resolution', 'standard resolution'))

    # Build prompt for video generation
    if prompt_type == 'video':
        # Create a cohesive and natural prompt for video generation
        system_prompt = f"""
        You are tasked with generating cinematic video prompts based on the concept: '{input_concept}'.
        The prompts should describe scenes in rich visual detail, ensuring that each includes positive and negative descriptions of the visuals.
        
        Incorporate the following settings into the video prompts:

        - Set the scene in the {decade}, capturing the essence and atmosphere of that time.
        - Reflect the overall {theme}, making it a central part of the visuals and tone.
        - Apply {art_style} to the visual elements, enhancing the aesthetic of each shot.
        - Use {lighting} to create mood and visual impact, appropriate for the scenes.
        - Ensure {framing} that complements the composition of each shot, focusing on clear, balanced visuals.
        - Include {camera_movement} to add dynamic motion to the scenes, creating fluid transitions.
        - Frame each shot with {shot_composition}, aligning the visual elements for maximum effect.
        - Set the scene during {time_of_day}, ensuring the visuals reflect the appropriate lighting conditions.
        - Use {camera} with {lens} to shoot the scenes, emphasizing the cinematic quality.
        - Ensure the video is shot in {resolution}, maintaining clarity and detail in all scenes.

        Each generated prompt should follow this format:

        positive: Describe the positive aspects of the scene or shot, highlighting the most important visual and cinematic details.
        negative: Describe what to avoid showing or exhibiting in the scene or shot, focusing on preventing common cinematic mistakes.
        --------------------

        Make sure the prompts are specific, cohesive, and visually compelling, and that all selected settings are reflected in each prompt.
        """
    elif prompt_type == 'audio':
        system_prompt = f"""
        You are tasked with generating sound prompts that complement the visual concept: '{input_concept}'.
        Each sound prompt must follow this format, focusing on specific sound details:

        positive: Describe the positive aspects of the soundscape, focusing on natural sound elements (e.g., rain, birds, footsteps).
        negative: Describe what to avoid in the soundscape, ensuring there are no distracting or out-of-place noises.
        --------------------

        Ensure that the sounds align with the visual scenes, reflecting the selected settings and overall atmosphere.
        """

    else:
        print("Invalid prompt type.")
        return None

    try:
        # Construct the Ollama API payload
        payload = {
            "model": model_name,
            "prompt": system_prompt,
            "max_tokens": 1500
        }
        headers = {'Content-Type': 'application/json'}

        # Send request to Ollama's local API
        response = requests.post(f"{OLLAMA_API_URL}", headers=headers, json=payload)

        # Print the raw response to inspect for issues
        print("Raw response:", response.text)

        if response.status_code == 200:
            try:
                # Parse the JSON response
                generated_prompts = response.json().get('output', '').strip()
                print(f"Generated Prompts:\n{generated_prompts}")
                return generated_prompts
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                return None
        else:
            print(f"Ollama API Error: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return None


def invert_positive_negative(generated_prompts):
    """
    Invert the positive and negative labels in the generated prompts.
    This is used for Chaos Mode in video prompts.
    """
    inverted_prompts = []
    prompt_pairs = generated_prompts.split('--------------------')

    for pair in prompt_pairs:
        if "positive:" in pair and "negative:" in pair:
            positive_part = pair.split("positive:", 1)[1].split("negative:", 1)[0].strip()
            negative_part = pair.split("negative:", 1)[1].strip()
            inverted_pair = f"positive: {negative_part}\nnegative: {positive_part}"
            inverted_prompts.append(inverted_pair)

    return "--------------------\n".join(inverted_prompts)

# Function to enable button when appropriate
def enable_button(button):
    button.config(state=tk.NORMAL, bg=button['bg'])

# Function to disable button
def disable_button(button):
    button.config(state=tk.DISABLED, bg="gray")

# Function to extract number from filename for sorting
def extract_number_from_filename(filename):
    numbers = ''.join(filter(str.isdigit, filename))
    return int(numbers) if numbers else 0

# Function to copy text to clipboard
def copy_to_clipboard(text):
    pyperclip.copy(text)
    messagebox.showinfo("Clipboard", f"Copied to clipboard.")

# Function to open website
def open_website(event):
    webbrowser.open_new("https://www.TemporalLab.com")

# ===========================
# ======= OPTIONS LISTS =====
# ===========================

THEMES = ["Adventure", "Sci-Fi", "Romance", "Horror", "Comedy"]
ART_STYLES = ["Realism", "Abstract", "Surrealism", "Minimalism", "Expressionism"]
LIGHTING_OPTIONS = ["Natural Light", "Artificial Light", "Low Key", "High Key"]
FRAMING_OPTIONS = ["Wide Shot", "Medium Shot", "Close-Up", "Extreme Close-Up"]
CAMERA_MOVEMENTS = ["Pan", "Tilt", "Dolly", "Zoom", "Track"]
SHOT_COMPOSITIONS = ["Rule of Thirds", "Symmetry", "Leading Lines", "Framing", "Depth"]
TIME_OF_DAY_OPTIONS = ["Morning", "Afternoon", "Evening", "Night", "Midnight"]
DECADES = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
CAMERAS = {
    "1960s": ["ARRI Alexa Mini", "Panavision Panaflex"],
    "1970s": ["ARRI Alexa Mini", "Panavision Panaflex"],
    "1980s": ["ARRI Alexa Mini", "Panavision Panaflex"],
    "1990s": ["ARRI Alexa Mini", "Panavision Panaflex"],
    "2000s": ["ARRI Alexa Mini", "Panavision Panaflex"],
    "2010s": ["ARRI Alexa Mini", "Panavision Panaflex"],
    "2020s": ["ARRI Alexa Mini", "Panavision Panaflex"]
}
LENSES = ["Wide Angle", "Telephoto", "Prime", "Zoom", "Macro"]
RESOLUTIONS = {
    "1960s": ["Standard Definition (SD)"],
    "1970s": ["Standard Definition (SD)"],
    "1980s": ["Standard Definition (SD)"],
    "1990s": ["Standard Definition (SD)"],
    "2000s": ["Standard Definition (SD)"],
    "2010s": ["High Definition (HD)", "4K"],
    "2020s": ["High Definition (HD)", "4K"]
}
WILDLIFE_ANIMALS = ["None", "Birds", "Deer", "Wolves", "Foxes"]
DOMESTICATED_ANIMALS = ["None", "Dogs", "Cats", "Horses", "Cows"]
HOLIDAYS = ["Halloween", "Christmas", "New Year", "Easter", "Thanksgiving"]

class MultimediaSuiteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Temporal Labs LLC - Multimedia Suite")
        self.root.configure(bg='#0A2239')
        
        # Initialize Ollama
        self.ensure_ollama_installed_and_model_available()
        
        # Initialize variables
        self.video_prompts = ""
        self.audio_prompts = ""
        self.duration = 10  # Default duration for sound effects
        self.video_prompt_file_path = ""
        self.audio_prompt_file_path = ""
        self.output_folder = ""
        self.video_options_set = False
        self.audio_options_set = False
        self.last_used_directory = LAST_USED_DIRECTORY

        # Initialize video_prompt_number_var here
        self.video_prompt_number_var = tk.IntVar(value=DEFAULT_PROMPTS)
        
        self.build_gui()

        # Initialize settings.json if it doesn't exist
        self.initialize_settings()

        
    def ensure_ollama_installed_and_model_available(self, model_name="llama3.2"):
        try:
            # Check if the Ollama server is running by sending a simple request
            response = requests.get(f"{OLLAMA_API_URL}")
            if response.status_code != 200:
                raise Exception("Ollama server is not running.")
            print("Ollama server is running.")
        except requests.exceptions.ConnectionError:
            print("Ollama server is not running, trying to start it...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not self.wait_for_ollama_server():  # No port needed now
                print("Failed to start Ollama.")
                sys.exit(1)
        
        # Ensure the model is pulled
        self.ensure_model_available(model_name)

        
    def validate_prompts(self, generated_prompts):
        """
        Validates that each prompt set contains exactly one positive and one negative statement.

        Args:
            generated_prompts (str): The concatenated prompts string.

        Returns:
            bool: True if all prompt sets are valid, False otherwise.
        """
        prompt_sets = generated_prompts.strip().split("--------------------")
        for prompt_set in prompt_sets:
            prompt_set = prompt_set.strip()
            if not prompt_set:
                continue
            positive_count = prompt_set.lower().count("positive:")
            negative_count = prompt_set.lower().count("negative:")
            if positive_count != 1 or negative_count != 1:
                print(f"Invalid prompt set detected:\n{prompt_set}")
                return False
        return True
        
    def start_ollama_server(self, port):
        """
        Starts the Ollama server on the given port.
        """
        try:
            print(f"Starting Ollama server on port {port}...")
            subprocess.Popen(["ollama", "serve", "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            global OLLAMA_API_URL
            OLLAMA_API_URL = f"http://localhost:{port}"
            print(f"Ollama server started successfully on port {port}.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Ollama Server Error", f"Failed to start Ollama server: {e}")
            sys.exit(1)


    def wait_for_ollama_server(self, timeout=60, interval=5):
        """
        Waits for the Ollama server to be up and running.
        Args:
            timeout (int): Maximum time to wait for the server to start (in seconds).
            interval (int): Time between status checks (in seconds).
        Returns:
            bool: True if the server is up, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{OLLAMA_API_URL}")
                if response.status_code == 200:
                    print("Ollama server is up and running.")
                    return True
            except requests.ConnectionError:
                pass  # Server not up yet
            time.sleep(interval)
        return False

    @staticmethod
    def detect_gpu():
        """
        Detects if a GPU is available using torch and returns a boolean value.
        If a GPU is found, it returns True; otherwise, it returns False.
        """
        try:
            import torch
            if torch.cuda.is_available():
                print("GPU detected.")
                return True
            else:
                print("No GPU detected.")
                return False
        except ImportError:
            print("Torch is not installed, assuming no GPU.")
            return False

        
    def build_gui(self):
        # Configure root grid for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)  # Menu
        self.root.rowconfigure(1, weight=0)  # Instructions
        self.root.rowconfigure(2, weight=1)  # Input
        self.root.rowconfigure(3, weight=0)  # Buttons
        self.root.rowconfigure(4, weight=1)  # Output
        self.root.rowconfigure(5, weight=0)  # Sound and Combine Buttons
        self.root.rowconfigure(6, weight=0)  # Footer

        # Menu Bar
        menubar = Menu(self.root)
        settings_menu = Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Set OpenAI API Key", command=self.set_openai_api_key)
        settings_menu.add_command(label="Set Output Directory", command=set_output_directory)
        settings_menu.add_command(label="Set ComfyUI Prompts Folder", command=set_comfyui_prompts_folder)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        self.root.config(menu=menubar)

        # Instructions
        self.instructions = tk.Label(
            self.root,
            text="Enter the desired prompt concept (max 450 characters):",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 14, 'bold')
        )
        self.instructions.grid(row=1, column=0, pady=10, sticky='ew', padx=20)

        # Input Frame
        self.input_frame = tk.Frame(self.root, bg='#0A2239')
        self.input_frame.grid(row=2, column=0, pady=5, padx=20, sticky='nsew')
        self.input_frame.columnconfigure(0, weight=1)

        self.input_text = ScrolledText(
            self.input_frame,
            font=('Helvetica', 12),
            wrap=tk.WORD,
            bg='#1E1E1E',
            fg='white',
            bd=2,
            relief=tk.GROOVE
        )
        self.input_text.grid(row=0, column=0, sticky='nsew')
        self.input_frame.rowconfigure(0, weight=1)

        self.char_count_label = tk.Label(
            self.input_frame,
            text="0/450 characters",
            bg='#0A2239',
            fg='light blue',
            font=('Helvetica', 10, 'italic')
        )
        self.char_count_label.grid(row=1, column=0, sticky="e", pady=(2,0))

        self.input_text.bind("<KeyRelease>", self.check_input_text)

        # Buttons Frame
        self.buttons_frame = tk.Frame(self.root, bg='#0A2239')
        self.buttons_frame.grid(row=3, column=0, pady=20, padx=20, sticky='ew')
        self.buttons_frame.columnconfigure((0,1,2,3), weight=1)

        # Video Prompt Options Button
        self.video_prompt_options_button = tk.Button(
            self.buttons_frame,
            text="Video Prompt Options",
            command=self.show_video_prompt_options,
            state=tk.NORMAL,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 14, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        self.video_prompt_options_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Audio Prompt Options Button
        self.audio_prompt_options_button = tk.Button(
            self.buttons_frame,
            text="Audio Prompt Options",
            command=self.show_audio_prompt_options,
            state=tk.NORMAL,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 14, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        self.audio_prompt_options_button.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

        self.generate_video_prompts_button = tk.Button(
            self.buttons_frame,
            text="Generate Video Prompts",
            command=self.generate_video_prompts,
            state=tk.DISABLED,
            bg="#007ACC",
            fg='white',
            font=('Helvetica', 14, 'bold'),
            activebackground="#005A9E",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        self.generate_video_prompts_button.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        self.generate_audio_prompts_button = tk.Button(
            self.buttons_frame,
            text="Generate Audio Prompts",
            command=self.generate_audio_prompts,
            state=tk.DISABLED,
            bg="#007ACC",
            fg='white',
            font=('Helvetica', 14, 'bold'),
            activebackground="#005A9E",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        self.generate_audio_prompts_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

        # Output Text Area
        self.output_frame = tk.Frame(self.root, bg='#0A2239')
        self.output_frame.grid(row=4, column=0, pady=10, padx=20, sticky='nsew')
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)

        self.output_text = ScrolledText(
            self.output_frame,
            font=('Helvetica', 12),
            wrap=tk.WORD,
            bg='#1E1E1E',
            fg='white',
            bd=2,
            relief=tk.GROOVE
        )
        self.output_text.grid(row=0, column=0, sticky='nsew', padx=(0,10))

        # Copy to Clipboard Link
        self.copy_link = tk.Label(
            self.output_frame,
            text="Copy to Clipboard",
            fg="light blue",
            bg='#0A2239',
            cursor="hand2",
            font=('Helvetica', 12, 'underline')
        )
        self.copy_link.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.copy_link.bind("<Button-1>", lambda e: copy_to_clipboard(self.output_text.get("1.0", tk.END).strip()))

        # Generate Sound Effects Button
        self.generate_sound_button = tk.Button(
            self.root,
            text="Generate Sound Effects",
            command=self.generate_sound_effects,
            state=tk.DISABLED,
            bg="#28a745",
            fg="white",
            font=('Helvetica', 14, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=25,
            height=2
        )
        self.generate_sound_button.grid(row=5, column=0, pady=20, padx=20)

        # COMBINE button
        self.combine_button = tk.Button(
            self.root,
            text="COMBINE",
            command=self.combine_media,
            state=tk.DISABLED,
            bg="#FFC107",
            fg="white",
            font=('Helvetica', 16, 'bold'),
            activebackground="#e0a800",
            activeforeground='white',
            cursor="hand2",
            width=30,
            height=3
        )
        self.combine_button.grid(row=6, column=0, pady=10, padx=20)

        # Footer Frame containing Logo
        self.footer_frame = tk.Frame(self.root, bg='#0A2239')
        self.footer_frame.grid(row=7, column=0, sticky='ew', padx=20, pady=10)
        self.footer_frame.columnconfigure(0, weight=1)

        # Logo with Link
        logo_url = "https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=450,fit=crop,q=95/A1aoblXx2KSKGq4r/tlclogorawnameonly-Awvk565gvMfLKVr4.png"
        try:
            response = requests.get(logo_url)
            response.raise_for_status()
            logo_image = Image.open(BytesIO(response.content)).resize((120, 120), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(
                self.footer_frame,
                image=self.logo_photo,
                bg='#0A2239',
                cursor="hand2"
            )
            logo_label.image = self.logo_photo
            logo_label.grid(row=0, column=0, sticky='e', padx=20, pady=5)
            logo_label.bind("<Button-1>", open_website)
        except Exception as e:
            print(f"Failed to download or display logo: {e}")
            messagebox.showerror("Logo Error", f"Failed to load logo: {e}")

        # Contact Information Label
        contact_info = "Sol@TemporalLab.com - Text 385-222-9920"
        contact_label = tk.Label(
            self.root,
            text=contact_info,
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 10)
        )
        contact_label.grid(row=8, column=0, sticky='s', pady=5)

        # Apply styles
        self.apply_styles()

    def apply_styles(self):
        style = ttk.Style()
        style.configure('TCheckbutton', background='#0A2239', foreground='white', font=('Helvetica', 12))
        style.configure('TButton', font=('Helvetica', 12), relief='flat', background="#007ACC")

    def check_input_text(self, event=None):
        text = self.input_text.get("1.0", tk.END).strip()
        char_count = len(text)
        self.char_count_label.config(text=f"{char_count}/450 characters")

        # Enable or disable buttons based on text length
        if 1 <= char_count <= 450:
            enable_button(self.generate_video_prompts_button)
        else:
            disable_button(self.generate_video_prompts_button)

    def set_openai_api_key(self):
        global OPENAI_API_KEY
        OPENAI_API_KEY = simpledialog.askstring("Set OpenAI API Key", "Enter your OpenAI API Key:", show="*")
        if OPENAI_API_KEY:
            save_settings()

    def create_smart_directory_and_filenames(self, input_concept):
        sanitized_concept = re.sub(r'[^\w\s-]', '', input_concept).strip().replace(' ', '_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"{sanitized_concept}_{timestamp}"
        directory_path = os.path.join(self.output_folder, directory_name)

        video_directory = os.path.join(directory_path, "Video")
        audio_directory = os.path.join(directory_path, "Audio")

        os.makedirs(video_directory, exist_ok=True)
        os.makedirs(audio_directory, exist_ok=True)

        video_filename = f"{sanitized_concept}_video_prompts.txt"
        audio_filename = f"{sanitized_concept}_audio_prompts.txt"

        return directory_path, video_directory, audio_directory, video_filename, audio_filename


    # Function to open video prompt options
    def show_video_prompt_options(self):
        try:
            print("Opening video prompt options window.")
            self.video_options_window = tk.Toplevel(self.root)
            self.video_options_window.title("Video Prompt Options")
            self.video_options_window.configure(bg='#0A2239')

            print("Building video options...")
            self.build_video_options(self.video_options_window)
            print("Video options window built successfully.")

        except Exception as e:
            print(f"Error occurred in show_video_prompt_options: {e}")
            traceback.print_exc()

        
    def ensure_single_negative(self, formatted_prompts):
        """
        Ensures that each prompt set contains only one negative statement.
        Removes any additional negative statements.

        Args:
            formatted_prompts (str): The concatenated prompts string.

        Returns:
            str: The cleaned prompts with only one negative per set.
        """
        prompt_sets = formatted_prompts.split('--------------------')
        cleaned_prompts = ""

        for prompt_set in prompt_sets:
            prompt_set = prompt_set.strip()
            if not prompt_set:
                continue

            # Extract one positive and one negative using regex
            positive_match = re.search(r"positive:\s*(.+)", prompt_set, re.IGNORECASE)
            negative_matches = re.findall(r"negative:\s*(.+)", prompt_set, re.IGNORECASE)

            if positive_match and negative_matches:
                positive_prompt = positive_match.group(1).strip()
                negative_prompt = negative_matches[0].strip()  # Take only the first negative
                cleaned_prompts += f"positive: {positive_prompt}\nnegative: {negative_prompt}\n--------------------\n"
            else:
                # Handle incomplete prompt sets
                messagebox.showwarning("Formatting Warning", f"Incomplete or malformed prompt detected:\n{prompt_set}")
                continue

        return cleaned_prompts.strip()


    def validate_prompts(self, generated_prompts, expected_count):
        """
        Validates that the number of prompt sets matches the expected count and 
        that each set contains exactly one positive and one negative statement.

        Args:
            generated_prompts (str): The concatenated prompts string.
            expected_count (int): The expected number of prompt sets.

        Returns:
            bool: True if all prompt sets are valid, False otherwise.
        """
        prompt_sets = generated_prompts.strip().split("--------------------")
        
        # Validate that the number of prompt sets matches the expected count
        if len(prompt_sets) != expected_count:
            print(f"Expected {expected_count} prompt sets, but got {len(prompt_sets)}.")
            return False

        for prompt_set in prompt_sets:
            prompt_set = prompt_set.strip()
            if not prompt_set:
                continue
            positive_count = prompt_set.lower().count("positive:")
            negative_count = prompt_set.lower().count("negative:")
            if positive_count != 1 or negative_count != 1:
                print(f"Invalid prompt set detected:\n{prompt_set}")
                return False
        return True


    # Function to open audio prompt options
    def show_audio_prompt_options(self):
        self.audio_options_window = tk.Toplevel(self.root)
        self.audio_options_window.title("Audio Prompt Options")
        self.audio_options_window.configure(bg='#0A2239')

        self.build_audio_options(self.audio_options_window)

    def generate_video_prompts(self):
        """
        Generate video prompts that take all options into account.
        Ensure that every setting dynamically influences the final prompts, including visual and cinematic elements.
        """
        input_concept = self.input_text.get("1.0", tk.END).strip()

        if len(input_concept) == 0 or len(input_concept) > MAX_CHAR_LIMIT:
            messagebox.showerror("Input Error", f"The prompt must be between 1 and {MAX_CHAR_LIMIT} characters.")
            return

        # Ask the user to select an output directory for saving the prompts
        output_directory = filedialog.askdirectory(title="Select Output Directory")
        
        if not output_directory:
            messagebox.showwarning("Directory Selection", "No directory selected. Please select a valid directory.")
            return
        
        self.output_folder = output_directory  # Save the directory for future reference

        try:
            # Gather all options set by the user
            video_options = {
                "theme": self.video_theme_var.get(),
                "art_style": self.video_art_style_var.get(),
                "lighting": self.video_lighting_var.get(),
                "framing": self.video_framing_var.get(),
                "camera_movement": self.video_camera_movement_var.get(),
                "shot_composition": self.video_shot_composition_var.get(),
                "time_of_day": self.video_time_of_day_var.get(),
                "decade": self.video_decade_var.get(),  # Keep this as a string, no conversion to int
                "camera": self.video_camera_var.get(),
                "lens": self.video_lens_var.get(),
                "resolution": self.video_resolution_var.get(),
                "wildlife_animal": self.wildlife_animal_var.get(),
                "domesticated_animal": self.domesticated_animal_var.get(),
                "soundscape_mode": self.video_soundscape_mode_var.get(),
                "holiday_mode": self.video_holiday_mode_var.get(),
                "selected_holidays": self.video_holidays_var.get(),
                "specific_modes": [mode for mode, var in self.video_specific_modes_vars.items() if var.get()],
                "no_people_mode": self.video_no_people_mode_var.get(),
                "chaos_mode": self.video_chaos_mode_var.get(),
                "story_mode": self.video_story_mode_var.get(),
                "remix_mode": self.video_remix_mode_var.get(),
            }

            # Build the context for options
            options_context = (
                f"Generate a detailed video prompt for each year in the {video_options['decade']}s. "
                f"Each prompt should reflect:\n"
                f"- Theme: {video_options['theme']}\n"
                f"- Art Style: {video_options['art_style']}\n"
                f"- Lighting: {video_options['lighting']}\n"
                f"- Framing: {video_options['framing']}\n"
                f"- Camera Movement: {video_options['camera_movement']}\n"
                f"- Shot Composition: {video_options['shot_composition']}\n"
                f"- Time of Day: {video_options['time_of_day']}\n"
                f"- Camera: {video_options['camera']}, Lens: {video_options['lens']}\n"
                f"- Resolution: {video_options['resolution']}\n"
            )

            # Add optional elements dynamically
            if video_options["wildlife_animal"]:
                options_context += f"- Feature a {video_options['wildlife_animal']}.\n"

            if video_options["domesticated_animal"]:
                options_context += f"- Include a {video_options['domesticated_animal']}.\n"

            if video_options["soundscape_mode"]:
                options_context += "- Incorporate soundscapes relevant to the scene.\n"

            if video_options["holiday_mode"]:
                options_context += f"- Apply holiday themes: {video_options['selected_holidays']}.\n"

            if video_options["no_people_mode"]:
                options_context += "- Focus on the environment or animals, without human figures.\n"

            if video_options["chaos_mode"]:
                options_context += "- Introduce chaotic elements that create tension or contrast in the visuals.\n"

            if video_options["story_mode"]:
                options_context += "- Ensure prompts flow together as a cohesive narrative.\n"

            if video_options["remix_mode"]:
                options_context += "- Add creative variations in visual styles or thematic choices.\n"

            # Build the final prompt by combining the options context
            prompt = (
                f"Generate {self.video_prompt_number_var.get()} detailed video prompts for the concept '{input_concept}' "
                f"based on the {video_options['decade']}s. Ensure that each prompt includes:\n"
                f"{options_context}"
            )

            # Call Ollama API or model to generate raw video prompts
            raw_video_prompts = self.generate_prompts_via_ollama(prompt, 'video', self.video_prompt_number_var.get())

            if not raw_video_prompts:
                raise Exception("No video prompts generated. Please try again.")

            # Clean and format the prompts
            cleaned_prompts = clean_prompt_text(raw_video_prompts)
            formatted_prompts = remove_unwanted_headers(cleaned_prompts)

            # Validate the generated prompts
            if not self.validate_prompts(formatted_prompts, self.video_prompt_number_var.get()):
                messagebox.showerror("Temporal Notification", "Please GENERATE VIDEO PROMPTS again. Occasionally it takes a couple of tries.")
                return

            # Save the prompts to the selected output folder
            directory, video_folder, audio_folder, video_filename, _ = self.create_smart_directory_and_filenames(input_concept, self.output_folder)
            video_save_path = os.path.join(video_folder, video_filename)
            save_to_file(formatted_prompts, video_save_path)

            # Initialize audio_save_folder if not done already
            self.audio_save_folder = audio_folder  # This fixes the error

            # Store the prompts in the class-level attribute `self.video_prompts`
            self.video_prompts = formatted_prompts  # Store for later use

            # Enable the button to generate audio prompts
            enable_button(self.generate_audio_prompts_button)

            # Display formatted prompts in the output text box
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Generated Video Prompts:\n\n" + formatted_prompts)

            # Optionally, log the save paths for verification
            print(f"Video prompts saved to: {video_save_path}")
            print(f"Audio prompts will be saved to: {self.audio_save_folder}")

        except Exception as e:
            messagebox.showerror("Prompt Generation Error", f"Failed to generate video prompts: {e}")
            print(f"Error generating video prompts: {e}")

    def set_output_directory(self):
        self.output_folder = filedialog.askdirectory(title="Select Output Directory")
        if not self.output_folder:
            messagebox.showwarning("Directory Selection", "No directory selected. Please select a valid directory.")
            return
        print(f"Output directory set to: {self.output_folder}")

    def ensure_prompt_count_update(self):
        """
        Forces the prompt count to update at least once to avoid blank generations.
        """
        self.video_prompt_number_var.set(DEFAULT_PROMPTS + 1)  # Increment once
        self.video_prompt_number_var.set(DEFAULT_PROMPTS)  # Reset to the original value

    def generate_audio_prompts(self):
        """
        Generate detailed audio prompts based on the input concept entered by the user.
        This function validates the input, sends the request to the API, and displays the generated prompts.
        It now integrates with the multi-layered audio system for more dynamic soundscape generation.
        """
        if not self.video_prompts:
            messagebox.showerror("Video Prompts Missing", "Please generate video prompts before generating audio prompts.")
            return

        input_concept = self.input_text.get("1.0", tk.END).strip()

        if len(input_concept) == 0 or len(input_concept) > MAX_CHAR_LIMIT:
            messagebox.showerror("Input Error", f"The prompt must be between 1 and {MAX_CHAR_LIMIT} characters.")
            return

        # Calculate the number of video prompts
        video_prompt_count = self.video_prompts.count('positive:')  # Count how many 'positive:' labels exist in video prompts
        print(f"Video Prompt Count: {video_prompt_count}")

        try:
            # Generate sonic descriptions for each video prompt
            sonic_descriptions = []
            for video_prompt in self.video_prompts.split("--------------------"):
                video_prompt = video_prompt.strip()
                if not video_prompt:
                    continue

                # Create a sound-specific prompt from the video prompt
                sound_prompt = (
                    f"Generate a sonic description based on the following visual scene: '{video_prompt}'. "
                    "Focus only on the sounds, atmosphere, background noises, ambient audio, and specific sonic elements. "
                    "Do not describe visual elements. Describe what the listener would hear in this scene, including subtle "
                    "details like environmental sounds, echoes, footsteps, machinery, voices, or any relevant sonic cues."
                )

                # Send the sound prompt to Ollama to generate the sonic description
                translated_sonic_prompt = self.generate_prompts_via_ollama(sound_prompt, 'audio', 1)  # Process one audio prompt at a time
                
                if not translated_sonic_prompt:
                    raise Exception("No sonic description generated. Please try again.")
                
                # Clean and store the sonic description
                cleaned_sonic_prompt = clean_prompt_text(translated_sonic_prompt)
                sonic_descriptions.append(cleaned_sonic_prompt)

            # Join the sonic descriptions together with the same format as video prompts
            formatted_sonic_prompts = "\n--------------------\n".join(sonic_descriptions)

            # Save and display the formatted audio prompts
            directory, _, audio_folder, _, audio_filename = self.create_smart_directory_and_filenames(input_concept, OUTPUT_DIRECTORY)
            self.audio_save_folder = audio_folder
            audio_save_path = os.path.join(self.audio_save_folder, audio_filename)
            save_to_file(formatted_sonic_prompts, audio_save_path)

            # Store the audio prompts for later use
            self.audio_prompts = formatted_sonic_prompts
            enable_button(self.generate_sound_button)

            # Display the formatted audio prompts
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Generated Audio Prompts:\n\n" + formatted_sonic_prompts)

            # Now generate the sound effects based on the multi-layer system
            self.generate_audio_menu()  # Opens the new menu for selecting sound layers

        except Exception as e:
            messagebox.showerror("Prompt Generation Error", f"Failed to generate audio prompts: {e}")
            print(f"Error generating audio prompts: {e}")
            
    def initialize_ollama(self):
        """
        Initializes the Ollama API URL assuming the server is already running.
        """
        ollama_port = 11434  # Set to your known port
        
        # Define OLLAMA_API_URL globally so it can be used elsewhere
        global OLLAMA_API_URL
        OLLAMA_API_URL = f"http://localhost:{ollama_port}"  # Adjust if necessary
        
        print(f"Ollama server assumed to be running on port {ollama_port}")
            
    def ensure_single_negative(self, formatted_prompts):
        """
        Ensures that each prompt set contains only one negative statement.
        Removes any additional negative statements.
        """
        prompt_sets = formatted_prompts.split('--------------------')
        cleaned_prompts = ""

        for prompt_set in prompt_sets:
            prompt_set = prompt_set.strip()
            if not prompt_set:
                continue

            positive_match = re.search(r"positive:\s*(.+)", prompt_set, re.IGNORECASE)
            negative_matches = re.findall(r"negative:\s*(.+)", prompt_set, re.IGNORECASE)

            if positive_match and negative_matches:
                positive_prompt = positive_match.group(1).strip()
                negative_prompt = negative_matches[0].strip()  # Take only the first negative
                cleaned_prompts += f"positive: {positive_prompt}\nnegative: {negative_prompt}\n--------------------\n"
            else:
                messagebox.showwarning("Formatting Warning", f"Incomplete or malformed prompt detected:\n{prompt_set}")
                continue

        return cleaned_prompts.strip()
        
    def initialize_settings(self):
        """
        Initializes the settings.json file if it doesn't exist.
        """
        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(DEFAULT_SETTINGS, f, indent=4)
            print("Initialized settings.json with default settings.")
        else:
            print("settings.json already exists.")

        

    def check_coherency_with_ollama(video_prompts, audio_prompts):
        """
        Checks the coherency between video and audio prompts using Ollama and returns a score between 0 and 1.

        Args:
            video_prompts (list of str): List of positive video prompts.
            audio_prompts (list of str): List of generated audio prompts.

        Returns:
            float: Average coherency score between 0 and 1.
        """
        score = 0
        count = min(len(video_prompts), len(audio_prompts))

        for i in range(count):
            try:
                system_prompt = "You are an AI assistant that evaluates the coherency between a video prompt and an audio prompt. Rate the coherency on a scale from 0 to 1."
                user_prompt = f"Video Prompt: {video_prompts[i]}\nAudio Prompt: {audio_prompts[i]}\n\nProvide a coherency score between 0 and 1."

                full_prompt = f"{system_prompt}\n\n{user_prompt}"

                # Execute Ollama command to get the coherency score using llama3.1
                process = subprocess.Popen(
                    ["ollama", "run", "llama3.2", "--prompt"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=full_prompt)

                if process.returncode != 0:
                    raise Exception(f"Ollama Error: {stderr.strip()}")

                response = stdout.strip()

                # Extract the float score from the response
                coherency_score_match = re.search(r"(\d\.\d+)", response)
                if coherency_score_match:
                    coherency_score = float(coherency_score_match.group(1))
                    score += coherency_score
                else:
                    raise ValueError("No valid coherency score found in the response.")

            except Exception as e:
                print(f"Error in coherency check with Ollama: {e}")
                continue

        average_score = score / count if count > 0 else 0
        return average_score
        
    def invert_positive_negative(self, formatted_prompts):
        """
        Inverts the positive and negative prompts for Chaos Mode.

        Args:
            formatted_prompts (str): The concatenated prompts string.

        Returns:
            str: The inverted prompts string.
        """
        prompt_sets = formatted_prompts.split('--------------------')
        inverted_prompts = ""

        for prompt_set in prompt_sets:
            prompt_set = prompt_set.strip()
            if not prompt_set:
                continue

            positive_match = re.search(r"positive:\s*(.+)", prompt_set, re.IGNORECASE)
            negative_match = re.search(r"negative:\s*(.+)", prompt_set, re.IGNORECASE)

            if positive_match and negative_match:
                positive_prompt = positive_match.group(1).strip()
                negative_prompt = negative_match.group(1).strip()
                # Swap positive and negative
                inverted_prompts += f"positive: {negative_prompt}\nnegative: {positive_prompt}\n--------------------\n"
            else:
                continue

        return inverted_prompts.strip()

        
    def generate_prompts_via_ollama(self, input_concept, prompt_type, number_of_prompts, options=None):
        system_prompt = f"""
        You are an AI assistant tasked with generating prompts for video generation models. Each prompt must strictly follow the format below, with no additional information or explanation:

        Example format:

        positive: Describe the positive aspects of the scene or shot in detail.
        negative: Describe what to avoid in the scene or shot in detail.
        --------------------

        Generate {number_of_prompts} sets of {prompt_type} prompts based on the following concept: '{input_concept}'. Ensure that each prompt set strictly follows the Correct Examples with both positive and negative format.
        """
        try:
            self.ensure_ollama_installed_and_model_available()
            api_url = f"{OLLAMA_API_URL}/api/generate"
            payload = {
                "model": REQUIRED_MODEL,
                "prompt": system_prompt,
                "max_tokens": 1500,
                "stream": False
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                raw_response = response.text.strip()
                raw_prompts = self.parse_raw_response(raw_response)
                return raw_prompts
            else:
                raise Exception(f"Ollama API returned an error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error generating prompts via Ollama: {e}")
            return None
                
    def ensure_model_available(self, model_name):
        try:
            model_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model_name not in model_list.stdout:
                print(f"Model '{model_name}' is not available locally. Pulling model...")
                subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"Model '{model_name}' is available.")
        except Exception as e:
            print(f"Error ensuring model availability: {e}")
            
    def parse_raw_response(self, raw_data):
        """
        Parses the raw JSON response from the API and extracts the 'response' field which contains the raw prompts.
        """
        try:
            # Add a debug statement for the raw_data
            print(f"Parsing raw data: {raw_data}")
            data = json.loads(raw_data)
            raw_prompts = data.get("response", "")
            if not raw_prompts:
                print("No 'response' field found in the API response.")
            return raw_prompts
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return ""



    def ensure_audioldm2_installed(self):
        """
        Ensures that AudioLDM2 and torchaudio are installed as Python modules.
        If not, attempts to install them. Also checks for system dependencies.
        """
        try:
            import audioldm2
            import torchaudio
            print("AudioLDM2 and torchaudio are installed.")
        except ImportError as e:
            missing_package = str(e).split("'")[1]
            response = messagebox.askyesno(
                "Missing Dependency",
                f"The package '{missing_package}' is not installed. Would you like to install it now?"
            )
            if response:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', missing_package])
                    messagebox.showinfo("Installation Complete", f"'{missing_package}' has been installed successfully.")
                except subprocess.CalledProcessError as install_error:
                    messagebox.showerror("Installation Error", f"Failed to install '{missing_package}':\n{install_error}")
                    sys.exit(1)
            else:
                messagebox.showwarning("Dependency Required", f"'{missing_package}' is required to generate sound effects. Exiting...")
                sys.exit(1)

        # Ensure torchaudio is installed and matches torch version
        try:
            import torch
            import torchaudio

            # Check if torch and torchaudio versions are compatible
            torch_version = torch.__version__.split('+')[0]
            torchaudio_version = torchaudio.__version__.split('+')[0]

            if not torchaudio_version.startswith(torch_version[:5]):
                raise ImportError("torchaudio version does not match torch version.")

            print(f"torch version: {torch_version}, torchaudio version: {torchaudio_version}")

        except ImportError as e:
            response = messagebox.askyesno(
                "torchaudio Issue",
                f"torchaudio is missing or incompatible: {e}\nWould you like to reinstall torchaudio now?"
            )
            if response:
                try:
                    # Determine if the system has CUDA support
                    try:
                        import torch
                        cuda_available = torch.cuda.is_available()
                    except:
                        cuda_available = False

                    if cuda_available:
                        # Example for CUDA 11.8; adjust based on your system
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchaudio', '--extra-index-url', 'https://download.pytorch.org/whl/cu118'])
                    else:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchaudio', '--extra-index-url', 'https://download.pytorch.org/whl/cpu'])

                    messagebox.showinfo("Reinstallation Complete", "torchaudio has been reinstalled successfully.")

                    # Re-import to verify
                    import torchaudio
                    print(f"Reinstalled torchaudio version: {torchaudio.__version__}")

                except subprocess.CalledProcessError as e:
                    messagebox.showerror("Reinstallation Error", f"Failed to reinstall torchaudio:\n{e}")
                    sys.exit(1)
            else:
                messagebox.showwarning("torchaudio Required", "torchaudio is required for AudioLDM2. Exiting...")
                sys.exit(1)

        # Check for Microsoft Visual C++ Redistributable
        if not self.check_visual_cplusplus_installed():
            response = messagebox.askyesno(
                "Missing Dependency",
                "Microsoft Visual C++ Redistributable is not installed. It's required for torchaudio.\nWould you like to download it now?"
            )
            if response:
                try:
                    webbrowser.open_new("https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170")
                    messagebox.showinfo("Install Visual C++", "Please install the Microsoft Visual C++ Redistributable from your browser and restart the application.")
                    sys.exit(1)
                except Exception as e:
                    messagebox.showerror("Download Error", f"Failed to open the browser for downloading Visual C++ Redistributable:\n{e}")
                    sys.exit(1)
            else:
                messagebox.showwarning("Dependency Required", "Microsoft Visual C++ Redistributable is required for torchaudio. Exiting...")
                sys.exit(1)

    def check_visual_cplusplus_installed(self):
        """
        Checks if the Microsoft Visual C++ Redistributable is installed.
        This is a basic check by looking for common registry entries.
        """
        import winreg

        try:
            registry = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
            key = winreg.OpenKey(registry, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64")
            value, regtype = winreg.QueryValueEx(key, "Installed")
            winreg.CloseKey(key)
            return value == 1
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error checking Visual C++ Redistributable: {e}")
            return False

            
    def run_audioldm2(self, prompt_text, output_filename, index):
        try:
            # Determine device
            device = "cuda" if self.detect_gpu() else "cpu"
            
            # Construct the AudioLDM2 command
            command = [
                sys.executable, '-m', 'audioldm2',
                '-t', prompt_text,
                '-s', output_filename,
                '--model_name', self.audio_model_name_var.get(),
                '--device', device,  # Correctly set device
                '--ddim_steps', str(self.audio_ddim_steps_var.get()),
                '--guidance_scale', str(self.audio_guidance_scale_var.get()),
                '--duration', str(self.duration),
                '--n_candidate_gen_per_text', str(self.audio_n_candidate_var.get()),
                '--seed', str(self.audio_seed_var.get()),
                '--mode', 'generation'
            ]

            print(f"[AudioLDM2] Executing command: {' '.join(command)}")
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy()  # Inherit environment variables
            )
            print(f"[AudioLDM2] Generated sound effect for prompt {index}: {prompt_text}")
        except subprocess.CalledProcessError as e:
            print(f"[AudioLDM2 Error] Failed to generate sound effect for prompt {index}: {e.stderr}")
            messagebox.showerror("AudioLDM2 Error", f"Failed to generate sound effect for prompt {index}.\nError: {e.stderr}")
        except Exception as e:
            print(f"[AudioLDM2 Error] Unexpected error for prompt {index}: {e}")
            messagebox.showerror("AudioLDM2 Error", f"Failed to generate sound effect for prompt {index}.\nError: {e}")
        finally:
            # Update progress bar
            self.progress_bar['value'] = index
            self.progress_window.update_idletasks()



    def generate_sound_effects(self):
        print("generate_sound_effects called")
        if not self.audio_prompts:
            messagebox.showwarning("Prompt Error", "No audio prompts found. Please generate audio prompts first.")
            return

        # Ask for the duration
        duration = self.audio_length_var.get()
        if not duration:
            messagebox.showwarning("Duration Error", "Please provide a valid duration.")
            return

        self.duration = duration  # Store duration in the class instance
        print(f"Audio Duration: {self.duration} seconds")

        # Use the base directory and save the sound effects in the "Audio" folder
        audio_save_folder = self.audio_save_folder

        if not audio_save_folder or not os.path.exists(audio_save_folder):
            messagebox.showerror("Invalid Audio Save Folder", "The audio save folder is invalid or does not exist.")
            print("Invalid audio save folder path.")
            return

        # Create the output directory if it doesn't exist
        os.makedirs(audio_save_folder, exist_ok=True)
        print(f"Audio save folder: {audio_save_folder}")

        # Split the audio prompts into individual prompts and remove any unwanted prefixes
        prompts = [re.sub(r'negative:\s*', '', p.strip()) for p in self.audio_prompts.strip().split('--------------------') if p.strip()]
        print(f"Number of audio prompts to process: {len(prompts)}")

        # Retrieve the number of prompts to generate
        number_of_prompts = self.video_prompt_number_var.get()
        waveforms_per_prompt = self.audio_waveforms_var.get()
        inference_steps = self.audio_inference_steps_var.get()
        seed = self.audio_seed_var.get()
        model_name = self.audio_model_name_var.get()

        print(f"Generating {number_of_prompts} prompts with {waveforms_per_prompt} waveforms each.")

        # Ensure AudioLDM2 and torchaudio are installed
        self.ensure_audioldm2_installed()

        # Initialize the AudioLDM2 pipeline
        try:
            repo_id = "cvssp/audioldm2-large"  # Adjust based on your needs
            pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
            device = "cuda" if self.detect_gpu() else "cpu"
            pipe = pipe.to(device)
            print(f"AudioLDM2 pipeline loaded on {device}.")
        except Exception as e:
            messagebox.showerror("Pipeline Error", f"Failed to load AudioLDM2 pipeline:\n{e}")
            print(f"Failed to load AudioLDM2 pipeline: {e}")
            return

        # Create a progress window and store references in the class instance
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Generating Sound Effects")
        self.progress_window.geometry("400x100")
        self.progress_window.grab_set()  # Make the progress window modal

        progress_label = tk.Label(
            self.progress_window,
            text="Generating sound effects...",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_window, orient='horizontal', length=300, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.progress_bar['maximum'] = number_of_prompts * waveforms_per_prompt

        def run_all_generations():
            print("run_all_generations started")
            for i, prompt_text in enumerate(prompts[:number_of_prompts], start=1):
                if prompt_text:
                    print(f"Generating audio for prompt {i}: {prompt_text}")
                    try:
                        generator = torch.Generator(device=device).manual_seed(seed)
                        audio_samples = pipe(
                            prompt_text,
                            negative_prompt="Low quality.",  # You can customize this
                            num_inference_steps=inference_steps,
                            audio_length_in_s=duration,
                            num_waveforms_per_prompt=waveforms_per_prompt,
                            generator=generator
                        ).audios

                        for j, audio in enumerate(audio_samples, start=1):
                            # Sanitize the prompt for filename
                            sanitized_prompt = re.sub(r'[^\w\s-]', '', prompt_text).strip().replace(' ', '_')[:50]
                            if not sanitized_prompt:
                                sanitized_prompt = f"prompt_{i}"
                            output_filename = os.path.join(audio_save_folder, f"sound_effect_{i}_{j}_{sanitized_prompt}.wav")

                            # Save the audio using scipy
                            scipy.io.wavfile.write(output_filename, rate=16000, data=audio)  # Removed .numpy()
                            print(f"Saved sound effect to {output_filename}")

                            # Update progress bar
                            self.progress_bar['value'] += 1
                            self.progress_window.update_idletasks()

                    except Exception as e:
                        print(f"[AudioLDM2 Error] Failed to generate sound effect for prompt {i}: {e}")
                        messagebox.showerror("AudioLDM2 Error", f"Failed to generate sound effect for prompt {i}.\nError: {e}")

            self.progress_window.destroy()
            messagebox.showinfo("Sound Effects Generated", f"Sound effects have been generated in the '{audio_save_folder}' folder.")
            enable_button(self.combine_button)
            print("Audio generation process completed.")

        # Start the thread for generation
        threading.Thread(target=run_all_generations, daemon=True).start()
        print("Audio generation thread started.")


    def load_settings(self):
        """
        Loads settings from the SETTINGS_FILE and assigns them to the respective variables.
        If SETTINGS_FILE does not exist, it initializes it with DEFAULT_SETTINGS.
        """
        if not os.path.exists(SETTINGS_FILE):
            initialize_settings()
        
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        
        # Load Video Options
        if 'video_options' in settings:
            video_options = settings['video_options']
            self.video_theme_var.set(video_options.get("theme", THEMES[0]))
            self.video_art_style_var.set(video_options.get("art_style", ART_STYLES[0]))
            self.video_lighting_var.set(video_options.get("lighting", LIGHTING_OPTIONS[0]))
            self.video_framing_var.set(video_options.get("framing", FRAMING_OPTIONS[0]))
            self.video_camera_movement_var.set(video_options.get("camera_movement", CAMERA_MOVEMENTS[0]))
            self.video_shot_composition_var.set(video_options.get("shot_composition", SHOT_COMPOSITIONS[0]))
            self.video_time_of_day_var.set(video_options.get("time_of_day", TIME_OF_DAY_OPTIONS[0]))
            self.video_decade_var.set(video_options.get("decade", DECADES[0]))
            self.update_video_camera_options()
            self.video_camera_var.set(video_options.get("camera", CAMERAS[self.video_decade_var.get()][0]))
            self.video_lens_var.set(video_options.get("lens", LENSES[0]))
            self.video_resolution_var.set(video_options.get("resolution", list(RESOLUTIONS.values())[0][0]))
            self.wildlife_animal_var.set(video_options.get("wildlife_animal", WILDLIFE_ANIMALS[0]))
            self.domesticated_animal_var.set(video_options.get("domesticated_animal", DOMESTICATED_ANIMALS[0]))
            self.video_soundscape_mode_var.set(video_options.get("soundscape_mode", False))
            self.video_holiday_mode_var.set(video_options.get("holiday_mode", False))
            if self.video_holiday_mode_var.get() and video_options.get("selected_holidays"):
                self.video_holidays_var.set(video_options["selected_holidays"][0])
                self.video_holidays_combobox.config(state="readonly")
            else:
                self.video_holidays_combobox.config(state="disabled")
            for mode in video_options.get("specific_modes", []):
                if mode in self.video_specific_modes_vars:
                    self.video_specific_modes_vars[mode].set(True)
            self.video_no_people_mode_var.set(video_options.get("no_people_mode", False))
            self.video_test_mode_var.set(video_options.get("test_mode", False))
            self.video_remix_mode_var.set(video_options.get("remix_mode", False))
            self.video_story_mode_var.set(video_options.get("story_mode", False))
            self.video_chaos_mode_var.set(video_options.get("chaos_mode", False))

        # Load Audio Options
        if 'audio_options' in settings:
            audio_options = settings['audio_options']
            self.audio_exclude_music_var.set(audio_options.get("exclude_music", False))
            self.audio_holiday_mode_var.set(audio_options.get("holiday_mode", False))
            if self.audio_holiday_mode_var.get() and audio_options.get("selected_holidays"):
                self.audio_holidays_var.set(audio_options["selected_holidays"][0])
                self.audio_holidays_combobox.config(state="readonly")
            else:
                self.audio_holidays_combobox.config(state="disabled")
            for mode in audio_options.get("specific_modes", []):
                if mode in self.audio_specific_modes_vars:
                    self.audio_specific_modes_vars[mode].set(True)
            self.audio_open_source_mode_var.set(audio_options.get("open_source_mode", True))
            self.audio_model_name_var.set(audio_options.get("model_name", "audioldm2-full-large-1150k"))
            self.audio_guidance_scale_var.set(audio_options.get("guidance_scale", 10.0))
            self.audio_ddim_steps_var.set(audio_options.get("ddim_steps", 100))
            self.audio_n_candidate_var.set(audio_options.get("n_candidate_gen_per_text", 5))
            self.audio_seed_var.set(audio_options.get("seed", 12345))
            
    def ensure_ollama_installed_and_model_available(self, model_name="llama3.2"):
        try:
            # Check if the Ollama server is running by sending a simple request
            response = requests.get(f"{OLLAMA_API_URL}")
            if response.status_code != 200:
                raise Exception("Ollama server is not running.")
            print("Ollama server is running.")
        except requests.exceptions.ConnectionError:
            print("Ollama server is not running, trying to start it...")
            subprocess.run(["ollama", "serve"], check=True)
            if not self.wait_for_ollama_server():
                print("Failed to start Ollama.")
                sys.exit(1)
        
        # Ensure the model is pulled
        self.ensure_model_available(model_name)


    def combine_video(self):
        """
        Combines each video file with the corresponding audio sound effect.
        Saves the combined media in the specified directories.

        This function performs the following steps:
        - Reads all video files from the Video folder.
        - Reads all audio files from the Audio folder.
        - Combines each video with its corresponding audio.
        - Saves the combined videos in a new folder named FINAL_VIDEOS_{number_of_prompts}_prompts_expected.
        - Copies the audio files to a new folder named FINAL_AUDIOS_{number_of_prompts}_prompts_expected.
        """

        import shutil
        from moviepy.video import fx as vfx

        # Collect video files
        video_files = sorted(
            [os.path.join(self.video_save_folder, f) for f in os.listdir(self.video_save_folder)
             if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))],
            key=extract_number_from_filename
        )

        if not video_files:
            messagebox.showwarning("No Video Files", "No video files found in the Video folder.")
            return

        # Collect audio files
        audio_files = sorted(
            [os.path.join(self.audio_save_folder, f) for f in os.listdir(self.audio_save_folder)
             if f.endswith(('.mp3', '.wav', '.aac'))],
            key=extract_number_from_filename
        )

        if not audio_files:
            messagebox.showwarning("No Audio Files", "No audio files found in the Audio folder.")
            return

        # Ensure the number of video files matches the number of audio files
        if len(video_files) != len(audio_files):
            messagebox.showwarning("File Mismatch", "The number of video files does not match the number of audio files.")
            return

        # Get the number of prompts
        number_of_prompts = self.video_prompt_number_var.get()

        # Ask the user to select the output directory
        chosen_directory = filedialog.askdirectory(title="Select Output Directory")
        if not chosen_directory:
            messagebox.showwarning("Output Directory", "No directory selected.")
            return

        # Create the output directories
        final_videos_folder = os.path.join(
            chosen_directory, f"FINAL_VIDEOS_{number_of_prompts}_prompts_expected")
        final_audios_folder = os.path.join(
            chosen_directory, f"FINAL_AUDIOS_{number_of_prompts}_prompts_expected")

        os.makedirs(final_videos_folder, exist_ok=True)
        os.makedirs(final_audios_folder, exist_ok=True)

        try:
            for i, (video_file, audio_file) in enumerate(zip(video_files, audio_files), start=1):
                video_clip = VideoFileClip(video_file)
                audio_clip = AudioFileClip(audio_file)

                # Adjust audio duration to match video duration
                if audio_clip.duration < video_clip.duration:
                    # Loop the audio to match video duration
                    audio_clip = audio_clip.fx(vfx.loop, duration=video_clip.duration)
                elif audio_clip.duration > video_clip.duration:
                    # Trim the audio to match video duration
                    audio_clip = audio_clip.subclip(0, video_clip.duration)

                # Set audio to video
                final_clip = video_clip.set_audio(audio_clip)

                # Save the combined video
                output_video_filename = os.path.join(final_videos_folder, f"combined_video_{i}.mp4")
                final_clip.write_videofile(output_video_filename, codec="libx264", audio_codec="aac")

                # Copy the audio file to the final_audios_folder
                output_audio_filename = os.path.join(final_audios_folder, os.path.basename(audio_file))
                shutil.copy(audio_file, output_audio_filename)

            messagebox.showinfo(
                "Combine Successful", f"Media combined and saved to: {final_videos_folder}")

        except Exception as e:
            print(f"Error during media combination: {e}")
            messagebox.showerror("Combine Error", f"An error occurred while combining media: {e}")



    def build_video_options(self, parent):
        options_label_frame = tk.LabelFrame(
            parent,
            text="Video Prompt Options",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 14, 'bold')
        )
        options_label_frame.pack(fill='both', expand=True, padx=10, pady=10)
        options_label_frame.columnconfigure((0, 1), weight=1)

        # Initialize variables
        self.video_theme_var = tk.StringVar()
        self.video_art_style_var = tk.StringVar()
        self.video_lighting_var = tk.StringVar()
        self.video_framing_var = tk.StringVar()
        self.video_camera_movement_var = tk.StringVar()
        self.video_shot_composition_var = tk.StringVar()
        self.video_time_of_day_var = tk.StringVar()
        self.video_decade_var = tk.StringVar()
        self.video_camera_var = tk.StringVar()
        self.video_lens_var = tk.StringVar()
        self.video_resolution_var = tk.StringVar()
        self.wildlife_animal_var = tk.StringVar()
        self.domesticated_animal_var = tk.StringVar()
        self.video_soundscape_mode_var = tk.BooleanVar()
        self.video_holiday_mode_var = tk.BooleanVar()
        self.video_holidays_var = tk.StringVar()
        self.video_specific_modes_vars = {}
        self.video_no_people_mode_var = tk.BooleanVar()
        self.video_test_mode_var = tk.BooleanVar()
        self.video_remix_mode_var = tk.BooleanVar()
        self.video_story_mode_var = tk.BooleanVar()
        self.video_chaos_mode_var = tk.BooleanVar()

        # Create dropdowns and other widgets
        self.create_dropdown(options_label_frame, "Theme:", THEMES, 0, 0, self.video_theme_var)
        self.create_dropdown(options_label_frame, "Art Style:", ART_STYLES, 1, 0, self.video_art_style_var)
        self.create_dropdown(options_label_frame, "Lighting:", LIGHTING_OPTIONS, 2, 0, self.video_lighting_var)
        self.create_dropdown(options_label_frame, "Framing:", FRAMING_OPTIONS, 3, 0, self.video_framing_var)
        self.create_dropdown(options_label_frame, "Camera Movement:", CAMERA_MOVEMENTS, 4, 0, self.video_camera_movement_var)
        self.create_dropdown(options_label_frame, "Shot Composition:", SHOT_COMPOSITIONS, 5, 0, self.video_shot_composition_var)
        self.create_dropdown(options_label_frame, "Time of Day:", TIME_OF_DAY_OPTIONS, 6, 0, self.video_time_of_day_var)

        # Decade Dropdown
        self.create_dropdown(options_label_frame, "Decade:", DECADES, 7, 0, self.video_decade_var)
        self.video_decade_var.trace('w', self.update_video_camera_options)
        self.video_decade_var.trace('w', self.update_resolution_options)

        # Camera Dropdown
        self.video_camera_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.video_camera_var,
            state="readonly",
            values=CAMERAS[DECADES[0]],
            font=('Helvetica', 12)
        )
        self.video_camera_combobox.grid(row=8, column=1, padx=10, pady=10, sticky='ew')
        self.create_label(options_label_frame, "Camera:", 8, 0)

        # Lens Dropdown
        self.create_dropdown(options_label_frame, "Lens:", LENSES, 9, 0, self.video_lens_var)

        # Resolution Dropdown (Initialize with resolutions from the default decade)
        self.resolution_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.video_resolution_var,
            state="readonly",
            values=RESOLUTIONS[DECADES[0]],
            font=('Helvetica', 12)
        )
        self.resolution_combobox.grid(row=10, column=1, padx=10, pady=10, sticky='ew')
        self.create_label(options_label_frame, "Resolution:", 10, 0)
        self.video_resolution_var.set(RESOLUTIONS[DECADES[0]][0])  # Set default resolution

        # Wildlife and Domesticated Animal Dropdowns
        self.create_dropdown(options_label_frame, "Wildlife Animal:", WILDLIFE_ANIMALS, 11, 0, self.wildlife_animal_var)
        self.create_dropdown(options_label_frame, "Domesticated Animal:", DOMESTICATED_ANIMALS, 12, 0, self.domesticated_animal_var)

        # Prompt Count Selection
        prompt_number_label = tk.Label(
            options_label_frame,
            text="Prompt Count",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        prompt_number_label.grid(row=13, column=0, padx=30, pady=10, sticky='e')

        self.video_prompt_number_var = tk.IntVar(value=DEFAULT_PROMPTS)
        self.video_prompt_number_spinbox = tk.Spinbox(
            options_label_frame,
            from_=1,
            to=MAX_PROMPTS,
            textvariable=self.video_prompt_number_var,
            font=('Helvetica', 12)
        )
        self.video_prompt_number_spinbox.grid(row=13, column=1, padx=10, pady=10, sticky='w')

        # No¹ Mode Checkbox
        self.video_no_people_mode_var = tk.BooleanVar()
        self.video_no_people_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="No¹ Mode - Removes people from the generated content",
            variable=self.video_no_people_mode_var,
            style='TCheckbutton'
        )
        self.video_no_people_checkbox.grid(row=14, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # SoundScape Mode Checkbox
        self.video_soundscape_mode_var = tk.BooleanVar()
        self.video_soundscape_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="SoundScape Mode - Generates dynamic soundscapes focused away from instrumental effects",
            variable=self.video_soundscape_mode_var,
            style='TCheckbutton'
        )
        self.video_soundscape_checkbox.grid(row=14, column=1, columnspan=2, sticky='w', padx=20, pady=2)

        # Story Mode Checkbox
        self.video_story_mode_var = tk.BooleanVar()
        self.video_story_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Story Mode - Ensures prompts flow cohesively as a narrative",
            variable=self.video_story_mode_var,
            style='TCheckbutton'
        )
        self.video_story_mode_checkbox.grid(row=16, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # Chaos Mode Checkbox
        self.video_chaos_mode_var = tk.BooleanVar()
        self.video_chaos_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Chaos Mode - Inverts positive and negative labels",
            variable=self.video_chaos_mode_var,
            style='TCheckbutton'
        )
        self.video_chaos_mode_checkbox.grid(row=16, column=1, columnspan=2, sticky='w', padx=20, pady=2)

        # Remix Mode Checkbox
        self.video_remix_mode_var = tk.BooleanVar()
        self.video_remix_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Remix Mode - COMING SOON",
            variable=self.video_remix_mode_var,
            style='TCheckbutton'
        )
        self.video_remix_mode_checkbox.grid(row=18, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # Holiday Mode Checkbox
        self.video_holiday_mode_var = tk.BooleanVar()
        self.video_holiday_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Holiday Mode - Applies seasonal or holiday-specific themes to prompts",
            variable=self.video_holiday_mode_var,
            style='TCheckbutton',
            command=self.update_video_modes
        )
        self.video_holiday_mode_checkbox.grid(row=18, column=1, columnspan=2, sticky='w', padx=20, pady=2)

        # Holiday Selection Dropdown (initially disabled)
        self.video_holidays_var = tk.StringVar()
        self.video_holidays_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.video_holidays_var,
            state="disabled",  # Initially disabled
            values=HOLIDAYS,
            font=('Helvetica', 12)
        )
        self.video_holidays_combobox.set(HOLIDAYS[0])
        self.video_holidays_combobox.grid(row=19, column=1, padx=10, pady=10, sticky='ew')
        self.create_label(options_label_frame, "Holiday:", 19, 0)

        # Specific Modes Section
        SPECIFIC_MODES_WITH_DESC = {
            # Add specific modes and descriptions here
            # Example:
            # "Mode Name": "Description of the mode."
        }
        self.video_specific_modes_vars = {}

        for idx, (mode, description) in enumerate(SPECIFIC_MODES_WITH_DESC.items()):
            var = tk.BooleanVar()
            ttk.Checkbutton(
                options_label_frame,
                text=mode,
                variable=var,
                style='TCheckbutton'
            ).grid(row=idx * 2, column=0, sticky='w', padx=20, pady=2)

            self.video_specific_modes_vars[mode] = var

            # Description in smaller font under the checkbox
            tk.Label(
                options_label_frame,
                text=description,
                bg='#0A2239',
                fg='light gray',
                font=('Helvetica', 10, 'italic')
            ).grid(row=idx * 2 + 1, column=0, sticky='w', padx=40, pady=(0, 10))

        # Save Button
        save_button = tk.Button(
            parent,
            text="Save Video Options",
            command=self.save_video_options,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 12, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        save_button.pack(pady=10)

        # After initializing all variables, load settings
        self.load_video_settings()
        
    def update_resolution_options(self, *args):
        decade = self.video_decade_var.get()
        resolutions = RESOLUTIONS.get(decade, [])
        if resolutions:
            self.resolution_combobox.config(values=resolutions)
            self.video_resolution_var.set(resolutions[0])  # Set default resolution
        else:
            self.resolution_combobox.config(values=[])
            self.video_resolution_var.set('')

            
    def load_video_settings(self):
        """
        Loads video settings from the SETTINGS_FILE and sets the corresponding variables.
        If the settings file is empty or invalid, it initializes it with default settings.
        """
        if not os.path.exists(SETTINGS_FILE):
            print(f"{SETTINGS_FILE} does not exist. Skipping video settings load.")
            return

        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Invalid or empty {SETTINGS_FILE}, initializing with default settings.")
            # Initialize with default settings and create settings.json
            settings = {
                "video_options": {
                    "theme": THEMES[0],
                    "art_style": ART_STYLES[0],
                    "lighting": LIGHTING_OPTIONS[0],
                    "framing": FRAMING_OPTIONS[0],
                    "camera_movement": CAMERA_MOVEMENTS[0],
                    "shot_composition": SHOT_COMPOSITIONS[0],
                    "time_of_day": TIME_OF_DAY_OPTIONS[0],
                    "decade": DECADES[0],
                    "camera": CAMERAS[DECADES[0]][0],
                    "lens": LENSES[0],
                    "resolution": RESOLUTIONS[DECADES[0]][0],
                    "wildlife_animal": WILDLIFE_ANIMALS[0],
                    "domesticated_animal": DOMESTICATED_ANIMALS[0],
                    "soundscape_mode": False,
                    "holiday_mode": False,
                    "selected_holidays": [],
                    "specific_modes": [],
                    "no_people_mode": False,
                    "test_mode": False,
                    "remix_mode": False,
                    "story_mode": False,
                    "chaos_mode": False
                },
                "audio_options": {
                    "exclude_music": False,
                    "holiday_mode": False,
                    "selected_holidays": [],
                    "specific_modes": [],
                    "open_source_mode": True,
                    "model_name": "audioldm2-full-large-1150k",
                    "guidance_scale": 10.0,
                    "ddim_steps": 100,
                    "n_candidate_gen_per_text": 5,
                    "seed": 12345
                }
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            print("Initialized settings.json with default settings.")
            return

        # Continue processing if settings are valid
        if 'video_options' in settings:
            video_options = settings['video_options']
            self.video_theme_var.set(video_options.get("theme", THEMES[0]))
            self.video_art_style_var.set(video_options.get("art_style", ART_STYLES[0]))
            self.video_lighting_var.set(video_options.get("lighting", LIGHTING_OPTIONS[0]))
            self.video_framing_var.set(video_options.get("framing", FRAMING_OPTIONS[0]))
            self.video_camera_movement_var.set(video_options.get("camera_movement", CAMERA_MOVEMENTS[0]))
            self.video_shot_composition_var.set(video_options.get("shot_composition", SHOT_COMPOSITIONS[0]))
            self.video_time_of_day_var.set(video_options.get("time_of_day", TIME_OF_DAY_OPTIONS[0]))
            self.video_decade_var.set(video_options.get("decade", DECADES[0]))
            self.update_video_camera_options()
            self.video_camera_var.set(video_options.get("camera", CAMERAS[self.video_decade_var.get()][0]))
            self.video_lens_var.set(video_options.get("lens", LENSES[0]))
            self.video_resolution_var.set(video_options.get("resolution", RESOLUTIONS[self.video_decade_var.get()][0]))
            self.wildlife_animal_var.set(video_options.get("wildlife_animal", WILDLIFE_ANIMALS[0]))
            self.domesticated_animal_var.set(video_options.get("domesticated_animal", DOMESTICATED_ANIMALS[0]))
            self.video_soundscape_mode_var.set(video_options.get("soundscape_mode", False))
            self.video_holiday_mode_var.set(video_options.get("holiday_mode", False))

            if self.video_holiday_mode_var.get() and video_options.get("selected_holidays"):
                self.video_holidays_var.set(video_options["selected_holidays"][0])
                self.video_holidays_combobox.config(state="readonly")
            else:
                self.video_holidays_combobox.config(state="disabled")

            for mode in video_options.get("specific_modes", []):
                if mode in self.video_specific_modes_vars:
                    self.video_specific_modes_vars[mode].set(True)

            self.video_no_people_mode_var.set(video_options.get("no_people_mode", False))
            self.video_test_mode_var.set(video_options.get("test_mode", False))
            self.video_remix_mode_var.set(video_options.get("remix_mode", False))
            self.video_story_mode_var.set(video_options.get("story_mode", False))
            self.video_chaos_mode_var.set(video_options.get("chaos_mode", False))

        else:
            print("No video options found in settings.")

            


    def build_audio_options(self, parent):
        """
        Builds the Audio Prompt Options GUI within the provided parent widget.
        This includes dropdowns, checkboxes, and other input elements for configuring audio prompts.
        """
        # Create a labeled frame for audio options
        options_label_frame = tk.LabelFrame(
            parent,
            text="Audio Prompt Options",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 14, 'bold')
        )
        options_label_frame.pack(fill='both', expand=True, padx=10, pady=10)
        options_label_frame.columnconfigure((0, 1), weight=1)  # Make columns expandable

        # Initialize variables for audio options
        self.audio_exclude_music_var = tk.BooleanVar()
        self.audio_holiday_mode_var = tk.BooleanVar()
        self.audio_holidays_var = tk.StringVar()
        self.audio_specific_modes_vars = {}
        self.audio_open_source_mode_var = tk.BooleanVar(value=True)
        self.audio_model_name_var = tk.StringVar(value="audioldm2-full-large-1150k")
        self.audio_guidance_scale_var = tk.DoubleVar(value=10.0)
        self.audio_ddim_steps_var = tk.IntVar(value=100)
        self.audio_n_candidate_var = tk.IntVar(value=5)
        self.audio_seed_var = tk.IntVar(value=12345)
        self.audio_device_var = tk.StringVar(value="cpu")  # Initialize audio_device_var with default 'cpu'
        self.audio_inference_steps_var = tk.IntVar(value=200)  # Default inference steps
        self.audio_length_var = tk.DoubleVar(value=10.0)  # Default audio length in seconds
        self.audio_waveforms_var = tk.IntVar(value=3)  # Default number of waveforms per prompt

        # 1. Exclude Music & Musical Instruments Checkbox
        self.audio_exclude_music_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Exclude Music & Musical Instruments",
            variable=self.audio_exclude_music_var,
            style='TCheckbutton'
        )
        self.audio_exclude_music_checkbox.grid(row=0, column=0, sticky='w', padx=10, pady=5)

        # Description for Exclude Music Mode
        exclude_music_description = tk.Label(
            options_label_frame,
            text="Prevents background music and instrumental sounds from being added to the soundscape.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        exclude_music_description.grid(row=1, column=0, sticky='w', padx=40, pady=(0, 10))

        # 2. Holiday Mode Checkbox
        self.audio_holiday_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Enable Holiday Mode",
            variable=self.audio_holiday_mode_var,
            command=self.update_audio_modes,  # Callback to enable/disable holiday selection
            style='TCheckbutton'
        )
        self.audio_holiday_mode_checkbox.grid(row=2, column=0, sticky='w', padx=10, pady=5)

        # Description for Holiday Mode
        holiday_mode_description = tk.Label(
            options_label_frame,
            text="Adds holiday-themed sounds (e.g., bells, seasonal wind) to the soundscape.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        holiday_mode_description.grid(row=3, column=0, sticky='w', padx=40, pady=(0, 10))

        # 3. Holiday Selection Dropdown (initially disabled)
        self.audio_holidays_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.audio_holidays_var,
            state="disabled",  # Initially disabled
            values=HOLIDAYS,  # Assuming HOLIDAYS is a predefined list
            font=('Helvetica', 12)
        )
        self.audio_holidays_combobox.set(HOLIDAYS[0])  # Set default holiday
        self.audio_holidays_combobox.grid(row=4, column=1, padx=10, pady=10, sticky='ew')
        self.create_label(options_label_frame, "Holiday:", 4, 0)

        # 4. Open-Source Mode Checkbox
        self.audio_open_source_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Enable Open-Source Mode (AudioLDM2)",
            variable=self.audio_open_source_mode_var,
            style='TCheckbutton'
        )
        self.audio_open_source_mode_checkbox.grid(row=5, column=0, sticky='w', padx=10, pady=5)

        # Description for Open-Source Mode
        open_source_mode_description = tk.Label(
            options_label_frame,
            text="Use AudioLDM2 for audio generation.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        open_source_mode_description.grid(row=6, column=0, sticky='w', padx=40, pady=(0, 10))

        # 5. Specific Modes Section (e.g., Remix Mode, Story Mode)
        specific_modes_label = tk.Label(
            options_label_frame,
            text="Specific Modes",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        specific_modes_label.grid(row=7, column=0, sticky='w', padx=10, pady=(10, 5))

        # Define specific modes and their descriptions
        SPECIFIC_AUDIO_MODES_WITH_DESC = {
            "Remix Mode": "Applies creative variations to the audio prompts.",
            "Story Mode": "Ensures prompts flow cohesively as a narrative."
        }
        self.audio_specific_modes_vars = {}

        # Create checkboxes for specific modes
        for idx, (mode, description) in enumerate(SPECIFIC_AUDIO_MODES_WITH_DESC.items(), start=8):
            var = tk.BooleanVar()
            ttk.Checkbutton(
                options_label_frame,
                text=mode,
                variable=var,
                style='TCheckbutton'
            ).grid(row=idx, column=0, sticky='w', padx=20, pady=2)

            self.audio_specific_modes_vars[mode] = var

            # Description in smaller font under the checkbox
            tk.Label(
                options_label_frame,
                text=description,
                bg='#0A2239',
                fg='light gray',
                font=('Helvetica', 10, 'italic')
            ).grid(row=idx+1, column=0, sticky='w', padx=40, pady=(0, 10))

        # 6. Device Selection Dropdown (CPU/GPU)
        self.create_dropdown(
            options_label_frame,
            "Device:",
            ["CPU", "GPU"],
            row=10,
            column=0,
            var=self.audio_device_var
        )

        # 7. Model Name Dropdown
        self.create_dropdown(
            options_label_frame,
            "Model Name:",
            [
                "audioldm2-full-large-1150k",
                "audioldm2-full",
                "audioldm2-music-665k",
                "audioldm2-speech-ljspeech",
                "audioldm2-speech-gigaspeech"
            ],
            row=11,
            column=0,
            var=self.audio_model_name_var
        )

        # 8. Guidance Scale Entry
        self.create_entry(
            options_label_frame,
            "Guidance Scale:",
            self.audio_guidance_scale_var,
            row=12,
            column=0
        )

        # 9. DDIM Steps Entry
        self.create_entry(
            options_label_frame,
            "DDIM Steps:",
            self.audio_ddim_steps_var,
            row=13,
            column=0
        )

        # 10. Number of Candidates Entry
        self.create_entry(
            options_label_frame,
            "Candidates per Text:",
            self.audio_n_candidate_var,
            row=14,
            column=0
        )

        # 11. Seed Entry
        self.create_entry(
            options_label_frame,
            "Seed:",
            self.audio_seed_var,
            row=15,
            column=0
        )

        # 12. Inference Steps Entry
        inference_steps_label = tk.Label(
            options_label_frame,
            text="Inference Steps:",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        inference_steps_label.grid(row=16, column=0, padx=10, pady=5, sticky='e')
        inference_steps_spinbox = tk.Spinbox(
            options_label_frame,
            from_=50,
            to=500,
            textvariable=self.audio_inference_steps_var,
            font=('Helvetica', 12)
        )
        inference_steps_spinbox.grid(row=16, column=1, padx=10, pady=5, sticky='w')

        # 13. Audio Length Entry
        length_label = tk.Label(
            options_label_frame,
            text="Audio Length (s):",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        length_label.grid(row=17, column=0, padx=10, pady=5, sticky='e')
        length_spinbox = tk.Spinbox(
            options_label_frame,
            from_=1.0,
            to=30.0,
            increment=0.5,
            textvariable=self.audio_length_var,
            font=('Helvetica', 12)
        )
        length_spinbox.grid(row=17, column=1, padx=10, pady=5, sticky='w')

        # 14. Number of Waveforms per Prompt
        waveforms_label = tk.Label(
            options_label_frame,
            text="Waveforms per Prompt:",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        waveforms_label.grid(row=18, column=0, padx=10, pady=5, sticky='e')
        waveforms_spinbox = tk.Spinbox(
            options_label_frame,
            from_=1,
            to=10,
            textvariable=self.audio_waveforms_var,
            font=('Helvetica', 12)
        )
        waveforms_spinbox.grid(row=18, column=1, padx=10, pady=5, sticky='w')

        # ---------------------
        # Save Button
        # ---------------------
        save_button = tk.Button(
            parent,
            text="Save Audio Options",
            command=self.save_audio_options,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 12, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        save_button.pack(pady=10)
        # ---------------------
        # Load Settings After Building GUI
        # ---------------------
        self.load_audio_settings()        
    def load_audio_settings(self):
        """
        Loads audio settings from the SETTINGS_FILE and sets the corresponding variables.
        """
        if not os.path.exists(SETTINGS_FILE):
            print(f"{SETTINGS_FILE} does not exist. Skipping audio settings load.")
            return

        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)

        if 'audio_options' in settings:
            audio_options = settings['audio_options']
            self.audio_exclude_music_var.set(audio_options.get("exclude_music", False))
            self.audio_holiday_mode_var.set(audio_options.get("holiday_mode", False))
            
            if self.audio_holiday_mode_var.get() and audio_options.get("selected_holidays"):
                self.audio_holidays_var.set(audio_options["selected_holidays"][0])
                self.audio_holidays_combobox.config(state="readonly")
            else:
                self.audio_holidays_combobox.config(state="disabled")
            
            for mode in audio_options.get("specific_modes", []):
                if mode in self.audio_specific_modes_vars:
                    self.audio_specific_modes_vars[mode].set(True)
            
            self.audio_open_source_mode_var.set(audio_options.get("open_source_mode", True))
            self.audio_model_name_var.set(audio_options.get("model_name", "audioldm2-full-large-1150k"))
            self.audio_guidance_scale_var.set(audio_options.get("guidance_scale", 10.0))
            self.audio_ddim_steps_var.set(audio_options.get("ddim_steps", 100))
            self.audio_n_candidate_var.set(audio_options.get("n_candidate_gen_per_text", 5))
            self.audio_seed_var.set(audio_options.get("seed", 12345))
            self.audio_device_var.set(audio_options.get("device", "cpu"))  # Load device setting
        else:
            print("No audio options found in settings.")

        
    def create_label(self, parent, text, row, column):
        label = tk.Label(
            parent,
            text=text,
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        label.grid(row=row, column=column, padx=10, pady=10, sticky='e')

    def create_entry(self, parent, label_text, var, row, column):
        label = tk.Label(
            parent,
            text=label_text,
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        label.grid(row=row, column=column, padx=10, pady=5, sticky='e')
        entry = tk.Entry(
            parent,
            textvariable=var,
            font=('Helvetica', 12)
        )
        entry.grid(row=row, column=column+1, padx=10, pady=5, sticky='ew')
        return entry

    def create_dropdown(self, parent, label_text, values_list, row, column, var):
        """
        Creates a labeled dropdown (Combobox) within the specified parent widget.

        Args:
            parent (tk.Widget): The parent widget.
            label_text (str): The text for the label.
            values_list (list): The list of values for the dropdown.
            row (int): The row position in the grid.
            column (int): The column position in the grid.
            var (tk.Variable): The tkinter variable associated with the dropdown.
        
        Returns:
            ttk.Combobox: The created Combobox widget.
        """
        label = tk.Label(
            parent,
            text=label_text,
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        label.grid(row=row, column=column, padx=10, pady=10, sticky='e')

        combobox = ttk.Combobox(
            parent,
            textvariable=var,
            state="readonly",
            values=values_list,
            font=('Helvetica', 12)
        )
        combobox.set(values_list[0])  # Set default value
        combobox.grid(row=row, column=column+1, padx=10, pady=10, sticky='ew')
        return combobox  # Return combobox if you need to access it later
        
    def update_video_camera_options(self, *args):
        decade = self.video_decade_var.get()
        cameras = CAMERAS.get(decade, [])
        if cameras:
            self.video_camera_combobox.config(values=cameras)
            self.video_camera_var.set(cameras[0])  # Set default camera
        else:
            self.video_camera_combobox.config(values=[])
            self.video_camera_var.set('')

    def update_video_modes(self):
        if self.video_holiday_mode_var.get():
            self.video_holidays_combobox.config(state="readonly")
        else:
            self.video_holidays_combobox.config(state="disabled")

    def update_audio_modes(self):
        if self.audio_holiday_mode_var.get():
            self.audio_holidays_combobox.config(state="readonly")
        else:
            self.audio_holidays_combobox.config(state="disabled")

    def save_video_options(self):
        self.video_options_set = True
        cinematic_options = {
            "theme": self.video_theme_var.get(),
            "art_style": self.video_art_style_var.get(),
            "lighting": self.video_lighting_var.get(),
            "framing": self.video_framing_var.get(),
            "camera_movement": self.video_camera_movement_var.get(),
            "shot_composition": self.video_shot_composition_var.get(),
            "time_of_day": self.video_time_of_day_var.get(),
            "decade": self.video_decade_var.get(),
            "camera": self.video_camera_var.get(),
            "lens": self.video_lens_var.get(),
            "resolution": self.video_resolution_var.get(),
            "wildlife_animal": self.wildlife_animal_var.get(),
            "domesticated_animal": self.domesticated_animal_var.get(),
            "soundscape_mode": self.video_soundscape_mode_var.get(),
            "holiday_mode": self.video_holiday_mode_var.get(),
            "selected_holidays": [self.video_holidays_var.get()] if self.video_holiday_mode_var.get() else [],
            "specific_modes": [mode for mode, var in self.video_specific_modes_vars.items() if var.get()],
            "no_people_mode": self.video_no_people_mode_var.get(),
            "test_mode": self.video_test_mode_var.get(),
            "remix_mode": self.video_remix_mode_var.get(),
            "story_mode": self.video_story_mode_var.get(),
            "chaos_mode": self.video_chaos_mode_var.get()
        }
        self.save_options_to_file('video_options', cinematic_options)
        self.video_options_window.destroy()
        
    def save_audio_prompt(filename, content):
        # Ensure the directory exists
        if not self.output_folder:
            self.set_output_directory()
        if not self.output_folder:
            return

        audio_save_path = os.path.join(self.output_folder, "Audio")
        os.makedirs(audio_save_path, exist_ok=True)  # Ensure the directory is created

        # Combine the chosen directory with the filename
        full_save_path = os.path.join(audio_save_path, filename)

        # Save the audio content to the selected directory
        with open(full_save_path, 'wb') as f:
            f.write(content)
        print(f"Audio prompt saved to {full_save_path}")


    def save_audio_options(self):
        """
        Save audio options to a JSON file, including all new layer settings.
        """
        audio_options = {
            "exclude_music": self.audio_exclude_music_var.get(),
            "holiday_mode": self.audio_holiday_mode_var.get(),
            "selected_holidays": self.audio_holidays_var.get(),
            "open_source_mode": self.audio_open_source_mode_var.get(),
            "model_name": self.audio_model_name_var.get(),
            "guidance_scale": self.audio_guidance_scale_var.get(),
            "ddim_steps": self.audio_ddim_steps_var.get(),
            "n_candidate": self.audio_n_candidate_var.get(),
            "seed": self.audio_seed_var.get(),
            "device": self.audio_device_var.get(),
            "inference_steps": self.audio_inference_steps_var.get(),
            "audio_length": self.audio_length_var.get(),
            "waveforms_per_prompt": self.audio_waveforms_var.get(),
        }

        with open(SETTINGS_FILE, 'w') as f:
            json.dump({"audio_options": audio_options}, f, indent=4)
        print("Audio options saved successfully.")


    def save_options_to_file(self, key, options):
        """
        Saves the provided options under the specified key in the SETTINGS_FILE.
        
        Args:
            key (str): The key under which to save the options ('video_options' or 'audio_options').
            options (dict): The options to save.
        """
        # Check if the settings file exists
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Handle empty or invalid JSON
                print(f"Invalid or empty {SETTINGS_FILE}, initializing with default settings.")
                settings = DEFAULT_SETTINGS.copy()  # Initialize with default settings
        else:
            settings = DEFAULT_SETTINGS.copy()  # Initialize with default settings if file doesn't exist

        # Update the specific key with the new options
        settings[key] = options

        # Save the updated settings back to the file
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"Saved {key} to {SETTINGS_FILE}.")
        except IOError as e:
            print(f"Failed to write settings to {SETTINGS_FILE}: {e}")
            messagebox.showerror("Save Error", f"Failed to save settings: {e}")


# Main Execution
if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(800, 600)
    app = MultimediaSuiteApp(root)
    root.mainloop()
