import os
import sys
import datetime
import json
import subprocess
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Menu, ttk
from tkinter.scrolledtext import ScrolledText
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips, concatenate_videoclips
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
import queue
import socket
import time
import logging
import scipy
import torch
from diffusers import AudioLDM2Pipeline
from pathlib import Path
import uuid



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
        
        
def parse_model_response(response):
    """
    Parses the model's response to extract the positive sounds and ensures the negative section is empty.

    Args:
        response (str): The raw response from the language model.

    Returns:
        str: Formatted audio prompt with 'positive:' and empty 'negative:' sections.
    """
    # Initialize variables
    positive_content = ""
    negative_content = ""

    # Use regex to find the 'positive:' section
    positive_match = re.search(r'positive:\s*(.*?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
    if positive_match:
        positive_content = positive_match.group(1).strip()
        # Remove any unintended separators or additional prompt sets
        positive_content = re.split(r'[-]{4,}', positive_content)[0].strip()
    else:
        # If 'positive:' is not found, return an empty string
        return ""

    # Use regex to find the 'negative:' section
    negative_match = re.search(r'negative:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if negative_match:
        negative_content = negative_match.group(1).strip()
        # Remove any unintended separators or additional prompt sets
        negative_content = re.split(r'[-]{4,}', negative_content)[0].strip()
    else:
        # If 'negative:' is not found, ensure it's present as empty
        negative_content = ""

    # Return the formatted prompt with 'negative:' empty if necessary
    formatted_prompt = f"positive: {positive_content}\nnegative: {negative_content}"
    return formatted_prompt        

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
        f.write(f"COMFYUI_PROMPTS_FOLDER={COMFYUI_PROMPTS_FOLDER}\n")
        f.write(f"LAST_USED_DIRECTORY={LAST_USED_DIRECTORY}\n")
    print("Settings have been saved to .env file.")

    
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

def clean_prompt_text(raw_text):
    """
    Cleans the raw text response by removing unnecessary whitespace and ensuring proper formatting.

    Args:
        raw_text (str): The raw text response from the API.

    Returns:
        str: The cleaned text.
    """
    # Remove any leading/trailing whitespace
    cleaned_text = raw_text.strip()
    return cleaned_text


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
    self.save_to_file(formatted_prompts_str, file_path)  # Assuming save_to_file is defined  
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
    global OUTPUT_DIRECTORY, LAST_USED_DIRECTORY
    directory = filedialog.askdirectory(title="Select Output Directory")
    if directory:
        OUTPUT_DIRECTORY = directory
        LAST_USED_DIRECTORY = directory  # Update last used directory
        save_settings()
    else:
        messagebox.showwarning("Output Directory", "No directory selected.")



# Function to set the ComfyUI INSPIRE prompts folder
def set_comfyui_prompts_folder():
    global COMFYUI_PROMPTS_FOLDER
    directory = filedialog.askdirectory(title="Select ComfyUI INSPIRE Prompts Folder")
    if directory:
        COMFYUI_PROMPTS_FOLDER = directory
        save_settings()
    else:
        messagebox.showwarning("ComfyUI Prompts Folder", "No directory selected.")


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
    """
    Enables a given Tkinter button.

    Args:
        button (tk.Button): The button to enable.
    """
    button.config(state='normal')

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

THEMES = [

    "Historical & Period Genres - Detailed historical visuals with emphasis on authenticity",
    "Historical - Authentic period visuals with meticulous attention to costumes and sets",
    "Historical Epic - Grandiose visuals with large-scale sets and crowd scenes",
    "Historical Fiction - Accurate period visuals blended with creative storytelling",
    "Period Piece - Detailed historical visuals with emphasis on authenticity",
    "Period Drama - Authentic visuals immersing viewers in a specific era",
    "Period Romance - Historical visuals with romantic and elegant aesthetics",
    "Sword and Sandal - Ancient world visuals with heroic figures and epic battles",
    "Neo-Western - Contemporary visuals with themes and aesthetics of classic Westerns",
    "Western - Expansive desert vistas and rustic settings capturing frontier life",
    "Spaghetti Western - Stylized Westerns with dramatic close-ups and wide landscapes",
    "Film Noir - High-contrast black-and-white imagery with dramatic shadow play",
    "Neo-Noir - Modern reinterpretation of noir with stylized lighting and urban settings",
    "German Expressionism - Stylized visuals with sharp angles and high contrasts",
    "Italian Neorealism - Naturalistic visuals with on-location shooting and non-actors",
    
    "Educational & Instructional - Clear and straightforward visuals designed to inform",
    "Educational - Clear and straightforward visuals designed to inform",
    "Documentary Drama - Realistic visuals with narrative elements",
    "Science Documentary - Clear and informative visuals explaining complex ideas",
    "Nature Documentary - Stunning visuals capturing wildlife and environments",
    "Educational Documentary - Clear and engaging visuals conveying information",
    "Environmental Documentary - Visuals highlighting ecological concerns",
    "Wildlife Documentary - Close-up visuals of animals in their habitats",

    "Documentary & Realism - Real-world footage with natural lighting and minimal staging",
    "Documentary - Real-world footage with natural lighting and minimal staging",
    "Travel Documentary - Visual journey through different cultures and places",
    "Nature Documentary - Stunning visuals capturing wildlife and environments",
    "Educational Documentary - Clear and engaging visuals conveying information",
    "Science Documentary - Informative visuals explaining scientific topics",
    "Crime Documentary - Investigative visuals exploring criminal activities",
    "True Crime - Realistic visuals delving into actual criminal cases",
    "Sports Documentary - Dynamic visuals showcasing athletic prowess",
    "Music Documentary - Captivating visuals of performances and behind-the-scenes",
    "Biographical Documentary - Real-life visuals chronicling individuals",
    "Wildlife Documentary - Close-up visuals of animals in their habitats",
    "Environmental Documentary - Visuals highlighting ecological concerns",
    "Cultural Documentary - Visual immersion into different societies",
    "Historical Documentary - Authentic visuals exploring past events",

    "Biographical & Real-Life - Realistic visuals focusing on accurate representations",
    "Biographical - Authentic visuals focusing on accurate representations of real-life settings",
    "Historical Drama - Period visuals with dramatic narratives",
    "Biographical Documentary - Real-life visuals chronicling individuals",
    "Coming of Age - Relatable visuals focusing on personal growth and self-discovery",
    "Social Commentary - Realistic visuals highlighting societal issues",
    "Slice of Life - Everyday visuals portraying ordinary experiences",
    "Independent - Resourceful visuals with creative approaches due to budget constraints",

    "Action & Adventure - High-energy visuals with dynamic camera movements and fast-paced editing",
    "Action - High-energy visuals with dynamic camera movements and fast-paced editing",
    "Adventure - Expansive landscapes and sweeping shots that emphasize epic journeys",
    "Martial Arts - Fluid camera work capturing choreographed fight sequences",
    "Swashbuckler - Energetic visuals featuring swordplay and daring feats",
    "Pirate - Nautical visuals with ships, the sea, and adventurous settings",
    "Kung Fu - Precise and fluid visuals emphasizing martial arts techniques",
    "Wuxia - Chinese martial arts with poetic visuals and wirework",
    "Samurai - Japanese historical visuals with emphasis on honor and combat",
    "Sword and Sorcery - Medieval-inspired visuals with magical elements and epic battles",
    "Sword and Sandal - Ancient world visuals with heroic figures and epic battles",

    "Comedy & Satire - Bright lighting and vibrant colors to enhance humorous elements",
    "Comedy - Bright lighting and vibrant colors to enhance humorous elements",
    "Satire - Exaggerated visuals and stylized imagery to highlight absurdities",
    "Parody - Mimicking the visual style of other genres for comedic effect",
    "Romantic Comedy - Light-hearted visuals with warm tones and urban settings",
    "Black Comedy - Contrasting dark themes with bright visuals to enhance irony",
    "Satirical Comedy - Exaggerated visuals critiquing societal norms",
    "Mockumentary - Documentary-style visuals used for satirical or comedic effect",
    "Postmodern Comedy - Self-referential visuals breaking the fourth wall",
    "Adventure Comedy - Light-hearted visuals with a sense of excitement and fun",
    "Road Comedy - Changing settings providing a backdrop for humor",
    "Musical Comedy - Energetic and colorful visuals enhancing musical numbers",

    "Crime & Thriller - Dark visuals exploring criminal underworlds",
    "Crime - Gritty urban settings with stark contrasts and shadowy lighting",
    "Crime Drama - Urban visuals focusing on moral complexities",
    "Detective - Dark urban visuals with an emphasis on clues and investigative elements",
    "Heist - Slick visuals with precise framing highlighting intricate plans",
    "Mystery Thriller - Shadowy visuals and tight editing to maintain suspense",
    "Conspiracy Thriller - Shadowy visuals creating a sense of paranoia",
    "Political Thriller - Intriguing visuals within the sphere of politics",
    "Techno Thriller - Modern visuals highlighting advanced technology and its implications",
    "Supernatural Thriller - Eerie visuals with unexplained phenomena",
    "Action Thriller - High-intensity visuals combining action and tension",
    "Science Fiction Thriller - Futuristic visuals with suspenseful narratives",
    "Adventure Thriller - Exotic visuals with thrilling plots",
    "Crime Thriller - Dark visuals exploring criminal underworlds",
    "Environmental Thriller - Natural visuals highlighting ecological threats",
    "Historical Thriller - Period visuals with suspenseful storytelling",
    "Urban Thriller - Cityscapes with a sense of urgency and danger",
    "Rural Thriller - Remote settings enhancing isolation and suspense",

    "Drama & Melodrama - Visually rich settings enhancing emotional connections",
    "Drama - Intimate cinematography focusing on character expressions and emotions",
    "Family Drama - Warm visuals focusing on interpersonal relationships",
    "Political Drama - Formal visuals within governmental settings",
    "War Drama - Realistic and intense visuals depicting the impacts of war",
    "Coming of Age Drama - Visuals capturing the transition from youth to adulthood",
    "Melodrama - Heightened emotions conveyed through expressive visuals",
    "Psychological Drama - Deep focus on characters through intimate visuals",
    "Urban Drama - Realistic city visuals focusing on societal issues",
    "Rural Drama - Naturalistic visuals highlighting country life",
    "Espionage Drama - International visuals with an air of secrecy",
    "Medical Drama - Clinical visuals within hospital settings",
    "Legal Drama - Courtroom visuals focusing on justice and conflict",

    "Fantasy & Science Fiction - Grand and immersive visuals with extensive world-building",
    "Fantasy - Imaginative worlds with rich, detailed visuals and special effects",
    "Science Fiction - Futuristic designs with innovative technology and visual effects",
    "Epic Fantasy - Grand and immersive visuals with extensive world-building",
    "Dark Fantasy - Fantastical visuals with ominous and gothic undertones",
    "Science Fantasy - Blending futuristic technology with fantastical elements in visuals",
    "Sword and Sorcery - Medieval-inspired visuals with magical elements and epic battles",
    "Urban Fantasy - Modern city visuals infused with magical elements",
    "Science Fiction Comedy - Futuristic visuals presented with humor",
    "Science Fantasy Adventure - Visuals combining technological and magical elements",
    "Space Opera - Expansive cosmic visuals with elaborate spacecraft and alien worlds",
    "Time Travel Adventure - Diverse visuals spanning multiple eras",
    "Alternate History - Familiar visuals altered to reflect a different reality",
    "Dystopian - Bleak and oppressive settings reflecting a degraded society",
    "Utopian - Idealized visuals with harmonious and clean designs",
    "Magical Realism - Realistic settings infused with subtle fantastical elements",
    "Superhero - Bold and dynamic visuals with emphasis on action and special effects",
    "Technothriller - Modern visuals highlighting advanced technology and its implications",
    "Cyberpunk - Neon-lit urban environments with a high-tech, low-life aesthetic",
    "Steampunk - Victorian-era visuals infused with steam-powered technology",

    "Horror & Suspense - Disturbing visuals that unsettle by exploring the mind",
    "Horror - Atmospheric lighting and suspenseful framing to evoke fear",
    "Psychological Horror - Disturbing visuals that unsettle by exploring the mind",
    "Found Footage Horror - Authentic visuals creating immediacy and realism",


    "Mystery & Suspense - Shadowy visuals creating a sense of intrigue",
    "Mystery - Obscured visuals and tight framing to create intrigue",
    "Film Noir - High-contrast black-and-white imagery with dramatic shadow play",
    "Detective Noir - Shadow-filled visuals emphasizing mystery",
    "Mystery Thriller - Shadowy visuals and tight editing to maintain suspense",
    "Psychological Thriller - Unsettling visuals creating tension and ambiguity",
    "Gothic - Dark, ornate visuals with high-contrast lighting and elaborate costumes",
    "Supernatural Thriller - Eerie visuals with unexplained phenomena",
    "Thriller Drama - Suspenseful visuals enhancing dramatic tension",
    "Courtroom Drama - Formal visuals focusing on dialogue and character interactions",
    "Legal Thriller - Tense courtroom visuals with suspenseful elements",

    "Experimental & Arthouse - Artistic visuals creating a unique and unsettling experience",
    "Arthouse - Focus on aesthetic experimentation and visual artistry over mainstream appeal",
    "Experimental - Non-traditional visuals employing abstract and avant-garde techniques",
    "Avant-Garde - Innovative and abstract visuals challenging conventional cinema",
    "Art Film - Emphasis on visual expression and thematic depth over commercial appeal",
    "Expressionism - Distorted sets and dramatic lighting conveying emotional states",
    "Absurdist - Surreal and illogical visuals challenging perceptions of reality",
    "Surrealism - Dream-like imagery with abstract and symbolic visuals",
    "Dogme 95 - Naturalistic visuals with hand-held cameras and natural light",
    "Art House - Emphasis on visual storytelling with symbolic and metaphorical imagery",
    "Existential - Minimalist visuals exploring themes of existence and meaning",
    "Postmodern - Blending of styles and self-referential visuals challenging norms",
    "Found Footage - Handheld camera work creating a raw and immediate aesthetic",
    "Mumblecore - Minimalist visuals with natural lighting and improvised feel",
    "Film Essay - Visuals used to explore and present ideas or arguments",

    "Musical & Dance - Dynamic choreography captured with fluid camera movements",
    "Musical - Vibrant colors and dynamic choreography captured with fluid camera movements",
    "Musical Comedy - Energetic and colorful visuals enhancing musical numbers",
    "Musical Drama - Emotional storytelling enhanced by musical performances",
    "Dance Film - Kinetic visuals capturing movement and rhythm",
    "Musical Anthology - Different musical styles showcased visually",

    "Epic & Grand - Grand scale productions with sweeping cinematography and elaborate sets",
    "Epic - Grand scale productions with sweeping cinematography and elaborate sets",
    "Historical Epic - Grandiose visuals with large-scale sets and crowd scenes",
    "Space Opera - Expansive cosmic visuals with elaborate spacecraft and alien worlds",
    "Sword and Sandal - Ancient world visuals with heroic figures and epic battles",
    "Grand Adventure - Visually compelling journeys with character depth",
    "Mythological Adventure - Epic visuals rooted in ancient tales",

    "Youth & Family - Visually compelling journeys with character depth",
    "Teen Drama - Contemporary visuals reflecting youth culture",
    "Coming of Age - Relatable visuals focusing on personal growth and self-discovery",
    "Children's - Bright and colorful visuals appealing to a younger audience",
    "Family - Warm and inviting visuals suitable for all ages",

    "Science & Technology - Realistic visuals grounded in scientific accuracy",
    "Science Fiction - Futuristic designs with innovative technology and visual effects",
    "Hard Science Fiction - Realistic visuals grounded in scientific accuracy",
    "Military Science Fiction - Futuristic warfare visuals with advanced weaponry",
    "Space Exploration - Awe-inspiring visuals of space and celestial bodies",
    "Alien Invasion - Dramatic visuals depicting extraterrestrial encounters",
    "Psychological Science Fiction - Mind-bending visuals exploring complex ideas",
    "Biopunk - Organic and biological visuals showcasing genetic manipulation",
    "Technothriller - Modern visuals highlighting advanced technology and its implications",
    "Cyberpunk Thriller - Neon-lit visuals with high-tech and gritty urban environments",
    "Science Mystery - Enigmatic visuals presenting puzzles and phenomena"

]


ART_STYLES = [

    "Documentary Style",
    "Realism",
    "Retro Futurism",
    "Cyberpunk Aesthetic",
    "Steampunk Aesthetic",
    "Gothic Aesthetic",
    "Post-apocalyptic Aesthetic",
    "Fantasy Aesthetic",
    "Futuristic Aesthetic",
    "Dystopian Aesthetic",
    "Utopian Aesthetic",
    "Magical Realism",

    # Art Movements and Styles
    "Surrealism",
    "Expressionism",
    "Realism",
    "Impressionism",
    "Minimalism",
    "Abstract",
    "Cubism",
    "Pop Art",
    "Dadaism",
    "Futurism",
    "Constructivism",
    "Baroque",
    "Gothic",
    "Romanticism",
    "Symbolism",
    "Art Deco",
    "Art Nouveau",
    "Bauhaus",
    "Kinetic Art",
    "Op Art",
    "Neo-Expressionism",
    "Digital Art",
    "Glitch Art",
    "Pixel Art",
    "Vaporwave",
    "Cyberpunk",
    "Steampunk",
    "Low Poly",
    "High Fantasy",
    "Anime Style",
    "Comic Book Style",
    "Graphic Novel Style",
    "Watercolor Style",
    "Ink Wash Style",
    "Sketch Style",
    "Photorealism",
    "Collage",
    "Mixed Media",
    "Street Art",
    "Graffiti",
    "Tattoo Art",
    "Mosaic",
    "Stained Glass",
    "Typography",
    "Calligraphy"
]

LIGHTING_OPTIONS = [
    "Contextual Lighting - Fits the scene’s purpose and setting",

    "Natural Lighting - Ambient and organic light sources",
    "Sunlight - Bright and natural outdoor lighting",
    "Golden Hour - Warm sunset tones for soft, dramatic visuals",
    "Blue Hour - Cool twilight hues with soft contrasts",
    "Moonlight - Gentle, bluish nocturnal glow",
    "Candlelight - Warm, flickering light source",
    "Firelight - Natural and dynamic flame light",

    "Studio Lighting - Controlled environment setups",
    "Softbox Lighting - Even, diffused light for portraits",
    "Ring Light - Circular lighting emphasizing facial features",
    "Spotlight - Focused illumination for dramatic emphasis",
    "Continuous Lighting - Steady, reliable light for various scenes",
    "Strobe Lighting - Flash bursts for high contrast effects",
    "LED Lighting - Energy-efficient and versatile lighting source",
    "RGB Lighting - Customizable, multi-color lighting options",

    "Directional Lighting - Angle-specific light sources",
    "Side Lighting - Creates shadows and depth from the side",
    "Top Lighting - Overhead light for stark shadows",
    "Underlighting - Lighting from below, often for eerie effects",
    "Front Lighting - Direct lighting from the front",
    "Backlighting - Silhouette effect with light from behind",
    "Rim Lighting - Highlights edges for separation from background",
    "Cross Lighting - Two light sources creating dynamic shadows",
    "Split Lighting - Half light, half shadow effect",
    "Loop Lighting - Subtle cheek shadows, popular for portraits",

    "High & Low Key Lighting - Setting mood with light intensity",
    "High Key - Bright, low-contrast lighting for a light mood",
    "Low Key - Dark, high-contrast lighting for drama",
    "Chiaroscuro - Strong contrast for depth and intensity",
    "Rembrandt Lighting - Classic portrait with triangular shadows",
    "Ambient Lighting - Overall soft background light",
    "Motivated Lighting - Scene context-based lighting",

    "Colored & Thematic Lighting - Setting scenes with color and style",
    "Neon Lighting - Vibrant, urban colors",
    "Colored Gel Lighting - Color filters for mood setting",
    "Candle Glow - Warmth and coziness",
    "Blacklight - Ultraviolet for eerie or otherworldly effects",
    "Vintage Lighting - Nostalgic, warm tones",
    "Modern Lighting - Sleek, contemporary lighting style",
    "Futuristic Lighting - Cool, high-tech visuals",
    "Genre-Specific Lighting - Tailored for specific film genres",

    "Dynamic & Adaptive Lighting - Interactive and changing lighting setups",
    "Reactive Lighting - Changes with motion",
    "Timed Lighting - Scheduled, automated changes",
    "Transitional Lighting - Smooth intensity shifts",
    "Shadow Play - Artistic use of shadows",
    "Light Painting - Artistic, controlled light trails",

    "Specialized Lighting Techniques - Unique lighting styles",
    "Silhouette - Outlines only, background illuminated",
    "Gobo Lighting - Patterned lighting for texture",
    "Grid Lighting - Focused beams for isolation",
    "Projected Lighting - Shapes or patterns cast onto subjects",
    "Butterfly Lighting - Shadow under nose, flattering for portraits",
    "Split Lighting - Dividing face into light and dark halves",
    "Color Gradient Lighting - Smooth transitions across colors",

    "Practical Lighting - On-set, naturally motivated lighting",
    "Functional Practical Lighting - Utility-based lighting for realism",
    "Symbolic Practical Lighting - Adds meaning or enhances narrative",
    "Decorative Practical Lighting - Adds aesthetic appeal",
    "Thematic Practical Lighting - Matches set theme or style",
    "Mood Practical Lighting - Sets emotional tone",
    "Accent Lighting - Highlights specific details or features",

    "Control & Adjustment Lighting - Fine-tuning and precision",
    "Dimmed Lighting - Lower intensity for a subdued look",
    "Bright Lighting - High intensity for clear visibility",
    "Diffused Light - Soft, scattered illumination",
    "Bounce Lighting - Reflects light softly onto the subject",
    "Fill Lighting - Reduces shadows",
    "Directional Light - Specific angled light source",
    "Temperature Controlled Lighting - Adjusts warmth or coolness",
    "Intensity Controlled Lighting - Adjusts light strength",
    "Focus Controlled Lighting - Sharpens specific areas",
    "Shadowless Lighting - Eliminates shadows",
    "Matte Lighting - Soft, glare-free light",
    "Glossy Lighting - Reflective lighting for shine",

    "Atmospheric & Narrative Lighting - Creating mood and storytelling",
    "Atmospheric Lighting - Sets an environmental mood",
    "Narrative Lighting - Enhances story elements",
    "Symbolic Lighting - Represents abstract ideas",
    "Mood Lighting - Sets the emotional tone",
    "Thematic Lighting - Aligned with the scene’s theme",
    "Environmental Lighting - Matches surroundings realistically",

    "Edge & Rim Lighting - Contour and separation",
    "Edge Lighting - Highlights object edges",
    "Accent Rim Lighting - Soft edge highlights",
    "Diffuse Rim Lighting - Blurred edge emphasis",
    "Bounced Rim Lighting - Soft reflection from background",

    "Experimental & Creative Lighting - Innovative and bold styles",
    "Layered Lighting - Multiple lighting layers for complexity",
    "Patterned Lighting - Light textures and patterns",
    "Animated Lighting - Moving or fluctuating light sources",
    "Integrated Lighting - Seamless blending of sources",
    "Artistic Lighting - Creative, expressive light setups",
    "Technical Lighting - Precision and exactness in light control",
    "Balanced Lighting - Evenly spread light",
    "Asymmetrical Lighting - Uneven light distribution",
    "Complementary Lighting - Harmonious colors",
    "Light Blending - Smooth merging of light sources",

]

FRAMING_OPTIONS = [
    "Contextual Shot - Adapts to scene context",

    "Basic Shots - Standard framing and perspectives",
    "Wide Shot - Broad view of the scene",
    "Medium Shot - Waist level framing",
    "Close-Up - Detailed focus on subject",
    "Extreme Close-Up - Focus on minute details",
    "Long Shot - Full body view",
    "Full Shot - Complete figure in frame",
    "Headshot - Focus on head and shoulders",

    "Specialty Shots - Unique perspectives and framing",
    "Over-the-Shoulder - Perspective from behind subject",
    "Dutch Angle - Tilted frame for tension",
    "Bird's Eye View - Top-down angle",
    "Point of View - First-person perspective",
    "Two-Shot - Dual subjects in frame",
    "Cowboy Shot - Mid-thigh framing for stance",
    "Establishing Shot - Sets the scene context",
    "Aerial Shot - High elevation view of scene",
    "Insert Shot - Close detail of object or action",
    "Reverse Angle - Shows opposite perspective",

    "Movement Shots - Camera movement and dynamics",
    "Tracking Shot - Follows subject's movement",
    "Static Shot - Fixed camera position",
    "Pan Shot - Horizontal movement across scene",
    "Tilt Shot - Vertical movement up or down",
    "Zoom Shot - Adjusts focal distance",
    "Crane Shot - Camera elevated movement",
    "Dolly Shot - Smooth tracking movement",
    "Arc Shot - Circular movement around subject",
    "Rack Focus - Adjusts focus depth within scene",
    "Push In Shot - Moves camera closer to subject",
    "Pull Back Shot - Camera retreats from subject",
    
    "Composition Shots - Structured visual elements",
    "Symmetrical Composition - Balanced and mirrored sides",
    "Asymmetrical Composition - Uneven yet balanced layout",
    "Rule of Thirds Composition - Balanced along thirds",
    "Golden Ratio Composition - Natural proportional layout",
    "Centered Composition - Main focus in center of frame",
    "Negative Space Composition - Emphasizes emptiness",
    "Layered Composition - Multiple depth layers",
    "Foreground Interest - Elements in the foreground",
    "Background Interest - Emphasis on background elements",
    
    "Angle Shots - Perspective-based camera angles",
    "High Angle Shot - Camera looks down on subject",
    "Low Angle Shot - Camera looks up at subject",
    "Worm's Eye View - Ground-up perspective",
    "Extreme High Angle - Very elevated top-down view",
    "Extreme Low Angle - Very low, ground-level view",
    "Overhead Shot - Directly above subject",
    
    "Lens Effects - Optical and perspective adjustments",
    "Wide Angle Shot - Broad perspective and depth",
    "Telephoto Shot - Narrow focus on distant subjects",
    "Macro Shot - Close-up for small details",
    "Fisheye Shot - Distorted, rounded wide view",
    "Tilt-Shift Shot - Miniature, selective focus effect",
    "Anamorphic Shot - Wide, cinematic view",
    "Soft Focus - Slightly blurred edges for atmosphere",
    "Infrared Shot - Uses heat-sensitive imagery",
    "UV Shot - Captures ultraviolet light spectrum",

    "Special Effects Shots - Enhanced or stylized techniques",
    "Time-Lapse Shot - Accelerated time for changes",
    "Hyperlapse Shot - Accelerated movement across distance",
    "Slow Motion Shot - Reduces speed for emphasis",
    "Fast Motion Shot - Increases speed for urgency",
    "Frozen Time Shot - Static, suspended moment",
    "Bullet Time Shot - 360-degree view around subject",

    "Composition Techniques - Creative framing and layout",
    "Leading Lines - Guides viewer’s eye through frame",
    "Framing Within Framing - Nested, layered frames",
    "Silhouette Composition - Dark outline with bright background",
    "Reflection Composition - Mirror or reflective elements",
    "Pattern Composition - Repeated shapes or motifs",
    "Geometric Composition - Emphasis on shapes",
    "Organic Composition - Natural, flowing forms",
    "Minimalist Composition - Simple, clean elements",
    "Dynamic Composition - Active, engaging layout",
    "Static Composition - Fixed, balanced layout",
    "Juxtaposition Composition - Contrasting elements side by side",
    "Color Contrast Composition - Vibrant hue differences",
    
    "Layering Techniques - Depth and spatial relationships",
    "Foreground Element Shot - Subject in front of frame",
    "Background Element Shot - Emphasis on background",
    "Balanced Layering - Even distribution of elements",
    "Asymmetrical Layering - Uneven arrangement",
    "Central Layering - Focus on central layer",
    "Peripheral Layering - Emphasis on edges or background",
    
    "Space and Depth Shots - Perspective manipulation",
    "Depth of Field Shot - Clear foreground, blurred background",
    "Shallow Depth - Limited focus on foreground",
    "Deep Depth - Extended focus on entire frame",
    "Spatial Composition - Creates 3D depth",
    "Flat Composition - Emphasis on 2D elements",
    
    "Framing Techniques - Visual structure and context",
    "Balanced Framing - Equal distribution within frame",
    "Dynamic Framing - Engaging, active visual layout",
    "Narrative Framing - Enhances storytelling elements",
    "Symbolic Framing - Represents thematic ideas",
    "Thematic Framing - Aligned with scene’s theme",
]

CAMERA_MOVEMENTS = [
    "Contextual Movement - Adapts to scene needs",

    "Basic Movements - Core camera motions",
    "Pan - Horizontal sweep across a scene",
    "Tilt - Vertical sweep up or down",
    "Dolly - Moving towards or away from the subject",
    "Truck - Lateral (side-to-side) movement",
    "Zoom - Adjusting focal length to change perspective",
    "Crane - Elevating or lowering the camera smoothly",
    "Handheld - Unsteady, naturalistic motion",
    "Steadicam - Smooth tracking without shake",
    "Tracking Shot - Follows the subject’s movement",
    "Arc Shot - Circular motion around the subject",

    "Dynamic Movements - Quick or impactful adjustments",
    "Whip Pan - Rapid horizontal sweep",
    "Whip Tilt - Rapid vertical sweep",
    "Rack Focus - Shifts focus between planes",
    "Pull Back Reveal - Gradually reveals scene",
    "Push In - Moves closer to the subject",
    "Roll - Rotational movement for disorientation",
    "Boom - Overhead vertical reach",

    "Time-Based Movements - Altering perception of time",
    "Time-Lapse - Accelerated time to capture change",
    "Hyperlapse - Moving time-lapse across space",
    "Slow Motion - Reduced speed for emphasis",
    "Fast Motion - Increased speed for urgency",
    "Frozen Time - Static frame capturing a moment",
    "Bullet Time - 360-degree freeze around the subject",

    "Complex Movements - Combined or advanced techniques",
    "Pan and Tilt - Horizontal and vertical sweep combination",
    "Dolly Zoom - Changes focal length to distort depth",
    "Orbit Shot - Full circular movement around subject",
    "Jib Shot - Vertical crane movement with arc capabilities",
    "Helicopter Shot - High aerial sweeping motion",
    "Cable Cam Movement - Suspended, controlled path",
    "Monorail Movement - Fixed track-guided motion",
    "Slide Shot - Side-to-side glide without shake",

    "Adaptive & Controlled Movements - Guided or automated",
    "Motorized Steadicam - Mechanized smooth motion",
    "Drone Movement - Free aerial paths",
    "Gyroscopic Movement - Stabilized rotation",
    "Gimbal Stabilized Movement - Multi-axis smoothness",
    "RC Camera Movement - Remote-controlled paths",
    "Underwater Camera Movement - Fluid, aquatic paths",
    "Macro Camera Movement - Small, precise adjustments",
    "Wide Camera Movement - Expansive sweeps",
    "Narrow Camera Movement - Focused close-ups",

    "Creative & Expressive Movements - Artistic effects",
    "Fluid Movement - Smooth and natural transitions",
    "Staccato Movement - Quick, jerky stops",
    "Synchronized Movement - Coordinated with subject",
    "Random Movement - Unpredictable motion for effect",
    "Narrative Movement - Enhances storytelling elements",
    "Symbolic Movement - Adds thematic significance",
    "Expressive Movement - Emotionally driven motion",
    "Functional Movement - Purpose-driven adjustment",

    "Transition Movements - Changing from one shot to another",
    "Smooth Transition Movement - Seamless camera changes",
    "Abrupt Transition Movement - Sudden shift",
    "Gradual Transition Movement - Slow, drawn-out change",
    "Creative Transition Movement - Unique or artistic swap",
    "Narrative Transition Movement - Enhances story development",

    "Advanced Tracking Movements - Following specific paths",
    "Reverse Tracking - Moving in opposite direction of subject",
    "Side Tracking - Lateral follow alongside subject",
    "Overhead Tracking - Elevated tracking from above",
    "Ground Tracking - Low, ground-level follow",
    "Subject Tracking - Focused on main subject’s movement",
    "Environment Tracking - Following the setting or background",
    "Dynamic Tracking - Variable speed and fluidity",
    "Static Tracking - Fixed speed follow",
    "Smooth Tracking - Even, steady following",
    
    "Technical & Specialized Movements - Precise applications",
    "Predictive Movement - Anticipates subject’s motion",
    "Reactive Movement - Responds to subject’s actions",
    "Controlled Movement - Managed, stabilized paths",
    "Uncontrolled Movement - Free-form, organic paths",
    "Automated Movement - Pre-programmed motion",
    "Manual Movement - Hand-operated paths",
    "Adaptive Movement - Adjusts to subject’s position",

    "Multi-Directional Movements - Complex or intersecting paths",
    "Figure-Eight Movement - Intersecting circular paths",
    "Circular Tracking - Complete round path around subject",
    "Linear Tracking - Straight follow along a path",
    "Diagonal Tracking - Angled follow for unique perspective",
    "Omni-Directional Tracking - Full range of motion around subject",
    "Intersecting Tracking - Crossing paths for layered effect",
    "Continuous Tracking - Unbroken path for uninterrupted follow",
    "Intermittent Tracking - Occasional stops for emphasis",
    
    "Composition & Framing Movements - Adjusts perspective and layout",
    "Symmetrical Movement - Balanced, mirror-like motion",
    "Asymmetrical Movement - Uneven path for dynamic composition",
    "Easing Movement - Smooth start and stop",
    "Abrupt Movement - Sudden start or end",
    "Smooth Transition Movement - Seamless blend between shots",
    "Creative Framing - Frames subject with artistic alignment",
    "Rule of Thirds Framing - Aligns subject on balanced thirds",
]

SHOT_COMPOSITIONS = [
    "Contextual Framing - Adapts to scene context",

    "Basic Framing Techniques - Core visual structures",
    "Rule of Thirds - Aligns subject on balanced thirds",
    "Symmetrical - Mirror balance for harmony",
    "Asymmetrical - Uneven balance for dynamism",
    "Centered Composition - Main focus centered in frame",
    "Golden Ratio - Natural, proportional layout",
    "Framing - Creates a frame within a frame for depth",
    "Leading Lines - Directs viewer's gaze through the frame",
    "Diagonal Composition - Slanted lines for energy",
    "Negative Space - Emphasizes emptiness around subject",
    
    "Depth and Layering - Adds dimension and spatial relationships",
    "Depth - Uses focus layers for a 3D effect",
    "Layering - Multiple visual layers for depth",
    "Foreground Interest - Subject placed prominently in front",
    "Background Interest - Emphasizes background details",
    "Rule of Odds - Odd number of elements for visual appeal",
    "Golden Spiral - Curved composition for flow",
    "Frame within Frame - Nested framing adds dimension",

    "Balance and Symmetry - Structured visual harmony",
    "Dynamic Symmetry - Active, balanced composition",
    "Balanced Composition - Even distribution of elements",
    "Unbalanced Composition - Uneven distribution for focus",
    "Symmetrical Balance - Mirror-like element placement",
    "Radial Balance - Elements emanating from a central point",
    "Visual Weight - Importance given to specific elements",

    "Contrast and Juxtaposition - Creates visual tension and interest",
    "Juxtaposition - Contrast between elements",
    "Color Contrast - Differing hues for emphasis",
    "Light Contrast - Bright vs dark for depth",
    "Texture Contrast - Smooth vs rough textures",
    "Shape Contrast - Geometric vs organic forms",
    "Size Contrast - Large vs small elements",
    "Focus Contrast - Sharp vs blurred areas",

    "Patterns and Repetition - Adds rhythm and structure",
    "Patterns - Repeating motifs for rhythm",
    "Repetition - Iterated elements to guide viewer’s eye",
    "Geometric Patterns - Structured shapes for order",
    "Organic Patterns - Natural shapes for fluidity",
    "Texture - Surface details add interest",

    "Visual Path and Flow - Guides viewer’s gaze through the scene",
    "Visual Path - Directs eye movement through frame",
    "Movement - Suggests action or leads viewer’s focus",
    "Rhythm - Repeating elements create visual beat",
    "Hierarchy - Organized by element importance",
    "Harmony - Unified, cohesive elements",
    "Contrast - Differing elements for emphasis",
    "Emphasis - Highlighted focus for importance",

    "Shape and Form - Defines structure within the composition",
    "Circular Composition - Round elements for unity",
    "Triangular Composition - Three-point focus adds stability",
    "Linear Composition - Straight lines for direction",
    "Zigzag Composition - Angled lines add energy",
    "Radiant Composition - Spreading lines for focus",
    "Grid Composition - Structured, ordered layout",

    "Textural and Color Harmony - Blends elements through cohesive tones",
    "Color Harmony - Unified color scheme for cohesion",
    "Texture Harmony - Similar textures for unity",
    "Shape Harmony - Consistent shapes for flow",
    "Color Texture Harmony - Combined color and texture",
    "Depth Texture Harmony - Blends focus with texture",

    "Advanced Focus and Detail - Adds visual intricacy",
    "Focal Point - Primary area of attention",
    "Secondary Focus - Supporting details",
    "Tertiary Focus - Additional points of interest",
    "Highlight - Emphasizes bright areas",
    "Shadow - Darkened areas for depth",
    "Silhouette - Outline focus against light background",
    "Reflections - Mirror images add interest",
    "Subframing - Partial framing for added dimension",

    "Special Techniques - Artistic and expressive elements",
    "Rule Breaking Composition - Defies conventional layout",
    "Minimalist Composition - Simplifies to key elements",
    "Maximalist Composition - Rich, complex arrangement",
    "Isolated Element - Single subject focus",
    "Grouped Elements - Clustered for emphasis",
    "Distributed Elements - Spread across frame",
    "Central Element - Focused in center",
    "Peripheral Elements - Placed at frame edges",
]

TIME_OF_DAY_OPTIONS = [
"00:00 - Midnight", "01:00 - Late night", "02:00 - Late night", "03:00 - Early morning",
"04:00 - Early morning", "05:00 - Dawn begins", "06:00 - Sunrise", "07:00 - Morning starts",
"08:00 - Morning", "09:00 - Morning", "10:00 - Late morning", "11:00 - Late morning",
"12:00 - Noon", "13:00 - Early afternoon", "14:00 - Afternoon", "15:00 - Afternoon",
"16:00 - Late afternoon", "17:00 - Evening begins", "18:00 - Sunset", "19:00 - Early evening",
"20:00 - Evening", "21:00 - Night starts", "22:00 - Night", "23:00 - Late night",
"Dawn - First light", "Morning - Start of day", "Noon - Midday", "Afternoon - Daytime",
"Dusk - Twilight begins", "Evening - End of day", "Night - Darkness", "Midnight - Deep night"
]

FRAME_RATE_TECHNIQUES = [
# 🎞️ BASIC FRAME RATES
"BASIC FRAME RATES - Standard Frame Rate - 24 fps",
"BASIC FRAME RATES - Television Frame Rate - NTSC/PAL fps",
"BASIC FRAME RATES - Cinema Frame Rate - 25/30 fps",
"BASIC FRAME RATES - High Frame Rate - Above 60 fps",
"BASIC FRAME RATES - Low Frame Rate - Below 24 fps",
"BASIC FRAME RATES - Filmic Frame Rate - Cinematic fps",
"BASIC FRAME RATES - Digital Frame Rate - Flexible fps",
"BASIC FRAME RATES - Static Frame Rate - Fixed fps",
"BASIC FRAME RATES - Consistent Frame Rate - Uniform fps",
"BASIC FRAME RATES - Variable Frame Rate - Changing fps",
"BASIC FRAME RATES - Adaptive Frame Rate - Responsive fps",
"BASIC FRAME RATES - Selective Frame Rate - Chosen fps",
"BASIC FRAME RATES - Dynamic Frame Rate - Variable fps",
"BASIC FRAME RATES - Uniform Speed Frame Rate - Consistent speeds",
"BASIC FRAME RATES - Static Speed Frame Rate - Fixed speeds",

# 🏃‍♂️ SPEED MANIPULATION
"SPEED MANIPULATION - Slow Motion - Reduced speed",
"SPEED MANIPULATION - Fast Motion - Increased speed",
"SPEED MANIPULATION - Undercranking - Slow frame rate",
"SPEED MANIPULATION - Overcranking - Fast frame rate",
"SPEED MANIPULATION - Variable Speed Frame Rate - Changing speeds",
"SPEED MANIPULATION - Uniform Speed Frame Rate - Consistent speeds",

# 🕰️ TIME MANIPULATION
"TIME MANIPULATION - Time-Lapse - Accelerated time",
"TIME MANIPULATION - Hyperlapse - Time and space fast",
"TIME MANIPULATION - Instant Replay Frame Rate - High fps for replay",

# 🎬 MOTION EFFECTS
"MOTION EFFECTS - Motion Blur - Blurred motion",
"MOTION EFFECTS - Motion Tracking - Following movement",
"MOTION EFFECTS - Motion Control - Precise movement",
"MOTION EFFECTS - Go Motion - Enhanced motion blur",
"MOTION EFFECTS - Bullet Time - 360-degree freeze",

# 📷 FRAME GENERATION TECHNIQUES
"FRAME GENERATION - Interpolation - Frame generation",
"FRAME GENERATION - Frame Blending - Smooth transitions",
"FRAME GENERATION - Frame Skipping - Missing frames",

# 🎨 VISUAL EFFECTS
"VISUAL EFFECTS - Stop Motion - Frame-by-frame",
"VISUAL EFFECTS - Pixilation - Live-action animation",
"VISUAL EFFECTS - Double Exposure - Overlayed images",
"VISUAL EFFECTS - Rotoscoping - Frame tracing",
"VISUAL EFFECTS - Freeze Frame - Static frame",
"VISUAL EFFECTS - Strobing - Flashing frames",

# 🔄 SPECIALIZED TECHNIQUES
"SPECIALIZED TECHNIQUES - Long Exposure - Light trails",
"SPECIALIZED TECHNIQUES - Temporal Aliasing - Time distortion",
"SPECIALIZED TECHNIQUES - Anamorphic Format - Wide cinematic",

# 🖥️ POST-PRODUCTION & EQUIPMENT
"POST-PRODUCTION & EQUIPMENT - Frame Blending - Smooth transitions",
"POST-PRODUCTION & EQUIPMENT - Motion Control - Precise movement",
"POST-PRODUCTION & EQUIPMENT - Motion Tracking - Following movement",

# 💾 FILE FORMATS & ENCODING
"FILE FORMATS & ENCODING - CINEon - Digital film",
"FILE FORMATS & ENCODING - Log Format - Dynamic range preservation",
"FILE FORMATS & ENCODING - Raw Format - Unprocessed data",
]

RESOLUTIONS = {
"Experimental and Proto (Pre-1900s)": [
"Very Low Definition - Experimental resolution of early photographs with limited detail - PROS: First recorded images, foundational for photography - CONS: Very grainy, extremely low detail",
"Glass Plate Quality - Gelatin dry plate or wet collodion photography resolution - PROS: Higher detail than early methods - CONS: Still lacks sharpness of modern formats"
],
"1900s": [
"Glass Plate Standard - Photography done with glass plates, offering sharper details than early experiments - PROS: High-quality for its time - CONS: Not yet film-based",
"Low Definition Film - Very early cinema using 35mm and other film formats - PROS: First moving images - CONS: Grainy, limited clarity, flickering",
"Low-Res Panoramic - Panoramic images with limited sharpness, using rudimentary panoramic cameras - PROS: Wide views captured - CONS: Distortion and low clarity"
],
"1910s": [
"Standard Definition (SD) - Early black-and-white film resolution using 35mm - PROS: Clearer than previous eras, early cinematic standards - CONS: Lacks sharpness by modern standards",
"Silent Film Standard - Early silent film resolutions using basic lenses - PROS: First practical moving picture quality - CONS: Limited color depth, flickering",
"Plate Film Quality - Continued use of plate photography with increasing sharpness - PROS: High-quality photography compared to film - CONS: Difficult to use and slow"
],
"1920s": [
"Standard Definition (SD) - Improved film quality with better lighting and lenses, typical of early silent films - PROS: Sharper images than the 1910s - CONS: Limited resolution by modern standards",
"Silent Film Standard - Resolutions based on 35mm film used for silent film with more dynamic range - PROS: Better clarity - CONS: Limited compared to later color films",
"Panoramic Film Quality - Improved panoramic film capturing large wide shots but still lacking detail - PROS: Wide-angle capture - CONS: Limited sharpness and contrast"
],
"1930s": [
"Standard Definition (SD) - The norm for early sound films, with better film stock and lighting techniques - PROS: More consistent quality, early cinematic standard - CONS: Still lacks modern sharpness",
"Film Resolution - The standard for films in this era, improving clarity but still black-and-white - PROS: Reliable for motion pictures - CONS: Low compared to modern film",
"Enhanced Silent Film Quality - Film for silent movies achieved better exposure but still limited in detail - PROS: Stable footage - CONS: Limited contrast"
],
"1940s": [
"Standard Definition (SD) - Now considered standard for black-and-white and Technicolor films, using 35mm and 16mm formats - PROS: Great quality for the era, iconic classic films - CONS: No widescreen formats yet",
"Film Quality - Films now shot with improved lenses and cameras, delivering more consistency - PROS: Better detail and lighting - CONS: Limited color range",
"Early Widescreen - Experiments with widescreen ratios (CinemaScope-like) but with limited resolution - PROS: Wider shots, more immersive - CONS: Limited sharpness"
],
"1950s": [
"Standard Definition (SD) - Color films using 35mm and early widescreen processes - PROS: Better image stability - CONS: Still low-definition compared to later developments",
"CinemaScope Widescreen - First true widescreen formats used in films like *The Robe* (1953) - PROS: More immersive viewing experiences - CONS: Early widescreen had limited resolution",
"Technicolor Quality - Color film with good contrast and saturation, used for iconic films - PROS: Early rich colors - CONS: Still low-definition",
"TV Standard Definition (480i) - Early television formats delivering 480i interlaced content - PROS: Affordable TV experience for many homes - CONS: Lower quality compared to films"
],
"1960s": [
"Standard Definition (SD) - Better lenses and film stock improved standard cinema and television - PROS: More reliable for TV and film - CONS: Still limited by modern standards",
"Television 480i - The continued dominance of 480i interlaced TV, now widespread in households - PROS: Affordable home viewing - CONS: Lacks sharpness, still interlaced",
"CinemaScope and Panavision - Higher-quality widescreen formats for film - PROS: Widescreen revolution - CONS: Still limited sharpness and contrast",
"Early 720p Equivalent (Film) - Some experimental films reached early equivalents of high-definition - PROS: Better sharpness for key films - CONS: Experimental, not widely adopted"
],
"1970s": [
"Standard Definition (SD) - The quality for television and early color broadcasts - PROS: Color added for many TV shows - CONS: Still low-definition",
"Television 480i - Color television became the standard, continuing at 480i resolution - PROS: Popularized color - CONS: Still limited to SD formats",
"Panavision Quality - High-quality film resolution with rich color contrast, used in *Star Wars* and *The Godfather* - PROS: Gorgeous cinematic results - CONS: Film grain and noise still present",
"IMAX Film - Large-format IMAX film introduced, offering higher resolution in limited settings - PROS: Incredible detail in large theaters - CONS: Rare, expensive, not consumer-accessible"
],
"1980s": [
"Standard Definition (SD) - Still the dominant resolution for both television and VHS - PROS: Color TVs fully adopted - CONS: Limited detail, VHS degraded quality further",
"VHS Resolution - 240p equivalent for home video, very low by modern standards - PROS: Affordable and popularized home video - CONS: Poor quality, prone to degradation",
"LaserDisc 480p - Early attempts at sharper home video formats with LaserDisc - PROS: Better than VHS - CONS: Still sub-HD",
"IMAX Film - Widespread use in specialty theaters, offering much better clarity - PROS: Exceptional for large screens - CONS: Limited to special locations",
"Betacam SP - Early broadcast-standard video format for professional use - PROS: High-quality for TV - CONS: Still sub-HD, limited to studios"
],
"1990s": [
"480p DVD Quality - The rise of DVD brought 480p progressive scan for home viewing - PROS: Major improvement over VHS - CONS: Still sub-HD",
"720p HD (Early) - Introduced in broadcast television and early HD camcorders - PROS: High-definition TV and cinema - CONS: Limited adoption",
"1080i HD (Broadcast) - High-definition broadcasts began with 1080i interlaced - PROS: Massive improvement over 480i TV - CONS: Interlaced, not as sharp as 1080p",
"IMAX Film - Continued excellence in IMAX projection with more widespread adoption - PROS: Extremely high resolution for large screens - CONS: Not yet available for home use",
"MiniDV 480p - Standard format for home video and indie films - PROS: Affordable, portable - CONS: Limited resolution"
],
"2000s": [
"720p HD - Widely adopted for HD broadcasts, the first true HD standard for TV - PROS: Sharp for the era - CONS: Outdated compared to 1080p",
"1080p Full HD - Became the standard for home video, HDTVs, and Blu-ray - PROS: Full HD, sharp detail - CONS: Large file sizes, requires more processing",
"2K Digital Cinema - Used in professional film production, higher than 1080p - PROS: Cinema-quality sharpness - CONS: Not widely adopted for home use",
"4K (Cinema) - Used in digital cinema for ultra-high resolution - PROS: Extremely detailed, beautiful image - CONS: Expensive, large data requirements",
"IMAX Digital - Widespread adoption of digital IMAX projectors offering unprecedented sharpness - PROS: Immersive experiences - CONS: Only available in specialty theaters"
],
"2010s": [
"1080p Full HD - Still the standard for most content, especially on TV and streaming - PROS: Very sharp and detailed - CONS: Outdated by the end of the decade",
"2K Digital - Used in many cinema applications, offering better-than-HD resolution - PROS: High quality - CONS: Not as sharp as 4K",
"4K UHD - Became the new standard for high-end TVs and film production - PROS: Ultra-high-definition, excellent for detail and large screens - CONS: Requires large data storage",
"5K - Used in some professional editing and photography, offering even sharper resolution - PROS: More detail than 4K - CONS: Limited consumer adoption",
"8K UHD - Experimental for the highest-end displays and professional uses - PROS: Insanely sharp, perfect for huge displays - CONS: Limited consumer use, expensive"
],
"2020s": [
"4K UHD - The standard for most content, from streaming to cinema - PROS: Sharp, available on most devices - CONS: Large file sizes, outdated by 8K",
"8K UHD - The new cutting-edge standard for displays and high-end cinema - PROS: Extreme clarity and sharpness - CONS: Large file sizes, high hardware requirements",
"12K Cinema - Experimental format for cinema production, offering unparalleled clarity - PROS: Stunning for huge screens - CONS: Overkill for most uses, massive file sizes",
"1440p QHD - Popular for gaming and mid-range monitors, offering a good balance - PROS: Sharp and faster processing than 4K - CONS: Not 4K",
"5.3K - Specialized use in action cameras and some professional equipment - PROS: Great for high-speed recording - CONS: Niche, limited adoption"
],
"Future": [
"8K UHD - Expected to be mainstream soon, providing incredible detail for large displays - PROS: Amazing clarity - CONS: Expensive, requires large storage and processing power",
"12K+ Cinema - Future-proofing for high-end productions, offering extreme clarity - PROS: Perfect for huge cinematic displays - CONS: Not practical for consumer use",
"16K - Projected as a future consumer standard for ultra-large screens - PROS: Insane detail, perfect for huge video walls - CONS: Requires immense storage and processing",
"32K - Theoretical next-gen resolution for niche applications - PROS: Hypothetical future format - CONS: Unlikely to be practical anytime soon"
]
}

# Decades list for UI or sorting purposes
DECADES = sorted(RESOLUTIONS.keys())

# Example usage:
selected_decade = "2020s"
print(RESOLUTIONS[selected_decade])


SOUND_EFFECTS = [
# At least 20 sound effect categories
"Atmospheric Sound", "Environmental Sound", "Mechanical Sound", "Dialogue Focus",
"Natural Soundscapes", "Foley Effects", "Ambient Noise", "Electronic Sounds",
"Animal Sounds", "Urban Soundscapes", "Rural Soundscapes", "Underwater Sounds",
"Space Ambience", "Industrial Noise", "Weather Effects", "Crowd Noise",
"Silence", "Echoes", "Reverberations", "Distortions", "Synthesized Sounds",
"Organic Textures", "Acoustic Sounds", "Abstract Sounds"
]

LENSES = [
# At least 20 lens types
"Standard Lens", "Wide Angle", "Telephoto", "Macro", "Fisheye", "Prime", "Zoom", "Anamorphic",
"Tilt-Shift", "Ultra Wide Angle", "Short Telephoto",
"Long Telephoto", "Catadioptric", "Soft Focus", "Infrared Lens", "UV Lens",
"Cine Lens", "Portrait Lens", "Super Telephoto", "Pancake Lens", "Refractive Lens",
"Mirror Lens", "Perspective Control", "Fish-Eye Circular"
]

WILDLIFE_ANIMALS = [
"-",
# 🐾 MAMMALS
"MAMMALS - Lion",
"MAMMALS - Elephant",
"MAMMALS - Tiger",
"MAMMALS - Wolf",
"MAMMALS - Rhino",
"MAMMALS - Bear",
"MAMMALS - Fox",
"MAMMALS - Deer",
"MAMMALS - Leopard",
"MAMMALS - Giraffe",
"MAMMALS - Zebra",
"MAMMALS - Hippo",
"MAMMALS - Buffalo",
"MAMMALS - Hyena",
"MAMMALS - Jaguar",
"MAMMALS - Panda",
"MAMMALS - Koala",
"MAMMALS - Kangaroo",
"MAMMALS - Chimpanzee",
"MAMMALS - Gorilla",
"MAMMALS - Bison",
"MAMMALS - Antelope",
"MAMMALS - Coyote",
"MAMMALS - Ibex",
"MAMMALS - Impala",
"MAMMALS - Jackal",
"MAMMALS - Mandrill",
"MAMMALS - Okapi",
"MAMMALS - Orangutan",
"MAMMALS - Platypus",
"MAMMALS - Quokka",
"MAMMALS - Serval",
"MAMMALS - Snow Leopard",
"MAMMALS - Tasmanian Devil",
"MAMMALS - Uakari",
"MAMMALS - Wallaby",
"MAMMALS - Xerus",
"MAMMALS - Yak",
"MAMMALS - Zorilla",
"MAMMALS - Aardvark",
"MAMMALS - Binturong",
"MAMMALS - Baboon",
"MAMMALS - Badger",
"MAMMALS - Beaver",
"MAMMALS - Bighorn Sheep",
"MAMMALS - Bobcat",
"MAMMALS - Bonobo",
"MAMMALS - Buffalo",
"MAMMALS - Bull",
"MAMMALS - Caribou",
"MAMMALS - Chinchilla",
"MAMMALS - Cougar",
"MAMMALS - Dingo",
"MAMMALS - Fennec Fox",
"MAMMALS - Grizzly Bear",
"MAMMALS - Guanaco",
"MAMMALS - Gopher",
"MAMMALS - Hamster",
"MAMMALS - Hare",
"MAMMALS - Hedgehog",
"MAMMALS - Hyrax",
"MAMMALS - Jackrabbit",
"MAMMALS - Jaguarundi",
"MAMMALS - Marmoset",
"MAMMALS - Marmot",
"MAMMALS - Marten",
"MAMMALS - Mink",
"MAMMALS - Mole",
"MAMMALS - Moose",
"MAMMALS - Mosquito",
"MAMMALS - Mountain Goat",
"MAMMALS - Mountain Lion",
"MAMMALS - Mule",
"MAMMALS - Muskrat",
"MAMMALS - Pika",
"MAMMALS - Piranha",
"MAMMALS - Red Panda",
"MAMMALS - Reindeer",
"MAMMALS - Ram",
"MAMMALS - Raccoon",
"MAMMALS - Possum",
"MAMMALS - Prairie Dog",
"MAMMALS - Puma",
"MAMMALS - Python",
"MAMMALS - Ram",
"MAMMALS - Weasel",
"MAMMALS - Wolverine",
"MAMMALS - Wombat",

# 🦅 BIRDS
"BIRDS - Eagle",
"BIRDS - Falcon",
"BIRDS - Vulture",
"BIRDS - Hawk",
"BIRDS - Owl",
"BIRDS - Heron",
"BIRDS - Hummingbird",
"BIRDS - Ibis",
"BIRDS - Kestrel",
"BIRDS - Kingfisher",
"BIRDS - Kiwi",
"BIRDS - Kookaburra",
"BIRDS - Lark",
"BIRDS - Macaw",
"BIRDS - Magpie",
"BIRDS - Manta Ray",
"BIRDS - Myna",
"BIRDS - Nightingale",
"BIRDS - Parakeet",
"BIRDS - Peacock",
"BIRDS - Pelican",
"BIRDS - Penguin",
"BIRDS - Peregrine Falcon",
"BIRDS - Quetzal",
"BIRDS - Raven",
"BIRDS - Robin",
"BIRDS - Sparrow",
"BIRDS - Starling",
"BIRDS - Sun Conure",
"BIRDS - Scarlet Macaw",
"BIRDS - Blue Jay",
"BIRDS - Bluebird",
"BIRDS - Cardinal",
"BIRDS - Cockatoo",
"BIRDS - Finch",
"BIRDS - Mockingbird",
"BIRDS - Seagull",
"BIRDS - Toucan",
"BIRDS - Trumpeter Swan",
"BIRDS - Woodpecker",
"BIRDS - Zebu",
"BIRDS - Blue-and-Gold Macaw",
"BIRDS - Hyacinth Macaw",
"BIRDS - Blue Dragon Nudibranch",
"BIRDS - Painted Turtle",
"BIRDS - Cerberus",
"BIRDS - Pegasus",

# 🐍 REPTILES
"REPTILES - Shark",
"REPTILES - Crocodile",
"REPTILES - Alligator",
"REPTILES - Snake",
"REPTILES - Lizard",
"REPTILES - Turtle",
"REPTILES - Tortoise",
"REPTILES - Chameleon",
"REPTILES - Gecko",
"REPTILES - Komodo Dragon",
"REPTILES - Iguana",
"REPTILES - Anaconda",
"REPTILES - Cobra",
"REPTILES - Python",
"REPTILES - Rattlesnake",
"REPTILES - Boa Constrictor",
"REPTILES - Viper",
"REPTILES - Garter Snake",
"REPTILES - Monitor Lizard",
"REPTILES - Skink",
"REPTILES - Iguana",
"REPTILES - Alligator Snapping Turtle",
"REPTILES - Red-eared Slider",
"REPTILES - Green Iguana",
"REPTILES - Box Turtle",
"REPTILES - Softshell Turtle",
"REPTILES - Sea Turtle",
"REPTILES - Leatherback Turtle",
"REPTILES - Hawksbill Turtle",
"REPTILES - Loggerhead Turtle",
"REPTILES - Olive Ridley Turtle",
"REPTILES - Painted Turtle",
"REPTILES - Snapping Turtle",
"REPTILES - Musk Turtle",
"REPTILES - Red-bellied Turtle",
"REPTILES - Eastern Box Turtle",
"REPTILES - Hermann's Tortoise",
"REPTILES - Greek Tortoise",
"REPTILES - Sulcata Tortoise",
"REPTILES - Pancake Tortoise",
"REPTILES - Radiated Tortoise",
"REPTILES - Leopard Gecko",
"REPTILES - Bearded Dragon",
"REPTILES - Ball Python",
"REPTILES - Corn Snake",
"REPTILES - Red-eared Slider",
"REPTILES - Blue-tongued Skink",
"REPTILES - King Snake",
"REPTILES - Garter Snake",
"REPTILES - African Sideneck Turtle",
"REPTILES - Eastern Box Turtle",
"REPTILES - Russian Tortoise",
"REPTILES - Hermann's Tortoise",
"REPTILES - Greek Tortoise",
"REPTILES - Argentine Black and White Tegu",
"REPTILES - Pancake Tortoise",
"REPTILES - Radiated Tortoise",
"REPTILES - Sulcata Tortoise",
"REPTILES - Desert Tortoise",
"REPTILES - Mud Turtle",
"REPTILES - Map Turtle",
"REPTILES - Musk Turtle",
"REPTILES - Painted Turtle",
"REPTILES - Softshell Turtle",
"REPTILES - Wood Turtle",
"REPTILES - Snapping Turtle",
"REPTILES - Leatherback Sea Turtle",
"REPTILES - Loggerhead Sea Turtle",
"REPTILES - Green Sea Turtle",
"REPTILES - Hawksbill Sea Turtle",
"REPTILES - Olive Ridley Sea Turtle",

# 🐟 FISH
"FISH - Shark",
"FISH - Salmon",
"FISH - Tuna",
"FISH - Clownfish",
"FISH - Goldfish",
"FISH - Guppy",
"FISH - Betta",
"FISH - Angelfish",
"FISH - Tetra",
"FISH - Cichlid",
"FISH - Discus",
"FISH - Molly",
"FISH - Swordtail",
"FISH - Neon Tetra",
"FISH - Zebra Danio",
"FISH - Pleco",
"FISH - Gourami",
"FISH - Tiger Barb",
"FISH - Cherry Barb",
"FISH - Rasbora",
"FISH - Corydoras",
"FISH - Oscar",
"FISH - Silver Dollar",
"FISH - Cardinal Tetra",
"FISH - Discus Fish",
"FISH - Bristlenose Pleco",
"FISH - Harlequin Rasbora",
"FISH - Electric Yellow Cichlid",
"FISH - Pearl Gourami",
"FISH - Rummy Nose Tetra",
"FISH - Bolivian Ram",
"FISH - Firemouth Cichlid",
"FISH - Ram Cichlid",
"FISH - Jelly Bean Tetra",
"FISH - Leopard Danio",
"FISH - Black Molly",
"FISH - Blue Tang",
"FISH - Sailfin Molly",
"FISH - Fantail Goldfish",
"FISH - Pearlscale Goldfish",
"FISH - Rosy Red Minnow",
"FISH - Brichardi Cichlid",
"FISH - Peacock Bass",
"FISH - Convict Cichlid",
"FISH - Piranha",
"FISH - Pufferfish",
"FISH - Lionfish",
"FISH - Clownfish",
"FISH - Manta Ray",
"FISH - Mackerel",
"FISH - Barracuda",
"FISH - Anglerfish",
"FISH - Moray Eel",
"FISH - Electric Eel",
"FISH - Parrotfish",
"FISH - Boxfish",
"FISH - Triggerfish",
"FISH - Surgeonfish",
"FISH - Goblin Shark",
"FISH - Whale Shark",
"FISH - Great White Shark",
"FISH - Hammerhead Shark",
"FISH - Tiger Shark",
"FISH - Bull Shark",
"FISH - Lemon Shark",
"FISH - Nurse Shark",
"FISH - Thresher Shark",
"FISH - Blacktip Shark",
"FISH - Blue Shark",
"FISH - Sand Tiger Shark",
"FISH - Leopard Shark",
"FISH - Basking Shark",
"FISH - Whale Ray",
"FISH - Sturgeon",
"FISH - Catfish",
"FISH - Eel",
"FISH - Barramundi",
"FISH - Cobia",
"FISH - Mahi-Mahi",
"FISH - Dorado",
"FISH - Snapper",
"FISH - Grouper",
"FISH - Tarpon",
"FISH - Sailfish",
"FISH - Marlin",
"FISH - Dorado",
"FISH - Wahoo",
"FISH - Blue Marlin",
"FISH - White Marlin",

# 🐜 INSECTS
"INSECTS - Butterfly",
"INSECTS - Ladybug",
"INSECTS - Dragonfly",
"INSECTS - Praying Mantis",
"INSECTS - Ant",
"INSECTS - Beetle",
"INSECTS - Grasshopper",
"INSECTS - Cockroach",
"INSECTS - Termite",
"INSECTS - Firefly",
"INSECTS - Moth",
"INSECTS - Wasp",
"INSECTS - Flea",
"INSECTS - Tick",
"INSECTS - Lacewing",
"INSECTS - Earwig",
"INSECTS - Silverfish",
"INSECTS - Aphid",
"INSECTS - Cicada",
"INSECTS - Katydid",
"INSECTS - Mealybug",
"INSECTS - Thrips",
"INSECTS - Stink Bug",
"INSECTS - Weevil",
"INSECTS - Whitefly",
"INSECTS - Scorpion",
"INSECTS - Locust",
"INSECTS - Mantid",
"INSECTS - Sawfly",
"INSECTS - Mayfly",
"INSECTS - Stonefly",
"INSECTS - Borer Beetle",
"INSECTS - Assassin Bug",
"INSECTS - Leafhopper",
"INSECTS - Longhorn Beetle",
"INSECTS - Lace Bug",
"INSECTS - Boxelder Bug",
"INSECTS - Slug",
"INSECTS - Snail",
"INSECTS - Spider",
"INSECTS - Scorpion",
"INSECTS - Bumblebee",
"INSECTS - Hornet",
"INSECTS - Dragonfly",
"INSECTS - Damselfly",
"INSECTS - Cicada",
"INSECTS - Fire Ant",
"INSECTS - June Bug",
"INSECTS - Monarch Butterfly",
"INSECTS - Atlas Moth",
"INSECTS - Emperor Moth",
"INSECTS - Luna Moth",
"INSECTS - Goliath Beetle",
"INSECTS - Hercules Beetle",
"INSECTS - Atlas Beetle",
"INSECTS - Rhinoceros Beetle",
"INSECTS - Stick Insect",
"INSECTS - Walking Stick",
"INSECTS - Katydid",
"INSECTS - Lacewing",
"INSECTS - Firefly",
"INSECTS - Lace Bug",
"INSECTS - Ladybird",
"INSECTS - Paper Wasp",
"INSECTS - Yellow Jacket",
"INSECTS - Asian Giant Hornet",

# 🕷️ ARACHNIDS
"ARACHNIDS - Spider",
"ARACHNIDS - Scorpion",
"ARACHNIDS - Tick",
"ARACHNIDS - Mite",
"ARACHNIDS - Harvestman",
"ARACHNIDS - Daddy Longlegs",
"ARACHNIDS - Vinegaroon",
"ARACHNIDS - Whip Scorpion",
"ARACHNIDS - Pseudoscorpion",
"ARACHNIDS - Sun Spider",
"ARACHNIDS - Trapdoor Spider",
"ARACHNIDS - Jumping Spider",
"ARACHNIDS - Orb Weaver",
"ARACHNIDS - Funnel-web Spider",
"ARACHNIDS - Wolf Spider",
"ARACHNIDS - Brown Recluse",
"ARACHNIDS - Black Widow",
"ARACHNIDS - Tarantula",
"ARACHNIDS - Huntsman Spider",
"ARACHNIDS - Recluse Spider",
"ARACHNIDS - Cellar Spider",
"ARACHNIDS - Comb-footed Spider",
"ARACHNIDS - Crab Spider",
"ARACHNIDS - Sac Spider",
"ARACHNIDS - Long-bodied Cellar Spider",
"ARACHNIDS - Water Spider",
"ARACHNIDS - Trapdoor Spider",
"ARACHNIDS - Wolf Spider",
"ARACHNIDS - Nursery Web Spider",
"ARACHNIDS - Lynx Spider",
"ARACHNIDS - Black Lace Weaver",
"ARACHNIDS - Golden Orb Weaver",
"ARACHNIDS - Hobo Spider",
"ARACHNIDS - Gray Wall Spider",
"ARACHNIDS - Brazilian Wandering Spider",
"ARACHNIDS - Brown Widow Spider",
"ARACHNIDS - Redback Spider",
"ARACHNIDS - Desert Recluse",
"ARACHNIDS - Black House Spider",
"ARACHNIDS - Brown House Spider",
"ARACHNIDS - Baltic Spiny Orb Weaver",
"ARACHNIDS - Black Lace Web Spider",
"ARACHNIDS - Sphodros Spider",

# 🦀 CRUSTACEANS
"CRUSTACEANS - Lobster",
"CRUSTACEANS - Crab",
"CRUSTACEANS - Shrimp",
"CRUSTACEANS - Krill",
"CRUSTACEANS - Barnacle",
"CRUSTACEANS - Krab",
"CRUSTACEANS - Ghost Shrimp",
"CRUSTACEANS - Fiddler Crab",
"CRUSTACEANS - Hermit Crab",
"CRUSTACEANS - Pistol Shrimp",
"CRUSTACEANS - Rock Lobster",
"CRUSTACEANS - Snow Crab",
"CRUSTACEANS - Blue Crab",
"CRUSTACEANS - Red King Crab",
"CRUSTACEANS - Coconut Crab",
"CRUSTACEANS - Horseshoe Crab",
"CRUSTACEANS - Signal Crayfish",
"CRUSTACEANS - Dungeness Crab",
"CRUSTACEANS - Velvet Crab",
"CRUSTACEANS - Stone Crab",
"CRUSTACEANS - Red Swamp Crayfish",
"CRUSTACEANS - American Lobster",
"CRUSTACEANS - Japanese Spider Crab",
"CRUSTACEANS - European Green Crab",
"CRUSTACEANS - Chinese Mitten Crab",
"CRUSTACEANS - Ghost Shrimp",
"CRUSTACEANS - Mantis Shrimp",
"CRUSTACEANS - Tiger Prawn",
"CRUSTACEANS - Macrobrachium Prawn",
"CRUSTACEANS - Blue King Crab",
"CRUSTACEANS - Black King Crab",
"CRUSTACEANS - Tiger Crab",
"CRUSTACEANS - Long-legged Crab",
"CRUSTACEANS - Velvet Prawn",
"CRUSTACEANS - Blue Crab",
"CRUSTACEANS - Water Flea",
"CRUSTACEANS - Copepod",
"CRUSTACEANS - Amphipod",
"CRUSTACEANS - Isopod",
"CRUSTACEANS - Mysid Shrimp",
"CRUSTACEANS - Fairy Shrimp",
"CRUSTACEANS - Nautilus",
"CRUSTACEANS - Crayfish",
"CRUSTACEANS - Ghost Shrimp",
"CRUSTACEANS - Pistol Shrimp",
"CRUSTACEANS - Boxer Shrimp",
"CRUSTACEANS - Cleaner Shrimp",
"CRUSTACEANS - Peppermint Shrimp",
"CRUSTACEANS - Mantis Shrimp",
"CRUSTACEANS - Snapping Shrimp",
"CRUSTACEANS - Reef Crab",
"CRUSTACEANS - Brine Shrimp",
"CRUSTACEANS - Red Rock Crab",
"CRUSTACEANS - Marine Lobster",
"CRUSTACEANS - Slipper Lobster",
"CRUSTACEANS - Galathea Lobster",

# 🐸 AMPHIBIANS
"AMPHIBIANS - Frog",
"AMPHIBIANS - Toad",
"AMPHIBIANS - Salamander",
"AMPHIBIANS - Newt",
"AMPHIBIANS - Axolotl",
"AMPHIBIANS - Caecilian",
"AMPHIBIANS - Tree Frog",
"AMPHIBIANS - Poison Dart Frog",
"AMPHIBIANS - Bullfrog",
"AMPHIBIANS - Tiger Salamander",
"AMPHIBIANS - Fire Salamander",
"AMPHIBIANS - Green Tree Frog",
"AMPHIBIANS - American Toad",
"AMPHIBIANS - Red-eyed Tree Frog",
"AMPHIBIANS - Spotted Salamander",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Spring Peeper",
"AMPHIBIANS - Wood Frog",
"AMPHIBIANS - Pickerel Frog",
"AMPHIBIANS - Pacific Tree Frog",
"AMPHIBIANS - Woodhouse's Toad",
"AMPHIBIANS - Blue Poison Dart Frog",
"AMPHIBIANS - Hellbender",
"AMPHIBIANS - Green Salamander",
"AMPHIBIANS - Fire-bellied Toad",
"AMPHIBIANS - Eastern Newt",
"AMPHIBIANS - Boreal Chorus Frog",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Spotted Salamander",
"AMPHIBIANS - Pickerel Frog",
"AMPHIBIANS - Pacific Tree Frog",
"AMPHIBIANS - Red-legged Frog",
"AMPHIBIANS - Green Frog",
"AMPHIBIANS - American Bullfrog",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Northern Leopard Frog",
"AMPHIBIANS - Plains Leopard Frog",
"AMPHIBIANS - Tiger Frog",
"AMPHIBIANS - Green Frog",
"AMPHIBIANS - Wood Frog",
"AMPHIBIANS - Green Tree Frog",
"AMPHIBIANS - Blue Poison Dart Frog",
"AMPHIBIANS - Red-eyed Tree Frog",
"AMPHIBIANS - Spring Peeper",
"AMPHIBIANS - Woodhouse's Toad",
"AMPHIBIANS - American Toad",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Bullfrog",
"AMPHIBIANS - Eastern Toad",
"AMPHIBIANS - Green Frog",
"AMPHIBIANS - Northern Leopard Frog",
"AMPHIBIANS - Plains Leopard Frog",
"AMPHIBIANS - Pickerel Frog",
"AMPHIBIANS - Pacific Tree Frog",
"AMPHIBIANS - Red-legged Frog",
"AMPHIBIANS - Spotted Salamander",
"AMPHIBIANS - Spring Peeper",
"AMPHIBIANS - Tiger Salamander",
"AMPHIBIANS - Tree Frog",
"AMPHIBIANS - Toad",
"AMPHIBIANS - Water Frog",

# 🌊 SEA ANIMALS
"SEA ANIMALS - Shark",
"SEA ANIMALS - Whale",
"SEA ANIMALS - Dolphin",
"SEA ANIMALS - Manta Ray",
"SEA ANIMALS - Octopus",
"SEA ANIMALS - Shrimp",
"SEA ANIMALS - Lobster",
"SEA ANIMALS - Crab",
"SEA ANIMALS - Starfish",
"SEA ANIMALS - Sea Turtle",
"SEA ANIMALS - Jellyfish",
"SEA ANIMALS - Coral",
"SEA ANIMALS - Eel",
"SEA ANIMALS - Seal",
"SEA ANIMALS - Walrus",
"SEA ANIMALS - Manatee",
"SEA ANIMALS - Beluga Whale",
"SEA ANIMALS - Orca",
"SEA ANIMALS - Blue Whale",
"SEA ANIMALS - Humpback Whale",
"SEA ANIMALS - Gray Whale",
"SEA ANIMALS - Killer Whale",
"SEA ANIMALS - Bottlenose Dolphin",
"SEA ANIMALS - Pilot Whale",
"SEA ANIMALS - Dugong",
"SEA ANIMALS - Sea Lion",
"SEA ANIMALS - Harbor Seal",
"SEA ANIMALS - Green Sea Turtle",
"SEA ANIMALS - Loggerhead Turtle",
"SEA ANIMALS - Leatherback Turtle",
"SEA ANIMALS - Kemp's Ridley Turtle",
"SEA ANIMALS - Hawksbill Turtle",
"SEA ANIMALS - Spinner Dolphin",
"SEA ANIMALS - Risso's Dolphin",
"SEA ANIMALS - Atlantic Puffin",
"SEA ANIMALS - Common Dolphin",
"SEA ANIMALS - Basking Shark",
"SEA ANIMALS - Great White Shark",
"SEA ANIMALS - Whale Shark",
"SEA ANIMALS - Blue Marlin",
"SEA ANIMALS - Swordfish",
"SEA ANIMALS - Mako Shark",
"SEA ANIMALS - Tiger Shark",
"SEA ANIMALS - Thresher Shark",
"SEA ANIMALS - Nurse Shark",
"SEA ANIMALS - Hammerhead Shark",
"SEA ANIMALS - Blacktip Shark",
"SEA ANIMALS - Whale Ray",
"SEA ANIMALS - Moray Eel",
"SEA ANIMALS - Lionfish",
"SEA ANIMALS - Angelfish",
"SEA ANIMALS - Lion's Mane Jellyfish",
"SEA ANIMALS - Moon Jellyfish",
"SEA ANIMALS - Sea Urchin",
"SEA ANIMALS - Blue-ringed Octopus",
"SEA ANIMALS - Giant Pacific Octopus",
"SEA ANIMALS - Box Jellyfish",
"SEA ANIMALS - Portuguese Man o' War",
"SEA ANIMALS - Cuttlefish",
"SEA ANIMALS - Sea Slug",
"SEA ANIMALS - Giant Clam",
"SEA ANIMALS - Cone Snail",
"SEA ANIMALS - Sea Cucumber",
"SEA ANIMALS - Nudibranch",
"SEA ANIMALS - Horseshoe Crab",
"SEA ANIMALS - Electric Eel",
"SEA ANIMALS - Pufferfish",
"SEA ANIMALS - Crown-of-Thorns Starfish",
"SEA ANIMALS - Blue Sea Dragon",
"SEA ANIMALS - Mantis Shrimp",
"SEA ANIMALS - Sea Spider",
"SEA ANIMALS - Blue Dragon Nudibranch",
"SEA ANIMALS - Sea Anemone",
"SEA ANIMALS - Sea Horse",
"SEA ANIMALS - Frilled Shark",
"SEA ANIMALS - Goblin Shark",
"SEA ANIMALS - Sea Lamprey",
"SEA ANIMALS - Sperm Whale",
"SEA ANIMALS - Bowhead Whale",
"SEA ANIMALS - Fin Whale",
"SEA ANIMALS - Sei Whale",
"SEA ANIMALS - Bryde's Whale",
"SEA ANIMALS - Cuvier's Beaked Whale",
"SEA ANIMALS - Pygmy Sperm Whale",
"SEA ANIMALS - Short-finned Pilot Whale",
"SEA ANIMALS - False Killer Whale",
"SEA ANIMALS - Striped Dolphin",
"SEA ANIMALS - Rough-toothed Dolphin",
"SEA ANIMALS - Indo-Pacific Humpback Dolphin",
"SEA ANIMALS - Atlantic Humpback Dolphin",
"SEA ANIMALS - Amazon River Dolphin",
"SEA ANIMALS - Irrawaddy Dolphin",
"SEA ANIMALS - Vaquita",
"SEA ANIMALS - Dall's Porpoise",
"SEA ANIMALS - Harbor Porpoise",
"SEA ANIMALS - Finless Porpoise",
"SEA ANIMALS - Burmeister's Porpoise",
"SEA ANIMALS - Spectacled Porpoise",
"SEA ANIMALS - Chinese White Dolphin",
"SEA ANIMALS - Hector's Dolphin",
"SEA ANIMALS - Maui's Dolphin",
"SEA ANIMALS - Commerson's Dolphin",
"SEA ANIMALS - Spotted Dolphin",
"SEA ANIMALS - Hourglass Dolphin",
"SEA ANIMALS - Black Dolphin",
"SEA ANIMALS - Australian Snubfin Dolphin",
"SEA ANIMALS - Fraser's Dolphin",
"SEA ANIMALS - Atlantic Bottlenose Dolphin",
"SEA ANIMALS - Pacific Bottlenose Dolphin",

# 🦀 CRUSTACEANS
"CRUSTACEANS - Lobster",
"CRUSTACEANS - Crab",
"CRUSTACEANS - Shrimp",
"CRUSTACEANS - Krill",
"CRUSTACEANS - Barnacle",
"CRUSTACEANS - Ghost Shrimp",
"CRUSTACEANS - Fiddler Crab",
"CRUSTACEANS - Hermit Crab",
"CRUSTACEANS - Pistol Shrimp",
"CRUSTACEANS - Rock Lobster",
"CRUSTACEANS - Snow Crab",
"CRUSTACEANS - Blue Crab",
"CRUSTACEANS - Red King Crab",
"CRUSTACEANS - Coconut Crab",
"CRUSTACEANS - Horseshoe Crab",
"CRUSTACEANS - Signal Crayfish",
"CRUSTACEANS - Dungeness Crab",
"CRUSTACEANS - Velvet Crab",
"CRUSTACEANS - Stone Crab",
"CRUSTACEANS - Red Swamp Crayfish",
"CRUSTACEANS - American Lobster",
"CRUSTACEANS - Japanese Spider Crab",
"CRUSTACEANS - European Green Crab",
"CRUSTACEANS - Chinese Mitten Crab",
"CRUSTACEANS - Mantis Shrimp",
"CRUSTACEANS - Tiger Prawn",
"CRUSTACEANS - Macrobrachium Prawn",
"CRUSTACEANS - Blue King Crab",
"CRUSTACEANS - Black King Crab",
"CRUSTACEANS - Tiger Crab",
"CRUSTACEANS - Long-legged Crab",
"CRUSTACEANS - Velvet Prawn",
"CRUSTACEANS - Blue Crab",
"CRUSTACEANS - Water Flea",
"CRUSTACEANS - Copepod",
"CRUSTACEANS - Amphipod",
"CRUSTACEANS - Isopod",
"CRUSTACEANS - Mysid Shrimp",
"CRUSTACEANS - Fairy Shrimp",
"CRUSTACEANS - Nautilus",
"CRUSTACEANS - Crayfish",
"CRUSTACEANS - Pistol Shrimp",
"CRUSTACEANS - Boxer Shrimp",
"CRUSTACEANS - Cleaner Shrimp",
"CRUSTACEANS - Peppermint Shrimp",
"CRUSTACEANS - Mantis Shrimp",
"CRUSTACEANS - Snapping Shrimp",
"CRUSTACEANS - Reef Crab",
"CRUSTACEANS - Brine Shrimp",
"CRUSTACEANS - Red Rock Crab",
"CRUSTACEANS - Giant Clam",
"CRUSTACEANS - Cone Snail",
"CRUSTACEANS - Sea Cucumber",
"CRUSTACEANS - Nudibranch",
"CRUSTACEANS - Horseshoe Crab",
"CRUSTACEANS - Electric Eel",
"CRUSTACEANS - Pufferfish",
"CRUSTACEANS - Crown-of-Thorns Starfish",
"CRUSTACEANS - Blue Sea Dragon",
"CRUSTACEANS - Mantis Shrimp",
"CRUSTACEANS - Sea Spider",
"CRUSTACEANS - Blue Dragon Nudibranch",
"CRUSTACEANS - Sea Anemone",
"CRUSTACEANS - Sea Horse",
"CRUSTACEANS - Frilled Shark",
"CRUSTACEANS - Goblin Shark",
"CRUSTACEANS - Sea Lamprey",
"CRUSTACEANS - Sperm Whale",
"CRUSTACEANS - Bowhead Whale",
"CRUSTACEANS - Fin Whale",
"CRUSTACEANS - Sei Whale",
"CRUSTACEANS - Bryde's Whale",
"CRUSTACEANS - Cuvier's Beaked Whale",
"CRUSTACEANS - Pygmy Sperm Whale",
"CRUSTACEANS - Short-finned Pilot Whale",
"CRUSTACEANS - False Killer Whale",
"CRUSTACEANS - Striped Dolphin",
"CRUSTACEANS - Rough-toothed Dolphin",
"CRUSTACEANS - Indo-Pacific Humpback Dolphin",
"CRUSTACEANS - Atlantic Humpback Dolphin",
"CRUSTACEANS - Amazon River Dolphin",
"CRUSTACEANS - Irrawaddy Dolphin",
"CRUSTACEANS - Vaquita",
"CRUSTACEANS - Dall's Porpoise",
"CRUSTACEANS - Harbor Porpoise",
"CRUSTACEANS - Finless Porpoise",
"CRUSTACEANS - Burmeister's Porpoise",
"CRUSTACEANS - Spectacled Porpoise",
"CRUSTACEANS - Chinese White Dolphin",
"CRUSTACEANS - Hector's Dolphin",
"CRUSTACEANS - Maui's Dolphin",
"CRUSTACEANS - Commerson's Dolphin",
"CRUSTACEANS - Spotted Dolphin",
"CRUSTACEANS - Hourglass Dolphin",
"CRUSTACEANS - Black Dolphin",
"CRUSTACEANS - Australian Snubfin Dolphin",
"CRUSTACEANS - Fraser's Dolphin",
"CRUSTACEANS - Atlantic Bottlenose Dolphin",
"CRUSTACEANS - Pacific Bottlenose Dolphin",

# 🐜 INSECTS
"INSECTS - Butterfly",
"INSECTS - Ladybug",
"INSECTS - Dragonfly",
"INSECTS - Praying Mantis",
"INSECTS - Ant",
"INSECTS - Beetle",
"INSECTS - Grasshopper",
"INSECTS - Cockroach",
"INSECTS - Termite",
"INSECTS - Firefly",
"INSECTS - Moth",
"INSECTS - Wasp",
"INSECTS - Flea",
"INSECTS - Tick",
"INSECTS - Lacewing",
"INSECTS - Earwig",
"INSECTS - Silverfish",
"INSECTS - Aphid",
"INSECTS - Cicada",
"INSECTS - Katydid",
"INSECTS - Mealybug",
"INSECTS - Thrips",
"INSECTS - Stink Bug",
"INSECTS - Weevil",
"INSECTS - Whitefly",
"INSECTS - Scorpion",
"INSECTS - Locust",
"INSECTS - Mantid",
"INSECTS - Sawfly",
"INSECTS - Mayfly",
"INSECTS - Stonefly",
"INSECTS - Borer Beetle",
"INSECTS - Assassin Bug",
"INSECTS - Leafhopper",
"INSECTS - Longhorn Beetle",
"INSECTS - Lace Bug",
"INSECTS - Boxelder Bug",
"INSECTS - Slug",
"INSECTS - Snail",
"INSECTS - Spider",
"INSECTS - Scorpion",
"INSECTS - Bumblebee",
"INSECTS - Hornet",
"INSECTS - Damselfly",
"INSECTS - Fire Ant",
"INSECTS - June Bug",
"INSECTS - Monarch Butterfly",
"INSECTS - Atlas Moth",
"INSECTS - Emperor Moth",
"INSECTS - Luna Moth",
"INSECTS - Goliath Beetle",
"INSECTS - Hercules Beetle",
"INSECTS - Atlas Beetle",
"INSECTS - Rhinoceros Beetle",
"INSECTS - Stick Insect",
"INSECTS - Walking Stick",
"INSECTS - Katydid",
"INSECTS - Lacewing",
"INSECTS - Firefly",
"INSECTS - Lace Bug",
"INSECTS - Ladybird",
"INSECTS - Paper Wasp",
"INSECTS - Yellow Jacket",
"INSECTS - Asian Giant Hornet",

# 🐸 AMPHIBIANS
"AMPHIBIANS - Frog",
"AMPHIBIANS - Toad",
"AMPHIBIANS - Salamander",
"AMPHIBIANS - Newt",
"AMPHIBIANS - Axolotl",
"AMPHIBIANS - Caecilian",
"AMPHIBIANS - Tree Frog",
"AMPHIBIANS - Poison Dart Frog",
"AMPHIBIANS - Bullfrog",
"AMPHIBIANS - Tiger Salamander",
"AMPHIBIANS - Fire Salamander",
"AMPHIBIANS - Green Tree Frog",
"AMPHIBIANS - American Toad",
"AMPHIBIANS - Red-eyed Tree Frog",
"AMPHIBIANS - Spotted Salamander",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Spring Peeper",
"AMPHIBIANS - Wood Frog",
"AMPHIBIANS - Pickerel Frog",
"AMPHIBIANS - Pacific Tree Frog",
"AMPHIBIANS - Red-legged Frog",
"AMPHIBIANS - Green Frog",
"AMPHIBIANS - American Bullfrog",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Northern Leopard Frog",
"AMPHIBIANS - Plains Leopard Frog",
"AMPHIBIANS - Troodon",
"AMPHIBIANS - Troodon",
"AMPHIBIANS - Masked Frog",
"AMPHIBIANS - Mountain Frog",
"AMPHIBIANS - Clawed Frog",
"AMPHIBIANS - Orange-Spotted Frog",
"AMPHIBIANS - Tree Frog",
"AMPHIBIANS - Whistling Frog",
"AMPHIBIANS - Leopard Frog",
"AMPHIBIANS - Green Tree Frog",
"AMPHIBIANS - Amazon Milk Frog",
"AMPHIBIANS - Blue Poison Dart Frog",
"AMPHIBIANS - White's Tree Frog",
"AMPHIBIANS - Fire-bellied Toad",
"AMPHIBIANS - Eastern Newt",
"AMPHIBIANS - Boreal Chorus Frog",
"AMPHIBIANS - Gray Tree Frog",
"AMPHIBIANS - Pickerel Frog",
"AMPHIBIANS - Pacific Tree Frog",
"AMPHIBIANS - Red-legged Frog",
"AMPHIBIANS - Spotted Salamander",
"AMPHIBIANS - Spring Peeper",
"AMPHIBIANS - Tiger Salamander",
"AMPHIBIANS - Tree Frog",
"AMPHIBIANS - Toad",
"AMPHIBIANS - Water Frog",

# 🌿 PLANTS
"PLANTS - Venus Flytrap",
"PLANTS - Rose",
"PLANTS - Sunflower",
"PLANTS - Tulip",
"PLANTS - Orchid",
"PLANTS - Cactus",
"PLANTS - Aloe Vera",
"PLANTS - Bamboo",
"PLANTS - Lavender",
"PLANTS - Daffodil",
"PLANTS - Marigold",
"PLANTS - Peony",
"PLANTS - Fern",
"PLANTS - Ivy",
"PLANTS - Bonsai",
"PLANTS - Bamboo Orchid",
"PLANTS - Jasmine",
"PLANTS - Hydrangea",
"PLANTS - Lilac",
"PLANTS - Magnolia",
"PLANTS - Pothos",
"PLANTS - Snake Plant",
"PLANTS - Spider Plant",
"PLANTS - Philodendron",
"PLANTS - ZZ Plant",
"PLANTS - Ficus",
"PLANTS - Monstera",
"PLANTS - Peace Lily",
"PLANTS - Begonia",
"PLANTS - Pilea",
"PLANTS - Ceropegia",
"PLANTS - African Violet",
"PLANTS - English Ivy",
"PLANTS - Calathea",
"PLANTS - Schefflera",
"PLANTS - Dracaena",
"PLANTS - Croton",
"PLANTS - Asparagus Fern",
"PLANTS - Kalanchoe",
"PLANTS - Bromeliad",
"PLANTS - Hoya",
"PLANTS - Echeveria",
"PLANTS - Christmas Cactus",
"PLANTS - Kokedama",
"PLANTS - Air Plant",
"PLANTS - String of Pearls",
"PLANTS - Variegated Rubber Plant",
"PLANTS - Peperomia",
"PLANTS - Tradescantia",
"PLANTS - Schefflera Arboricola",
"PLANTS - Chinese Evergreen",
"PLANTS - Pilea Peperomioides",
"PLANTS - Anthurium",
"PLANTS - Prayer Plant",
"PLANTS - Pothos",
"PLANTS - Dracaena",
"PLANTS - Croton",
"PLANTS - Asparagus Fern",
"PLANTS - Kalanchoe",
"PLANTS - Bromeliad",
"PLANTS - Hoya",
"PLANTS - Echeveria",
"PLANTS - Christmas Cactus",
"PLANTS - Kokedama",
"PLANTS - Air Plant",
"PLANTS - String of Pearls",
"PLANTS - Variegated Rubber Plant",
"PLANTS - Peperomia",
"PLANTS - Tradescantia",
"PLANTS - Schefflera Arboricola",
"PLANTS - Chinese Evergreen",
"PLANTS - Pilea Peperomioides",
"PLANTS - Anthurium",
"PLANTS - Prayer Plant",
"PLANTS - Philodendron",
"PLANTS - ZZ Plant",
]

DOMESTICATED_ANIMALS = [
"-",
# 🐶 DOGS
"DOGS - Dog",
"DOGS - Puppy",
"DOGS - Sheltie",
"DOGS - Maltese",
"DOGS - Yorkshire Terrier",
"DOGS - Cane Corso",
"DOGS - Dachshund",
"DOGS - Irish Setter",
"DOGS - Jack Russell Terrier",
"DOGS - Komondor",
"DOGS - Labrador Retriever",
"DOGS - Maltese Dog",
"DOGS - Norfolk Terrier",
"DOGS - Old English Sheepdog",
"DOGS - Papillon",
"DOGS - Quechua Sheepdog",
"DOGS - Samoyed",
"DOGS - Tibetan Mastiff",
"DOGS - Utonagan",
"DOGS - Vizsla",
"DOGS - Whippet",
"DOGS - Xoloitzcuintli",
"DOGS - Zonker",
"DOGS - Airedale Terrier",
"DOGS - Border Collie",
"DOGS - Chihuahua",
"DOGS - Doberman Pinscher",
"DOGS - English Bulldog",
"DOGS - French Bulldog",
"DOGS - Great Dane",
"DOGS - Havanese",
"DOGS - Irish Wolfhound",
"DOGS - King Charles Spaniel",
"DOGS - Lhasa Apso",
"DOGS - Miniature Schnauzer",
"DOGS - Newfoundland",
"DOGS - Pekingese",
"DOGS - Queensland Heeler",
"DOGS - Rottweiler",
"DOGS - Shiba Inu",
"DOGS - Toy Poodle",
"DOGS - Australian Shepherd",
"DOGS - Bichon Frise",
"DOGS - Cocker Spaniel",
"DOGS - Dandie Dinmont Terrier",
"DOGS - English Setter",
"DOGS - Flat-Coated Retriever",
"DOGS - Goldendoodle",
"DOGS - Harrier",
"DOGS - Italian Greyhound",
"DOGS - Japanese Chin",
"DOGS - Keeshond",
"DOGS - Leonberger",
"DOGS - Miniature Bull Terrier",
"DOGS - Norwich Terrier",
"DOGS - Pomeranian",
"DOGS - Queen Elizabeth Pocket Beagle",
"DOGS - Redbone Coonhound",
"DOGS - Silky Terrier",
"DOGS - Volpino Italiano",
"DOGS - Wirehaired Pointing Griffon",
"DOGS - Zuchon",
"DOGS - Anatolian Shepherd",
"DOGS - Basset Hound",
"DOGS - Cavalier King Charles Spaniel",
"DOGS - Dalmatian",
"DOGS - English Cocker Spaniel",
"DOGS - Finnish Spitz",
"DOGS - Glen of Imaal Terrier",
"DOGS - Icelandic Sheepdog",
"DOGS - Japanese Spitz",
"DOGS - Norwegian Elkhound",
"DOGS - Queen Elizabeth II Cavalier",
"DOGS - Rat Terrier",
"DOGS - Shetland Sheepdog",
"DOGS - Tibetan Spaniel",
"DOGS - United States Shepherd",
"DOGS - Welsh Corgi",
"DOGS - American Eskimo Dog",
"DOGS - Belgian Malinois",
"DOGS - Chinese Crested",
"DOGS - Dutch Shepherd",
"DOGS - English Springer Spaniel",
"DOGS - German Shepherd",
"DOGS - Italian Spinone",
"DOGS - Japanese Terrier",
"DOGS - Kerry Blue Terrier",
"DOGS - Lowchen",
"DOGS - Miniature Pinscher",
"DOGS - Olde English Bulldogge",
"DOGS - Scottish Terrier",
"DOGS - Tibetan Terrier",
"DOGS - Ugandan Kob",
"DOGS - Wirehaired Vizsla",
"DOGS - Australian Cattle Dog",
"DOGS - Beagle",
"DOGS - Cairn Terrier",
"DOGS - Golden Retriever",
"DOGS - Husky",
"DOGS - Poodle",
"DOGS - Shih Tzu",
"DOGS - Afghan Hound",
"DOGS - Border Terrier",
"DOGS - Fox Terrier",
"DOGS - German Shorthaired Pointer",
"DOGS - Irish Setter",

# 🐱 CATS
"CATS - Domestic Shorthair",
"CATS - Domestic Longhair",
"CATS - Siamese",
"CATS - Persian",
"CATS - Maine Coon",
"CATS - Ragdoll",
"CATS - Bengal",
"CATS - Sphynx",
"CATS - British Shorthair",
"CATS - Abyssinian",
"CATS - Scottish Fold",
"CATS - Russian Blue",
"CATS - Burmese",
"CATS - Devon Rex",
"CATS - Norwegian Forest Cat",
"CATS - Oriental Shorthair",
"CATS - Savannah",
"CATS - Birman",
"CATS - Tonkinese",
"CATS - American Shorthair",
"CATS - Turkish Angora",
"CATS - Balinese",
"CATS - Exotic Shorthair",
"CATS - Singapura",
"CATS - Havana Brown",
"CATS - LaPerm",
"CATS - Egyptian Mau",
"CATS - Chartreux",
"CATS - Ragamuffin",
"CATS - Manx",
"CATS - Turkish Van",
"CATS - Korat",
"CATS - Peterbald",
"CATS - Ocicat",
"CATS - Cymric",
"CATS - Japanese Bobtail",
"CATS - Burmilla",
"CATS - Munchkin",
"CATS - Pixiebob",
"CATS - Selkirk Rex",
"CATS - American Wirehair",

# 🐴 HORSES & EQUINES
"HORSES & EQUINES - Horse",
"HORSES & EQUINES - Pony",
"HORSES & EQUINES - Gelding",
"HORSES & EQUINES - Tennessee Walker",
"HORSES & EQUINES - Arabian Horse",
"HORSES & EQUINES - Thoroughbred",
"HORSES & EQUINES - Quarter Horse",
"HORSES & EQUINES - Morgan Horse",
"HORSES & EQUINES - Appaloosa",
"HORSES & EQUINES - Clydesdale",
"HORSES & EQUINES - Andalusian",
"HORSES & EQUINES - Belgian Horse",
"HORSES & EQUINES - Friesian Horse",
"HORSES & EQUINES - Haflinger",
"HORSES & EQUINES - Shetland Pony",
"HORSES & EQUINES - Connemara Pony",
"HORSES & EQUINES - Icelandic Horse",
"HORSES & EQUINES - Mustang",
"HORSES & EQUINES - Miniature Horse",
"HORSES & EQUINES - Welsh Pony",
"HORSES & EQUINES - Lipizzaner",
"HORSES & EQUINES - Suffolk Punch",
"HORSES & EQUINES - Hackney Horse",
"HORSES & EQUINES - Hanoverian",
"HORSES & EQUINES - Paint Horse",
"HORSES & EQUINES - Percheron",
"HORSES & EQUINES - Shire Horse",
"HORSES & EQUINES - Trakehner",
"HORSES & EQUINES - Trotter",
"HORSES & EQUINES - Paso Fino",
"HORSES & EQUINES - Tennessee Walking Horse",
"HORSES & EQUINES - Gypsy Vanner",
"HORSES & EQUINES - American Saddlebred",
"HORSES & EQUINES - Breton",
"HORSES & EQUINES - Falabella",
"HORSES & EQUINES - Marwari Horse",
"HORSES & EQUINES - Narragansett Pacer",

# 🐄 LIVESTOCK
"LIVESTOCK - Cow",
"LIVESTOCK - Chicken",
"LIVESTOCK - Sheep",
"LIVESTOCK - Pig",
"LIVESTOCK - Goat",
"LIVESTOCK - Donkey",
"LIVESTOCK - Llama",
"LIVESTOCK - Alpaca",
"LIVESTOCK - Zebu",
"LIVESTOCK - Belgian Blue",
"LIVESTOCK - Hampshire Hog",
"LIVESTOCK - Berkshire Pig",
"LIVESTOCK - Angus Cow",
"LIVESTOCK - Hereford Cow",
"LIVESTOCK - Jersey Cow",
"LIVESTOCK - Dorper Sheep",
"LIVESTOCK - Merino Sheep",
"LIVESTOCK - Nubian Goat",
"LIVESTOCK - Boer Goat",
"LIVESTOCK - Mule",
"LIVESTOCK - Saanen Goat",
"LIVESTOCK - Dorset Sheep",
"LIVESTOCK - Brahman Cattle",
"LIVESTOCK - Landrace Pig",
"LIVESTOCK - Pygmy Goat",
"LIVESTOCK - Hampshire Sheep",
"LIVESTOCK - Suffolk Sheep",
"LIVESTOCK - Texel Sheep",
"LIVESTOCK - Guernsey Cow",
"LIVESTOCK - Devon Sheep",
"LIVESTOCK - Alpine Goat",
"LIVESTOCK - Large White Pig",
"LIVESTOCK - Cheviot Sheep",
"LIVESTOCK - Ayrshire Cow",
"LIVESTOCK - Hampshire Cattle",
"LIVESTOCK - Targhee Sheep",
"LIVESTOCK - Alpine Sheep",
"LIVESTOCK - Alentejano Pig",
"LIVESTOCK - Baladi Sheep",
"LIVESTOCK - Dorper Goat",
"LIVESTOCK - Florida White Goat",

# 🐦 BIRDS
"BIRDS - Duck",
"BIRDS - Goose",
"BIRDS - Parrot",
"BIRDS - Budgerigar",
"BIRDS - Canary",
"BIRDS - Cockatiel",
"BIRDS - Pigeon",
"BIRDS - Quail",
"BIRDS - Lovebird",
"BIRDS - Macaw",
"BIRDS - Oriole",
"BIRDS - Peacock",
"BIRDS - Sparrow",
"BIRDS - Swan",
"BIRDS - Emu",
"BIRDS - Xantus's Hummingbird",
"BIRDS - Warbler",
"BIRDS - Finch",
"BIRDS - Cockatoo",
"BIRDS - Toucan",
"BIRDS - Kingfisher",
"BIRDS - Falcon",
"BIRDS - Hawk",
"BIRDS - Eagle",
"BIRDS - Owl",
"BIRDS - Heron",
"BIRDS - Pheasant",
"BIRDS - Kookaburra",
"BIRDS - Ibis",
"BIRDS - Tanager",
"BIRDS - Parakeet",
"BIRDS - Nightingale",
"BIRDS - Woodpecker",
"BIRDS - Bluebird",
"BIRDS - Cardinal",
"BIRDS - Mockingbird",
"BIRDS - Plover",
"BIRDS - Seagull",
"BIRDS - Starling",
"BIRDS - Parrotlet",
"BIRDS - Sun Conure",
"BIRDS - African Grey",
"BIRDS - Eclectus Parrot",
"BIRDS - Amazon Parrot",
"BIRDS - Quaker Parrot",
"BIRDS - Green Cheek Conure",
"BIRDS - Pionus Parrot",
"BIRDS - Blue-and-Gold Macaw",
"BIRDS - Scarlet Macaw",
"BIRDS - Hyacinth Macaw",
"BIRDS - Blue Jay",
"BIRDS - Robin",
"BIRDS - Wood Duck",
"BIRDS - Mandarin Duck",
"BIRDS - Mute Swan",
"BIRDS - Black Swan",
"BIRDS - Myna",
"BIRDS - Raven",
"BIRDS - Cormorant",
"BIRDS - Gull",
"BIRDS - Kite",
"BIRDS - Buzzard",

# 🐹 SMALL MAMMALS
"SMALL MAMMALS - Rabbit",
"SMALL MAMMALS - Hamster",
"SMALL MAMMALS - Ferret",
"SMALL MAMMALS - Guinea Pig",
"SMALL MAMMALS - Gerbil",
"SMALL MAMMALS - Hedgehog",
"SMALL MAMMALS - Mongoose",
"SMALL MAMMALS - Chinchilla",
"SMALL MAMMALS - Sugar Glider",
"SMALL MAMMALS - Rat",
"SMALL MAMMALS - Mouse",
"SMALL MAMMALS - Degus",
"SMALL MAMMALS - Squirrel",
"SMALL MAMMALS - Prairie Dog",
"SMALL MAMMALS - Shrew",
"SMALL MAMMALS - Opossum",
"SMALL MAMMALS - Mole",
"SMALL MAMMALS - Weasel",
"SMALL MAMMALS - Stoat",
"SMALL MAMMALS - Fennec Fox",
"SMALL MAMMALS - African Pygmy Hedgehog",
"SMALL MAMMALS - European Hedgehog",
"SMALL MAMMALS - Indian Palm Squirrel",
"SMALL MAMMALS - Eastern Chipmunk",
"SMALL MAMMALS - Northern Short-tailed Shrew",
"SMALL MAMMALS - Eastern Grey Squirrel",
"SMALL MAMMALS - American Badger",
"SMALL MAMMALS - European Mole",
"SMALL MAMMALS - Southern Flying Squirrel",
"SMALL MAMMALS - Prairie Vole",
"SMALL MAMMALS - Naked Mole Rat",
"SMALL MAMMALS - African Pouched Rat",
"SMALL MAMMALS - Eastern Cottontail Rabbit",
"SMALL MAMMALS - European Rabbit",
"SMALL MAMMALS - House Mouse",
"SMALL MAMMALS - Brown Rat",

# 🐟 FISH
"FISH - Goldfish",
"FISH - Koi",
"FISH - Betta",
"FISH - Guppy",
"FISH - Angelfish",
"FISH - Tetra",
"FISH - Cichlid",
"FISH - Discus",
"FISH - Molly",
"FISH - Swordtail",
"FISH - Neon Tetra",
"FISH - Zebra Danio",
"FISH - Pleco",
"FISH - Gourami",
"FISH - Clownfish",
"FISH - Tiger Barb",
"FISH - Cherry Barb",
"FISH - Rasbora",
"FISH - Corydoras",
"FISH - Oscar",
"FISH - Silver Dollar",
"FISH - Cardinal Tetra",
"FISH - Discus Fish",
"FISH - Bristlenose Pleco",
"FISH - Harlequin Rasbora",
"FISH - Electric Yellow Cichlid",
"FISH - Pearl Gourami",
"FISH - Rummy Nose Tetra",
"FISH - Bolivian Ram",
"FISH - Firemouth Cichlid",
"FISH - Ram Cichlid",
"FISH - Jelly Bean Tetra",
"FISH - Leopard Danio",
"FISH - Black Molly",
"FISH - Blue Tang",
"FISH - Sailfin Molly",
"FISH - Fantail Goldfish",
"FISH - Pearlscale Goldfish",
"FISH - Rosy Red Minnow",
"FISH - Brichardi Cichlid",
"FISH - Peacock Bass",
"FISH - Convict Cichlid",

# 🐢 REPTILES
"REPTILES - Turtle",
"REPTILES - Tortoise",
"REPTILES - Leopard Gecko",
"REPTILES - Bearded Dragon",
"REPTILES - Ball Python",
"REPTILES - Corn Snake",
"REPTILES - Green Iguana",
"REPTILES - Red-eared Slider",
"REPTILES - Box Turtle",
"REPTILES - Chameleon",
"REPTILES - Blue-tongued Skink",
"REPTILES - King Snake",
"REPTILES - Garter Snake",
"REPTILES - African Sideneck Turtle",
"REPTILES - Eastern Box Turtle",
"REPTILES - Russian Tortoise",
"REPTILES - Hermann's Tortoise",
"REPTILES - Greek Tortoise",
"REPTILES - Argentine Black and White Tegu",
"REPTILES - Pancake Tortoise",
"REPTILES - Radiated Tortoise",
"REPTILES - Sulcata Tortoise",
"REPTILES - Desert Tortoise",
"REPTILES - Mud Turtle",
"REPTILES - Map Turtle",
"REPTILES - Musk Turtle",
"REPTILES - Painted Turtle",
"REPTILES - Softshell Turtle",
"REPTILES - Wood Turtle",
"REPTILES - Snapping Turtle",
"REPTILES - Leatherback Sea Turtle",
"REPTILES - Loggerhead Sea Turtle",
"REPTILES - Green Sea Turtle",
"REPTILES - Hawksbill Sea Turtle",
"REPTILES - Olive Ridley Sea Turtle",

# 🐝 INSECTS
"INSECTS - Bee",
"INSECTS - Butterfly",
"INSECTS - Ladybug",
"INSECTS - Dragonfly",
"INSECTS - Praying Mantis",
"INSECTS - Ant",
"INSECTS - Beetle",
"INSECTS - Grasshopper",
"INSECTS - Cockroach",
"INSECTS - Termite",
"INSECTS - Firefly",
"INSECTS - Moth",
"INSECTS - Wasp",
"INSECTS - Flea",
"INSECTS - Tick",
"INSECTS - Lacewing",
"INSECTS - Earwig",
"INSECTS - Silverfish",
"INSECTS - Aphid",
"INSECTS - Cicada",
"INSECTS - Katydid",
"INSECTS - Mealybug",
"INSECTS - Thrips",
"INSECTS - Stink Bug",
"INSECTS - Weevil",
"INSECTS - Whitefly",
"INSECTS - Scorpion",
"INSECTS - Locust",
"INSECTS - Mantid",
"INSECTS - Sawfly",
"INSECTS - Mayfly",
"INSECTS - Stonefly",
"INSECTS - Borer Beetle",
"INSECTS - Assassin Bug",
"INSECTS - Leafhopper",
"INSECTS - Longhorn Beetle",
"INSECTS - Lace Bug",
"INSECTS - Boxelder Bug",
"INSECTS - Slug",
"INSECTS - Snail",
"INSECTS - Spider",
"INSECTS - Scorpion",

# 🌿 PLANTS
"PLANTS - Venus Flytrap",
"PLANTS - Rose",
"PLANTS - Sunflower",
"PLANTS - Tulip",
"PLANTS - Orchid",
"PLANTS - Cactus",
"PLANTS - Aloe Vera",
"PLANTS - Bamboo",
"PLANTS - Lavender",
"PLANTS - Daffodil",
"PLANTS - Marigold",
"PLANTS - Peony",
"PLANTS - Fern",
"PLANTS - Ivy",
"PLANTS - Bonsai",
"PLANTS - Bamboo Orchid",
"PLANTS - Jasmine",
"PLANTS - Hydrangea",
"PLANTS - Lilac",
"PLANTS - Magnolia",
"PLANTS - Pothos",
"PLANTS - Snake Plant",
"PLANTS - Spider Plant",
"PLANTS - Philodendron",
"PLANTS - ZZ Plant",
"PLANTS - Ficus",
"PLANTS - Monstera",
"PLANTS - Peace Lily",
"PLANTS - Begonia",
"PLANTS - Pilea",
"PLANTS - Ceropegia",
"PLANTS - African Violet",
"PLANTS - English Ivy",
"PLANTS - Calathea",
"PLANTS - Schefflera",
"PLANTS - Dracaena",
"PLANTS - Croton",
"PLANTS - Asparagus Fern",
"PLANTS - Kalanchoe",
"PLANTS - Bromeliad",
"PLANTS - Hoya",
"PLANTS - Echeveria",
"PLANTS - Christmas Cactus",
"PLANTS - Kokedama",
"PLANTS - Air Plant",
"PLANTS - String of Pearls",
"PLANTS - Variegated Rubber Plant",
"PLANTS - Peperomia",
"PLANTS - Tradescantia",
"PLANTS - Schefflera Arboricola",
"PLANTS - Chinese Evergreen",
"PLANTS - Pilea Peperomioides",
"PLANTS - Anthurium",
"PLANTS - Prayer Plant",

# 🦓 CRYPTOANIMALS
"CRYPTOANIMALS - Bigfoot (Sasquatch)",
"CRYPTOANIMALS - Loch Ness Monster (Nessie)",
"CRYPTOANIMALS - Chupacabra",
"CRYPTOANIMALS - Mothman",
"CRYPTOANIMALS - Jersey Devil",
"CRYPTOANIMALS - Yeti",
"CRYPTOANIMALS - Bunyip",
"CRYPTOANIMALS - Thunderbird",
"CRYPTOANIMALS - Mokele-Mbembe",
"CRYPTOANIMALS - Ogopogo",
"CRYPTOANIMALS - Skunk Ape",
"CRYPTOANIMALS - Dover Demon",
"CRYPTOANIMALS - Lizard Man of Scape Ore Swamp",
"CRYPTOANIMALS - Fresno Nightcrawler",
"CRYPTOANIMALS - Altamaha-ha",
"CRYPTOANIMALS - Flatwoods Monster",
"CRYPTOANIMALS - Beast of Bray Road",
"CRYPTOANIMALS - White River Monster",
"CRYPTOANIMALS - Hopkinsville Goblins",
"CRYPTOANIMALS - Lake Champlain Monster (Champ)",
"CRYPTOANIMALS - Beast of Gévaudan",
"CRYPTOANIMALS - Lagarfljót Worm",
"CRYPTOANIMALS - Mapinguari",
"CRYPTOANIMALS - Ahool",
"CRYPTOANIMALS - Batsquatch",
"CRYPTOANIMALS - Enfield Horror",
"CRYPTOANIMALS - Goatman",
"CRYPTOANIMALS - Mongolian Death Worm",
"CRYPTOANIMALS - Ningen",
"CRYPTOANIMALS - Nahuelito",
"CRYPTOANIMALS - Orang Pendek",
"CRYPTOANIMALS - Ozark Howler",
"CRYPTOANIMALS - Seelkee",
"CRYPTOANIMALS - Tatzelwurm",
"CRYPTOANIMALS - Yowie",
"CRYPTOANIMALS - Lake Worth Monster",
"CRYPTOANIMALS - Beast of Exmoor",
"CRYPTOANIMALS - Snallygaster",
"CRYPTOANIMALS - Lusca",
"CRYPTOANIMALS - Beast of Busco",
"CRYPTOANIMALS - Firebird",
"CRYPTOANIMALS - Beast of Noonday",
"CRYPTOANIMALS - Kerberos",
"CRYPTOANIMALS - Hupia",
"CRYPTOANIMALS - Serpent of Lagarfljót",
"CRYPTOANIMALS - Mongolian Death Worm",
"CRYPTOANIMALS - Ahool",
"CRYPTOANIMALS - Udege Warg",
"CRYPTOANIMALS - Mapinguari",
"CRYPTOANIMALS - Elwetritsch",
"CRYPTOANIMALS - Ahuizotl",
"CRYPTOANIMALS - Encantado",
"CRYPTOANIMALS - Golconda",
"CRYPTOANIMALS - Lizard Man",
"CRYPTOANIMALS - Ogopogo",
"CRYPTOANIMALS - Scaly Monster",
"CRYPTOANIMALS - Sea Serpent",
"CRYPTOANIMALS - Mokele-Mbembe",
"CRYPTOANIMALS - Kraken",
"CRYPTOANIMALS - Sea Monster",
"CRYPTOANIMALS - Megalodon",
"CRYPTOANIMALS - Globster",
"CRYPTOANIMALS - Neptune's Horse",
"CRYPTOANIMALS - Leviathan",
"CRYPTOANIMALS - Sea Devil",
"CRYPTOANIMALS - Lake Monsters",
"CRYPTOANIMALS - Ghost Ship Creature",
"CRYPTOANIMALS - Giant Squid",
"CRYPTOANIMALS - Sea Dragon",
"CRYPTOANIMALS - Sea Serpent",
"CRYPTOANIMALS - Monster Shark",
"CRYPTOANIMALS - Abyssal Beast",
"CRYPTOANIMALS - Oceanic Hydra",
"CRYPTOANIMALS - Deep Sea Leviathan",
"CRYPTOANIMALS - Phantom Whale",
"CRYPTOANIMALS - Ghost Dolphin",
"CRYPTOANIMALS - Black Sea Monster",
"CRYPTOANIMALS - Blue Lake Monster",
"CRYPTOANIMALS - River Serpent",
"CRYPTOANIMALS - Lake Monster",

# 🦕 DINOSAURS
"DINOSAURS - Tyrannosaurus Rex",
"DINOSAURS - Velociraptor",
"DINOSAURS - Triceratops",
"DINOSAURS - Brachiosaurus",
"DINOSAURS - Stegosaurus",
"DINOSAURS - Ankylosaurus",
"DINOSAURS - Allosaurus",
"DINOSAURS - Spinosaurus",
"DINOSAURS - Pteranodon",
"DINOSAURS - Diplodocus",
"DINOSAURS - Parasaurolophus",
"DINOSAURS - Iguanodon",
"DINOSAURS - Archaeopteryx",
"DINOSAURS - Carnotaurus",
"DINOSAURS - Deinonychus",
"DINOSAURS - Gallimimus",
"DINOSAURS - Compsognathus",
"DINOSAURS - Maiasaura",
"DINOSAURS - Oviraptor",
"DINOSAURS - Styracosaurus",
"DINOSAURS - Therizinosaurus",
"DINOSAURS - Dilophosaurus",
"DINOSAURS - Plesiosaurus",
"DINOSAURS - Elasmosaurus",
"DINOSAURS - Pachycephalosaurus",
"DINOSAURS - Troodon",
"DINOSAURS - Lambeosaurus",
"DINOSAURS - Torosaurus",
"DINOSAURS - Corythosaurus",
"DINOSAURS - Kentrosaurus",
"DINOSAURS - Camarasaurus",
"DINOSAURS - Herrerasaurus",
"DINOSAURS - Gigantoraptor",
"DINOSAURS - Mononykus",
"DINOSAURS - Ornithomimus",
"DINOSAURS - Suchomimus",
"DINOSAURS - Stygimoloch",
"DINOSAURS - Tsintaosaurus",
"DINOSAURS - Tropeognathus",
"DINOSAURS - Utahraptor",
"DINOSAURS - Vulcanodon",
"DINOSAURS - Zuniceratops",
"DINOSAURS - Europasaurus",
"DINOSAURS - Massospondylus",
"DINOSAURS - Plateosaurus",
"DINOSAURS - Gasosaurus",
"DINOSAURS - Shunosaurus",
"DINOSAURS - Eustreptospondylus",
"DINOSAURS - Isisaurus",
"DINOSAURS - Sinosauropteryx",
"DINOSAURS - Sinornithosaurus",
"DINOSAURS - Microraptor",
"DINOSAURS - Anchiornis",
"DINOSAURS - Beipiaosaurus",
"DINOSAURS - Caudipteryx",
"DINOSAURS - Epidexipteryx",
"DINOSAURS - Jeholornis",
"DINOSAURS - Liaoningosaurus",
"DINOSAURS - Saurornithoides",
"DINOSAURS - Wuerhosaurus",
"DINOSAURS - Zhanghenglong",
"DINOSAURS - Nanshiungosaurus",
"DINOSAURS - Changchunsaurus",
"DINOSAURS - Ornitholestes",
"DINOSAURS - Othnielia",
"DINOSAURS - Coelophysis",
"DINOSAURS - Procompsognathus",
"DINOSAURS - Herrerasaurus",
"DINOSAURS - Plateosaurus",
"DINOSAURS - Massospondylus",
"DINOSAURS - Eoraptor",
"DINOSAURS - Hexinlusaurus",
"DINOSAURS - Zupaysaurus",
"DINOSAURS - Eustreptospondylus",
"DINOSAURS - Heterodontosaurus",
"DINOSAURS - Lesothosaurus",
"DINOSAURS - Abrictosaurus",
"DINOSAURS - Pectinodon",
"DINOSAURS - Dromaeosaurus",
"DINOSAURS - Saurornitholestes",
"DINOSAURS - Austroraptor",
"DINOSAURS - Buitreraptor",
"DINOSAURS - Rahonavis",
"DINOSAURS - Troodon",
"DINOSAURS - Ornithomimus",
"DINOSAURS - Gallimimus",
"DINOSAURS - Struthiomimus",
"DINOSAURS - Ornitholestes",
"DINOSAURS - Tanycolagreus",
"DINOSAURS - Coelurus",
"DINOSAURS - Draconyx",
"DINOSAURS - Futalognkosaurus",
"DINOSAURS - Magyarosaurus",
"DINOSAURS - Omeisaurus",
"DINOSAURS - Euhelopus",
"DINOSAURS - Giraffatitan",
"DINOSAURS - Tornieria",
"DINOSAURS - Shunosaurus",
"DINOSAURS - Nigersaurus",
"DINOSAURS - Atlasaurus",
"DINOSAURS - Ligabueino",
"DINOSAURS - Omeisaurus",
"DINOSAURS - Patagotitan",
"DINOSAURS - Saltasaurus",
"DINOSAURS - Antarctosaurus",
"DINOSAURS - Titanosaurus",
"DINOSAURS - Nemegtosaurus",
"DINOSAURS - Rapetosaurus",
"DINOSAURS - Mamenchisaurus",
"DINOSAURS - Patagosaurus",
"DINOSAURS - Daxiatitan",
"DINOSAURS - Europasaurus",
"DINOSAURS - Argentinosaurus",
"DINOSAURS - Puertasaurus",
"DINOSAURS - Dreadnoughtus",
"DINOSAURS - Australotitan",
"DINOSAURS - Notocolossus",
"DINOSAURS - Phuwiangosaurus",
"DINOSAURS - Ruyangosaurus",
"DINOSAURS - Spinophorosaurus",
"DINOSAURS - Wamweracaudia",
"DINOSAURS - Xenoceratops",
"DINOSAURS - Yongjinglong",
"DINOSAURS - Zephyrosaurus",

# 🦄 MYTHICAL CREATURES
"MYTHICAL CREATURES - Dragon",
"MYTHICAL CREATURES - Unicorn",
"MYTHICAL CREATURES - Griffin",
"MYTHICAL CREATURES - Phoenix",
"MYTHICAL CREATURES - Mermaid",
"MYTHICAL CREATURES - Centaur",
"MYTHICAL CREATURES - Pegasus",
"MYTHICAL CREATURES - Hydra",
"MYTHICAL CREATURES - Basilisk",
"MYTHICAL CREATURES - Kraken",
"MYTHICAL CREATURES - Sphinx",
"MYTHICAL CREATURES - Chimera",
"MYTHICAL CREATURES - Minotaur",
"MYTHICAL CREATURES - Fairy",
"MYTHICAL CREATURES - Elf",
"MYTHICAL CREATURES - Dwarf",
"MYTHICAL CREATURES - Goblin",
"MYTHICAL CREATURES - Troll",
"MYTHICAL CREATURES - Werewolf",
"MYTHICAL CREATURES - Vampire",
"MYTHICAL CREATURES - Banshee",
"MYTHICAL CREATURES - Cerberus",
"MYTHICAL CREATURES - Leviathan",
"MYTHICAL CREATURES - Siren",
"MYTHICAL CREATURES - Dryad",
"MYTHICAL CREATURES - Nymph",
"MYTHICAL CREATURES - Satyr",
"MYTHICAL CREATURES - Cyclops",
"MYTHICAL CREATURES - Gorgon",
"MYTHICAL CREATURES - Faun",
"MYTHICAL CREATURES - Selkie",
"MYTHICAL CREATURES - Manticore",
"MYTHICAL CREATURES - Kelpie",
"MYTHICAL CREATURES - Dragon Turtle",
"MYTHICAL CREATURES - Unicorn Stallion",
"MYTHICAL CREATURES - Hippocampus",
"MYTHICAL CREATURES - Valkyrie",
"MYTHICAL CREATURES - Fenrir",
"MYTHICAL CREATURES - Jinn",
"MYTHICAL CREATURES - Kitsune",
"MYTHICAL CREATURES - Naga",
"MYTHICAL CREATURES - Baku",
"MYTHICAL CREATURES - Gnome",
"MYTHICAL CREATURES - Brownie",
"MYTHICAL CREATURES - Leprechaun",
"MYTHICAL CREATURES - Oni",
"MYTHICAL CREATURES - Yeti",
"MYTHICAL CREATURES - Bigfoot",
"MYTHICAL CREATURES - Chupacabra",
"MYTHICAL CREATURES - Mothman",
"MYTHICAL CREATURES - Wendigo",
"MYTHICAL CREATURES - Thunderbird",
"MYTHICAL CREATURES - Gargoyle",
"MYTHICAL CREATURES - Pegasus",
"MYTHICAL CREATURES - Hydra",
"MYTHICAL CREATURES - Cerberus",
"MYTHICAL CREATURES - Banshee",
"MYTHICAL CREATURES - Griffin",
"MYTHICAL CREATURES - Dragon",
"MYTHICAL CREATURES - Phoenix",
"MYTHICAL CREATURES - Unicorn",
"MYTHICAL CREATURES - Centaur",
"MYTHICAL CREATURES - Mermaid",
"MYTHICAL CREATURES - Basilisk",
"MYTHICAL CREATURES - Chimera",
"MYTHICAL CREatures - Gorgon",
"MYTHICAL CREATURES - Minotaur",
"MYTHICAL CREATURES - Hydra",
"MYTHICAL CREATURES - Kraken",
"MYTHICAL CREATURES - Sphinx",
"MYTHICAL CREATURES - Fairy",
"MYTHICAL CREATURES - Elf",
"MYTHICAL CREATURES - Dwarf",
"MYTHICAL CREATURES - Goblin",
"MYTHICAL CREATURES - Troll",
"MYTHICAL CREATURES - Werewolf",
"MYTHICAL CREATURES - Vampire",
"MYTHICAL CREATURES - Banshee",

# 🌊 SEA ANIMALS
"SEA ANIMALS - Dolphin",
"SEA ANIMALS - Seahorse",
"SEA ANIMALS - Clownfish",
"SEA ANIMALS - Blue Tang",
"SEA ANIMALS - Octopus",
"SEA ANIMALS - Shrimp",
"SEA ANIMALS - Lobster",
"SEA ANIMALS - Crab",
"SEA ANIMALS - Starfish",
"SEA ANIMALS - Sea Turtle",
"SEA ANIMALS - Jellyfish",
"SEA ANIMALS - Coral",
"SEA ANIMALS - Manta Ray",
"SEA ANIMALS - Shark",
"SEA ANIMALS - Whale",
"SEA ANIMALS - Eel",
"SEA ANIMALS - Seal",
"SEA ANIMALS - Walrus",
"SEA ANIMALS - Manatee",
"SEA ANIMALS - Beluga Whale",
"SEA ANIMALS - Orca",
"SEA ANIMALS - Blue Whale",
"SEA ANIMALS - Humpback Whale",
"SEA ANIMALS - Gray Whale",
"SEA ANIMALS - Killer Whale",
"SEA ANIMALS - Bottlenose Dolphin",
"SEA ANIMALS - Pilot Whale",
"SEA ANIMALS - Dugong",
"SEA ANIMALS - Sea Lion",
"SEA ANIMALS - Harbor Seal",
"SEA ANIMALS - Green Sea Turtle",
"SEA ANIMALS - Loggerhead Turtle",
"SEA ANIMALS - Leatherback Turtle",
"SEA ANIMALS - Kemp's Ridley Turtle",
"SEA ANIMALS - Hawksbill Turtle",
"SEA ANIMALS - Spinner Dolphin",
"SEA ANIMALS - Risso's Dolphin",
"SEA ANIMALS - Atlantic Puffin",
"SEA ANIMALS - Common Dolphin",
"SEA ANIMALS - Basking Shark",
"SEA ANIMALS - Great White Shark",
"SEA ANIMALS - Whale Shark",
"SEA ANIMALS - Blue Marlin",
"SEA ANIMALS - Swordfish",
"SEA ANIMALS - Mako Shark",
"SEA ANIMALS - Tiger Shark",
"SEA ANIMALS - Thresher Shark",
"SEA ANIMALS - Nurse Shark",
"SEA ANIMALS - Hammerhead Shark",
"SEA ANIMALS - Blacktip Shark",
"SEA ANIMALS - Whale Ray",
"SEA ANIMALS - Moray Eel",
"SEA ANIMALS - Lionfish",
"SEA ANIMALS - Angelfish",
"SEA ANIMALS - Lion's Mane Jellyfish",
"SEA ANIMALS - Moon Jellyfish",
"SEA ANIMALS - Sea Urchin",
"SEA ANIMALS - Blue-ringed Octopus",
"SEA ANIMALS - Giant Pacific Octopus",
"SEA ANIMALS - Box Jellyfish",
"SEA ANIMALS - Portuguese Man o' War",
"SEA ANIMALS - Cuttlefish",
"SEA ANIMALS - Sea Slug",
"SEA ANIMALS - Giant Clam",
"SEA ANIMALS - Cone Snail",
"SEA ANIMALS - Sea Cucumber",
"SEA ANIMALS - Nudibranch",
"SEA ANIMALS - Horseshoe Crab",
"SEA ANIMALS - Electric Eel",
"SEA ANIMALS - Pufferfish",
"SEA ANIMALS - Crown-of-Thorns Starfish",
"SEA ANIMALS - Blue Sea Dragon",
"SEA ANIMALS - Mantis Shrimp",
"SEA ANIMALS - Sea Spider",
"SEA ANIMALS - Blue Dragon Nudibranch",
"SEA ANIMALS - Sea Anemone",
"SEA ANIMALS - Sea Horse",
"SEA ANIMALS - Frilled Shark",
"SEA ANIMALS - Goblin Shark",
"SEA ANIMALS - Sea Lamprey",
"SEA ANIMALS - Sperm Whale",
"SEA ANIMALS - Bowhead Whale",
"SEA ANIMALS - Fin Whale",
"SEA ANIMALS - Sei Whale",
"SEA ANIMALS - Bryde's Whale",
"SEA ANIMALS - Cuvier's Beaked Whale",
"SEA ANIMALS - Pygmy Sperm Whale",
"SEA ANIMALS - Short-finned Pilot Whale",
"SEA ANIMALS - False Killer Whale",
"SEA ANIMALS - Striped Dolphin",
"SEA ANIMALS - Rough-toothed Dolphin",
"SEA ANIMALS - Indo-Pacific Humpback Dolphin",
"SEA ANIMALS - Atlantic Humpback Dolphin",
"SEA ANIMALS - Amazon River Dolphin",
"SEA ANIMALS - Irrawaddy Dolphin",
"SEA ANIMALS - Vaquita",
"SEA ANIMALS - Dall's Porpoise",
"SEA ANIMALS - Harbor Porpoise",
"SEA ANIMALS - Finless Porpoise",
"SEA ANIMALS - Burmeister's Porpoise",
"SEA ANIMALS - Spectacled Porpoise",
"SEA ANIMALS - Chinese White Dolphin",
"SEA ANIMALS - Hector's Dolphin",
"SEA ANIMALS - Maui's Dolphin",
"SEA ANIMALS - Commerson's Dolphin",
"SEA ANIMALS - Spotted Dolphin",
"SEA ANIMALS - Hourglass Dolphin",
"SEA ANIMALS - Black Dolphin",
"SEA ANIMALS - Australian Snubfin Dolphin",
"SEA ANIMALS - Fraser's Dolphin",
"SEA ANIMALS - Atlantic Bottlenose Dolphin",
"SEA ANIMALS - Pacific Bottlenose Dolphin",
]

HOLIDAYS = [
"-"
# 🎉 SEASONAL HOLIDAYS
"SEASONAL HOLIDAYS - New Year",
"SEASONAL HOLIDAYS - Valentine's Day",
"SEASONAL HOLIDAYS - Easter",
"SEASONAL HOLIDAYS - Halloween",
"SEASONAL HOLIDAYS - Thanksgiving",
"SEASONAL HOLIDAYS - Christmas",
"SEASONAL HOLIDAYS - Hanukkah",
"SEASONAL HOLIDAYS - Diwali",
"SEASONAL HOLIDAYS - Lunar New Year",
"SEASONAL HOLIDAYS - Mardi Gras",
"SEASONAL HOLIDAYS - Palm Sunday",
"SEASONAL HOLIDAYS - Good Friday",
"SEASONAL HOLIDAYS - Back to School",

# 🏛️ NATIONAL HOLIDAYS
"NATIONAL HOLIDAYS - Memorial Day",
"NATIONAL HOLIDAYS - Labor Day",
"NATIONAL HOLIDAYS - Veterans Day",
"NATIONAL HOLIDAYS - Independence Day",
"NATIONAL HOLIDAYS - Presidents' Day",
"NATIONAL HOLIDAYS - Columbus Day",
"NATIONAL HOLIDAYS - Flag Day",
"NATIONAL HOLIDAYS - National Day",

# 🕍 RELIGIOUS HOLIDAYS
"RELIGIOUS HOLIDAYS - Easter",
"RELIGIOUS HOLIDAYS - Hanukkah",
"RELIGIOUS HOLIDAYS - Diwali",
"RELIGIOUS HOLIDAYS - Ramadan",
"RELIGIOUS HOLIDAYS - Passover",
"RELIGIOUS HOLIDAYS - Yom Kippur",
"RELIGIOUS HOLIDAYS - Sukkot",
"RELIGIOUS HOLIDAYS - Rosh Hashanah",
"RELIGIOUS HOLIDAYS - Purim",
"RELIGIOUS HOLIDAYS - Navaratri",
"RELIGIOUS HOLIDAYS - Lent",

# 🌍 CULTURAL CELEBRATIONS
"CULTURAL CELEBRATIONS - St. Patrick's Day",
"CULTURAL CELEBRATIONS - Cinco de Mayo",
"CULTURAL CELEBRATIONS - Oktoberfest",
"CULTURAL CELEBRATIONS - Bastille Day",
"CULTURAL CELEBRATIONS - Guy Fawkes Night",
"CULTURAL CELEBRATIONS - Sinterklaas",

# 🏅 SPORTS AND EVENTS
"SPORTS AND EVENTS - Super Bowl",
"SPORTS AND EVENTS - FIFA World Cup",
"SPORTS AND EVENTS - Olympic Games",
"SPORTS AND EVENTS - NBA Finals",
"SPORTS AND EVENTS - Wimbledon",
"SPORTS AND EVENTS - Stanley Cup Finals",
"SPORTS AND EVENTS - Tour de France",
"SPORTS AND EVENTS - The Masters (Golf)",
"SPORTS AND EVENTS - World Series",
"SPORTS AND EVENTS - UEFA Champions League Final",
"SPORTS AND EVENTS - MLB All-Star Game",
"SPORTS AND EVENTS - NCAA March Madness",
"SPORTS AND EVENTS - Kentucky Derby",
"SPORTS AND EVENTS - Indianapolis 500",
"SPORTS AND EVENTS - Daytona 500",
"SPORTS AND EVENTS - Boston Marathon",
"SPORTS AND EVENTS - Ryder Cup",
"SPORTS AND EVENTS - PGA Championship",
"SPORTS AND EVENTS - NFL Draft",
"SPORTS AND EVENTS - NBA All-Star Weekend",
"SPORTS AND EVENTS - The Ashes (Cricket)",
"SPORTS AND EVENTS - Rugby World Cup",
"SPORTS AND EVENTS - World Athletics Championships",
"SPORTS AND EVENTS - Australian Open (Tennis)",
"SPORTS AND EVENTS - French Open (Tennis)",
"SPORTS AND EVENTS - US Open (Tennis)",
"SPORTS AND EVENTS - Indian Premier League (Cricket)",
"SPORTS AND EVENTS - Formula 1 Grand Prix",
"SPORTS AND EVENTS - PGA Tour Events",
"SPORTS AND EVENTS - ATP/WTA Finals (Tennis)",
"SPORTS AND EVENTS - Champions League Quarter-Finals",
"SPORTS AND EVENTS - UEFA European Championship",

# 🌐 AWARENESS DAYS
"AWARENESS DAYS - International Women's Day",
"AWARENESS DAYS - International Men's Day",
"AWARENESS DAYS - World Health Day",
"AWARENESS DAYS - World Environment Day",
"AWARENESS DAYS - World Mental Health Day",
"AWARENESS DAYS - International Peace Day",
"AWARENESS DAYS - World Food Day",
"AWARENESS DAYS - World Water Day",
"AWARENESS DAYS - International Day of Happiness",
"AWARENESS DAYS - World Kindness Day",
"AWARENESS DAYS - International Friendship Day",
"AWARENESS DAYS - World Tourism Day",
"AWARENESS DAYS - World Book Day",
"AWARENESS DAYS - World Music Day",
"AWARENESS DAYS - World Art Day",
"AWARENESS DAYS - World Poetry Day",
"AWARENESS DAYS - World Cinema Day",
"AWARENESS DAYS - World Television Day",
"AWARENESS DAYS - World Internet Day",
"AWARENESS DAYS - World Wildlife Day",
"AWARENESS DAYS - World Oceans Day",
"AWARENESS DAYS - World Habitat Day",
"AWARENESS DAYS - World Population Day",
"AWARENESS DAYS - World Economic Forum",
"AWARENESS DAYS - World Technology Day",
"AWARENESS DAYS - World Innovation Day",
"AWARENESS DAYS - World Design Day",
"AWARENESS DAYS - World Entrepreneurship Day",
"AWARENESS DAYS - World Creativity and Innovation Day",
"AWARENESS DAYS - International Day of Families",
"AWARENESS DAYS - International Day of Charity",
"AWARENESS DAYS - International Day for the Elimination of Racial Discrimination",
"AWARENESS DAYS - International Day of Persons with Disabilities",
"AWARENESS DAYS - International Day Against Drug Abuse and Illicit Trafficking",
"AWARENESS DAYS - International Day of Older Persons",
"AWARENESS DAYS - International Day of the Girl Child",
"AWARENESS DAYS - International Day for Biological Diversity",
"AWARENESS DAYS - International Migrants Day",
"AWARENESS DAYS - International Day of Forests",
"AWARENESS DAYS - International Day of the Midwife",
"AWARENESS DAYS - International Day of Cooperatives",
"AWARENESS DAYS - International Day of Solidarity with the Palestinian People",
"AWARENESS DAYS - World Health Organization Day",
"AWARENESS DAYS - World Press Freedom Day",
"AWARENESS DAYS - World Health Care Day",
"AWARENESS DAYS - World Nursing Day",
"AWARENESS DAYS - World Veterinary Day",
"AWARENESS DAYS - World Blood Donor Day",
"AWARENESS DAYS - World Autism Awareness Day",
"AWARENESS DAYS - World Breastfeeding Week",
"AWARENESS DAYS - World Oral Health Day",
"AWARENESS DAYS - World Diabetes Day",
"AWARENESS DAYS - World Kidney Day",
"AWARENESS DAYS - World Arthritis Day",
"AWARENESS DAYS - World Asthma Day",
"AWARENESS DAYS - World No Tobacco Day",
"AWARENESS DAYS - World Migratory Bird Day",
"AWARENESS DAYS - World Turtle Day",
"AWARENESS DAYS - World Bee Day",
"AWARENESS DAYS - World Polar Day",
"AWARENESS DAYS - World Meteorological Day",
"AWARENESS DAYS - World Earth Day",
"AWARENESS DAYS - World Soil Day",
"AWARENESS DAYS - World Energy Conservation Day",
"AWARENESS DAYS - World Telecommunication and Information Society Day",
"AWARENESS DAYS - World Antimicrobial Awareness Week",
"AWARENESS DAYS - World Homeless Day",
"AWARENESS DAYS - World Refugee Day",
"AWARENESS DAYS - World Children's Day",
"AWARENESS DAYS - World Elder Abuse Awareness Day",
"AWARENESS DAYS - World Humanitarian Day",
"AWARENESS DAYS - World Day for Cultural Diversity",
"AWARENESS DAYS - World Indigenous Peoples Day",
"AWARENESS DAYS - World Braille Day",
"AWARENESS DAYS - World Red Cross and Red Crescent Day",
"AWARENESS DAYS - World Voice Day",
"AWARENESS DAYS - World Science Day",
"AWARENESS DAYS - World Mathematics Day",
"AWARENESS DAYS - World Space Week",
"AWARENESS DAYS - World Postal Day",
"AWARENESS DAYS - World Day Against Trafficking in Persons",
"AWARENESS DAYS - World Landmine Day",

# 🍔 FOOD & DRINK DAYS
"FOOD & DRINK DAYS - National Pancake Day",
"FOOD & DRINK DAYS - National Margarita Day",
"FOOD & DRINK DAYS - National Taco Day",
"FOOD & DRINK DAYS - National Beer Day",
"FOOD & DRINK DAYS - National Coffee Day",
"FOOD & DRINK DAYS - National Pizza Day",
"FOOD & DRINK DAYS - National Donut Day",
"FOOD & DRINK DAYS - National Ice Cream Day",
"FOOD & DRINK DAYS - National Burger Day",
"FOOD & DRINK DAYS - National Popcorn Day",
"FOOD & DRINK DAYS - National Cookie Day",
"FOOD & DRINK DAYS - National Cupcake Day",
"FOOD & DRINK DAYS - National Bagel Day",
"FOOD & DRINK DAYS - National Brownie Day",
"FOOD & DRINK DAYS - National Macaron Day",
"FOOD & DRINK DAYS - National Waffle Day",
"FOOD & DRINK DAYS - National Tea Day",
"FOOD & DRINK DAYS - National Martini Day",
"FOOD & DRINK DAYS - National Cheese Day",
"FOOD & DRINK DAYS - National Peanut Butter Day",
"FOOD & DRINK DAYS - National Sauerkraut Day",
"FOOD & DRINK DAYS - National Brisket Day",
"FOOD & DRINK DAYS - National Lasagna Day",
"FOOD & DRINK DAYS - National Spaghetti Day",
"FOOD & DRINK DAYS - National Meatball Day",
"FOOD & DRINK DAYS - National Sausage Day",
"FOOD & DRINK DAYS - National Oyster Day",
"FOOD & DRINK DAYS - National Lobster Day",
"FOOD & DRINK DAYS - National Crab Day",
"FOOD & DRINK DAYS - National Shrimp Day",
"FOOD & DRINK DAYS - National Caviar Day",
"FOOD & DRINK DAYS - National Truffle Day",
"FOOD & DRINK DAYS - National Steak Day",
"FOOD & DRINK DAYS - National Filet Mignon Day",
"FOOD & DRINK DAYS - National Rib Day",
"FOOD & DRINK DAYS - National Meatloaf Day",
"FOOD & DRINK DAYS - National Baked Potato Day",
"FOOD & DRINK DAYS - National Mashed Potato Day",
"FOOD & DRINK DAYS - National French Fry Day",
"FOOD & DRINK DAYS - National Tater Tot Day",
"FOOD & DRINK DAYS - National Sweet Potato Day",
"FOOD & DRINK DAYS - National Yam Day",
"FOOD & DRINK DAYS - National Carrot Day",
"FOOD & DRINK DAYS - National Broccoli Day",
"FOOD & DRINK DAYS - National Spinach Day",
"FOOD & DRINK DAYS - National Kale Day",
"FOOD & DRINK DAYS - National Lettuce Day",
"FOOD & DRINK DAYS - National Tomato Day",
"FOOD & DRINK DAYS - National Pepper Day",
"FOOD & DRINK DAYS - National Cucumber Day",
"FOOD & DRINK DAYS - National Zucchini Day",
"FOOD & DRINK DAYS - National Eggplant Day",
"FOOD & DRINK DAYS - National Mushroom Day",
"FOOD & DRINK DAYS - National Onion Day",
"FOOD & DRINK DAYS - National Garlic Day",
"FOOD & DRINK DAYS - National Ginger Day",
"FOOD & DRINK DAYS - National Turmeric Day",
"FOOD & DRINK DAYS - National Basil Day",
"FOOD & DRINK DAYS - National Oregano Day",
"FOOD & DRINK DAYS - National Thyme Day",
"FOOD & DRINK DAYS - National Rosemary Day",
"FOOD & DRINK DAYS - National Mint Day",
"FOOD & DRINK DAYS - National Cilantro Day",
"FOOD & DRINK DAYS - National Parsley Day",
"FOOD & DRINK DAYS - National Sage Day",
"FOOD & DRINK DAYS - National Dill Day",
"FOOD & DRINK DAYS - National Chive Day",
"FOOD & DRINK DAYS - National Tarragon Day",
"FOOD & DRINK DAYS - National Bay Leaf Day",
"FOOD & DRINK DAYS - National Herb Day",
"FOOD & DRINK DAYS - National Spice Day",

# 👨‍👩‍👧‍👦 FAMILY AND SOCIAL HOLIDAYS
"FAMILY AND SOCIAL HOLIDAYS - Mother's Day",
"FAMILY AND SOCIAL HOLIDAYS - Father's Day",
"FAMILY AND SOCIAL HOLIDAYS - Grandparents' Day",
"FAMILY AND SOCIAL HOLIDAYS - National Siblings Day",
"FAMILY AND SOCIAL HOLIDAYS - International Day of Families",

# 🛡️ SECURITY AND LEGAL HOLIDAYS
"SECURITY AND LEGAL HOLIDAYS - World Day Against Trafficking in Persons",
"SECURITY AND LEGAL HOLIDAYS - World Press Freedom Day",
"SECURITY AND LEGAL HOLIDAYS - World Landmine Day",

# 🌱 ENVIRONMENTAL HOLIDAYS
"ENVIRONMENTAL HOLIDAYS - World Environment Day",
"ENVIRONMENTAL HOLIDAYS - World Earth Day",
"ENVIRONMENTAL HOLIDAYS - World Water Day",
"ENVIRONMENTAL HOLIDAYS - World Oceans Day",
"ENVIRONMENTAL HOLIDAYS - World Wildlife Day",
"ENVIRONMENTAL HOLIDAYS - World Soil Day",
"ENVIRONMENTAL HOLIDAYS - World Energy Conservation Day",
"ENVIRONMENTAL HOLIDAYS - World Migratory Bird Day",
"ENVIRONMENTAL HOLIDAYS - World Turtle Day",
"ENVIRONMENTAL HOLIDAYS - World Bee Day",
"ENVIRONMENTAL HOLIDAYS - World Polar Day",
"ENVIRONMENTAL HOLIDAYS - World Meteorological Day",

# ❓ MISCELLANEOUS
"MISCELLANEOUS - Boss's Day",
"MISCELLANEOUS - National Pet Day",
"MISCELLANEOUS - National Teacher Day",
"MISCELLANEOUS - National Nurses Day",
"MISCELLANEOUS - National Scientists Day",
"MISCELLANEOUS - National Heroes Day",
"MISCELLANEOUS - National Volunteer Day",
"MISCELLANEOUS - World Health Care Day",
"MISCELLANEOUS - World Nursing Day",
"MISCELLANEOUS - World Veterinary Day",
"MISCELLANEOUS - World Blood Donor Day",
]

SPECIFIC_MODES = [
"Documentary Mode","Action Mode"
]

CAMERAS = {
"Experimental and Proto (Pre-1900s)": [
"EXPERIMENTAL - Nicéphore Niépce - Camera Obscura (1826) - PROS: First photograph ever taken, foundational to photography - CONS: Incredibly slow exposure times, impractical for modern use",
"EXPERIMENTAL - Louis Daguerre - Daguerreotype Camera (1839) - PROS: Sharp images for its time, paved the way for portraiture - CONS: Complex and dangerous development process",
"EXPERIMENTAL - William Henry Fox Talbot - Calotype Camera (1840) - PROS: Created the first negative images, allowed for reproduction - CONS: Limited sharpness and contrast",
"EXPERIMENTAL - Frederick Scott Archer - Wet Collodion Camera (1851) - PROS: Faster exposure times and sharper images than previous methods - CONS: Required immediate chemical processing in a darkroom",
"EXPERIMENTAL - E. & H.T. Anthony & Co. - Folding Camera (1854) - PROS: First commercially successful folding camera - CONS: Large and cumbersome",
"EXPERIMENTAL - Thomas Sutton - Panoramic Camera (1859) - PROS: First camera to capture 120-degree panoramic views - CONS: Limited use cases and complex operation",
"EXPERIMENTAL - Jules Duboscq - Stereoscopic Camera (1860) - PROS: Created the illusion of 3D images, popular for stereoscopic photography - CONS: Limited depth, complex to use",
"EXPERIMENTAL - Kodak - Kodak Box Camera (1888) - PROS: First consumer camera with roll film, democratized photography - CONS: Poor image quality compared to professional systems",
"EXPERIMENTAL - Charles Bennett - Gelatin Dry Plate Camera (1878) - PROS: Faster exposure times, no need for immediate processing - CONS: Required bulky plates",
"EXPERIMENTAL - Eadweard Muybridge - Zoopraxiscope (1879) - PROS: First device to project moving images - CONS: Complex, limited to scientific study",
"EXPERIMENTAL - George Eastman - Roll Film Camera (1885) - PROS: Enabled continuous photography, portable - CONS: Lower quality compared to plate cameras",
"EXPERIMENTAL - Joseph Nicéphore Niépce - Physautotype (1832) - PROS: First camera to use lavender oil as a solvent - CONS: Extremely fragile images, experimental",
"EXPERIMENTAL - Thomas Sutton - First Panoramic Camera (1859) - PROS: First attempt at capturing wide-angle images - CONS: Distorted edges, primitive lens technology",
"EXPERIMENTAL - William Friese-Greene - Chronophotographic Camera (1889) - PROS: One of the first motion-picture cameras - CONS: Experimental, limited success",
"EXPERIMENTAL - Le Prince - Single-Lens Camera (1888) - PROS: First single-lens motion-picture camera - CONS: Highly experimental and fragile",
"EXPERIMENTAL - Anschütz - Tachyscope (1887) - PROS: High-speed photography for scientific purposes - CONS: Bulky, limited commercial use",
"EXPERIMENTAL - Etienne-Jules Marey - Chronophotographic Gun (1882) - PROS: First to capture multiple frames per second - CONS: Only useful for scientific purposes, difficult to use",
"EXPERIMENTAL - Simon Wing - Multiplying Camera (1870s) - PROS: Allowed for multiple exposures on a single plate - CONS: Low quality, experimental",
"EXPERIMENTAL - Ives - Kinetoscope Prototype (1889) - PROS: First prototype for motion picture playback - CONS: Experimental, not commercially viable",
"EXPERIMENTAL - Kodak - Pocket Kodak Camera (1895) - PROS: First pocket-sized consumer camera - CONS: Limited features and poor resolution"
],
"1900s": [
"PROFESSIONAL - Graflex - Graflex Reflex Camera (1902) - PROS: First reflex camera with a mirror system for viewing - CONS: Large and heavy, complex to operate",
"PROFESSIONAL - Kodak - Brownie (1900) - PROS: Popularized photography for the masses, affordable - CONS: Poor image quality, limited to snapshots",
"PROFESSIONAL - Contessa-Nettel - Nettel Camera (1903) - PROS: High-quality folding plate camera, versatile - CONS: Bulky and fragile",
"CONSUMER - Kodak - Vest Pocket Kodak (1907) - PROS: Compact and portable, great for amateur photographers - CONS: Limited control and basic functionality",
"PROFESSIONAL - Thornton-Pickard - Ruby Reflex (1905) - PROS: First reflex camera with a moving mirror, ideal for high-speed action - CONS: Large and cumbersome",
"CONSUMER - Kodak - Folding Pocket Camera (1906) - PROS: Compact and portable, ideal for travel photography - CONS: Limited image quality compared to professional cameras",
"CONSUMER - Kodak - No. 2 Folding Autographic Brownie (1909) - PROS: Added autographic feature for writing notes on film - CONS: Still limited to amateur use",
"PROFESSIONAL - Goerz - Tenax (1905) - PROS: Known for sharp lenses, high-quality plate images - CONS: Heavy and difficult to carry",
"PROFESSIONAL - Ica - Reflex Camera (1909) - PROS: Improved reflex design with better handling - CONS: Limited shutter speeds, delicate build",
"PROFESSIONAL - Graflex - Auto Graflex (1906) - PROS: Allowed quick plate changes for professional studio work - CONS: Expensive, heavy",
"CONSUMER - Kodak - No. 3A Folding Pocket (1903) - PROS: Easy to use for amateur photographers, compact - CONS: Low image quality by professional standards",
"PROFESSIONAL - Conley - Universal Camera (1907) - PROS: Designed for professional portraiture, sharp images - CONS: Heavy and bulky",
"CONSUMER - Kodak - Panoram No. 4 (1904) - PROS: First consumer panoramic camera, perfect for landscapes - CONS: Low resolution, mechanical issues",
"PROFESSIONAL - Ernemann - Stereo Piccolette (1908) - PROS: One of the first cameras for stereoscopic photography - CONS: Complex to use, prone to misalignment",
"PROFESSIONAL - Folmer & Schwing - Century Studio Camera (1902) - PROS: Large format, ideal for studio portraits - CONS: Heavy and immobile",
"PROFESSIONAL - Graflex - Revolving Back Camera (1907) - PROS: Revolutionary revolving back for changing orientations - CONS: Limited to large plate sizes",
"PROFESSIONAL - Eastman Kodak - Panoram Kodak (1900) - PROS: First panoramic consumer camera - CONS: Poor image quality, bulky",
"CONSUMER - Kodak - Autographic Kodak Jr. (1909) - PROS: Early version with autographic feature for writing on film - CONS: Limited professional use",
"CONSUMER - Ica - Ideal Camera (1908) - PROS: Light and compact, ideal for travel - CONS: Fragile and complex to operate",
"PROFESSIONAL - Carl Zeiss - Jena (1901) - PROS: High-quality lenses, sharp and clear images - CONS: Expensive and difficult to operate"
],
"1910s": [
"PROFESSIONAL - Ernemann - Ermanox (1910) - PROS: Known for fast lenses and low-light capabilities - CONS: Heavy and expensive",
"PROFESSIONAL - Kodak - Vest Pocket Autographic (1912) - PROS: Small, easy to carry, popular with soldiers in WWI - CONS: Limited image quality, basic features",
"PROFESSIONAL - Contessa - Nettel Deckrullo (1910) - PROS: Introduced metal focal plane shutter, ideal for fast exposures - CONS: Fragile and hard to repair",
"CONSUMER - Kodak - Brownie No. 2 (1913) - PROS: Cheap and easy to use for beginners - CONS: Poor image quality, limited control",
"PROFESSIONAL - Goerz - Anschütz Camera (1911) - PROS: High-speed shutter, ideal for capturing motion - CONS: Expensive, limited distribution",
"PROFESSIONAL - Ica - Volta Reflex (1915) - PROS: Early reflex system for more precise framing - CONS: Heavy and bulky",
"CONSUMER - Kodak - Autographic Special (1914) - PROS: Allowed users to add notes directly on the film - CONS: Basic functionality, limited to amateurs",
"PROFESSIONAL - Ernemann - Stereo Camera (1916) - PROS: Known for sharp 3D images, early stereoscopic photography - CONS: Complex to operate, expensive",
"PROFESSIONAL - ICA - Kinamo (1919) - PROS: Early motion picture camera, portable for its time - CONS: Heavy and fragile",
"PROFESSIONAL - Kodak - Kodak Panoram (1910) - PROS: Wide-angle panoramic shots, ideal for landscapes - CONS: Low detail, difficult to operate",
"CONSUMER - Kodak - Brownie Autographic (1915) - PROS: Added note-taking features on film - CONS: Limited to basic snapshots",
"PROFESSIONAL - Zeiss Ikon - Miroflex (1917) - PROS: Excellent for studio photography, high-quality lenses - CONS: Expensive and complex",
"PROFESSIONAL - Goerz - Tenax (1918) - PROS: Sharp optics for detailed shots - CONS: Heavy, impractical for field work",
"CONSUMER - Kodak - No. 3A Autographic (1919) - PROS: Compact design, popular among travelers - CONS: Basic features, limited resolution",
"PROFESSIONAL - Voigtländer - Avus (1915) - PROS: Versatile plate camera for professionals - CONS: Limited lens options",
"CONSUMER - Kodak - Pocket Kodak (1910) - PROS: First pocket-sized camera, easy to carry - CONS: Low image quality, limited manual control",
"PROFESSIONAL - Ernemann - Ermanox Folding Camera (1917) - PROS: Excellent for low-light photography - CONS: Complex and delicate",
"PROFESSIONAL - Thornton-Pickard - Imperial (1911) - PROS: Large format for studio work, great for portraits - CONS: Limited to static subjects",
"CONSUMER - Kodak - Vest Pocket Kodak (1915) - PROS: Popular among soldiers, highly portable - CONS: Low-quality images, simple operation",
"PROFESSIONAL - Folmer & Schwing - Century Graphic (1910) - PROS: Great for large-format landscape shots - CONS: Bulky and heavy, requires a tripod"
],

"1920s": [
"PROFESSIONAL - Leica - Leica I (1925) - PROS: First compact 35mm camera, sharp images, ideal for street and travel photography - CONS: Expensive and hard to operate for beginners",
"PROFESSIONAL - Ernemann - Ermanox (1924) - PROS: Excellent for low-light photography due to fast lenses - CONS: Large and heavy, requires expertise to handle",
"PROFESSIONAL - Akeley - Akeley Gyro (1923) - PROS: Gyroscopic stabilization for smooth motion pictures - CONS: Bulky, complex to operate",
"CONSUMER - Kodak - Brownie 2A (1920) - PROS: Affordable and easy for amateurs - CONS: Limited image quality, low manual control",
"PROFESSIONAL - Mitchell - Standard 35mm (1929) - PROS: High-quality motion picture camera, excellent for studio productions - CONS: Bulky and suited only for controlled environments",
"CONSUMER - Kodak - No. 2 Folding Autographic (1923) - PROS: Popular among amateur photographers for portability - CONS: Low-quality images by modern standards",
"PROFESSIONAL - Pathé - Pathé-Baby (1922) - PROS: First widely available amateur 9.5mm film camera - CONS: Limited to personal use, substandard image quality",
"PROFESSIONAL - Graflex - Super D (1923) - PROS: Reflex camera with sharp, detailed results - CONS: Heavy and challenging for on-the-go shooting",
"CONSUMER - Kodak - Autographic Vest Pocket (1925) - PROS: Compact and portable for snapshots - CONS: Poor resolution, basic features",
"PROFESSIONAL - Contessa-Nettel - Cocarette (1926) - PROS: High-quality folding plate camera, excellent detail - CONS: Difficult to operate, fragile",
"PROFESSIONAL - Leica - Leica II (1928) - PROS: Introduced interchangeable lenses for creative versatility - CONS: Expensive and complex for amateurs",
"PROFESSIONAL - Debrie - Parvo L (1921) - PROS: Compact, highly reliable for silent films - CONS: Limited lens options, cumbersome for location work",
"PROFESSIONAL - Bell & Howell - Filmo 70 (1923) - PROS: Lightweight and portable, perfect for newsreel and documentary - CONS: Limited control compared to larger studio cameras",
"CONSUMER - Kodak - Folding Pocket Camera (1925) - PROS: Portable and easy to use, ideal for travel - CONS: Basic functionality, low image resolution",
"PROFESSIONAL - Ica - Kinamo (1926) - PROS: First portable motion picture camera for 35mm film - CONS: Limited to handheld shooting",
"CONSUMER - Kodak - Rainbow Hawkeye Vest Pocket (1929) - PROS: Popular for its colorful, compact design - CONS: Basic camera with limited control",
"PROFESSIONAL - Pathé - Pathé 28mm Camera (1922) - PROS: Great for early amateur films - CONS: Subpar image quality compared to professional cameras",
"PROFESSIONAL - Bell & Howell - Eyemo 35mm (1925) - PROS: Portable and durable, ideal for fieldwork and documentaries - CONS: Limited for complex shots",
"PROFESSIONAL - Newman-Sinclair - Autokine (1927) - PROS: Early success in handheld cinematography - CONS: Limited lens flexibility",
"CONSUMER - Kodak - Pocket Kodak (1928) - PROS: Compact and simple to use for everyday photography - CONS: Limited resolution and features"
],
"1930s": [
"PROFESSIONAL - Bell & Howell - 2709 Standard (1930) - PROS: Creates stable, high-quality shots perfect for dramatic lighting setups - CONS: Heavy and manual, best for studio work",
"PROFESSIONAL - Arriflex - Kinarri 35 (1937) - PROS: Lightweight for handheld shots, great for dynamic motion - CONS: Noisy, limits use in dialogue scenes",
"CONSUMER - Kodak - Cine-Kodak Eight Model 20 (1932) - PROS: Compact and affordable for amateurs, simple to use - CONS: Limited to 8mm film with lower resolution",
"PROFESSIONAL - Eyemo - 35mm Camera (1936) - PROS: Portable, great for newsreels and documentary filmmaking - CONS: Limited creative flexibility",
"CONSUMER - Kodak - Brownie Movie Camera (1935) - PROS: First amateur 8mm movie camera, affordable - CONS: Poor image quality compared to professional film",
"PROFESSIONAL - Debrie - Parvo L (1934) - PROS: Reliable for studio films, captures sharp imagery - CONS: Heavy, suited only for controlled environments",
"PROFESSIONAL - Mitchell - Standard 35mm (1930) - PROS: Delivers cinematic imagery, ideal for big-budget studio productions - CONS: Large and difficult to use in dynamic environments",
"PROFESSIONAL - Newman-Sinclair - Autokine 35mm (1931) - PROS: Portable for handheld use, retains quality in outdoor settings - CONS: Noisy and difficult for dialogue scenes",
"CONSUMER - Kodak - Cine-Kodak Model K (1934) - PROS: Affordable 16mm option for home movies - CONS: Limited image quality, basic features",
"CONSUMER - Kodak - Brownie Special Six-20 (1938) - PROS: Affordable, easy-to-use snapshot camera - CONS: Very basic functionality, limited creative control",
"PROFESSIONAL - Vinten - Model H 35mm (1939) - PROS: Perfect for high-quality cinematic footage, sturdy build - CONS: Heavy and not portable",
"CONSUMER - Kodak - Bantam (1935) - PROS: Portable and compact, good for amateur photography - CONS: Basic functionality, lower image quality",
"PROFESSIONAL - Agfa - Movector 16mm (1933) - PROS: Compact, great for amateur filmmakers using 16mm - CONS: Less flexible for creative filming",
"PROFESSIONAL - De Vry - Standard 35mm (1931) - PROS: Known for durability and portability in documentaries - CONS: Limited features compared to later models",
"CONSUMER - Leica - 16mm Film Camera (1936) - PROS: Compact and sharp, great for personal films - CONS: Limited lens options",
"PROFESSIONAL - Arriflex - Kinarri 35 (1936) - PROS: Ideal for on-the-go handheld shots, fluid motion - CONS: Noisy operation makes it unsuitable for sound recording",
"PROFESSIONAL - Bell & Howell - Filmo 70-D (1935) - PROS: Durable and portable for fieldwork - CONS: Limited in creative flexibility",
"CONSUMER - Kodak - Bullet Special (1937) - PROS: Easy to use and portable, great for snapshots - CONS: Very basic controls, limited image quality",
"PROFESSIONAL - Newman-Sinclair - Autokine (1934) - PROS: Great for early handheld cinematography - CONS: Limited flexibility in lens and shot composition",
"PROFESSIONAL - Vinten - Model H 35mm (1939) - PROS: Delivers sharp, professional-level imagery - CONS: Large and suited to studio work only"
],
"1940s": [
"PROFESSIONAL - Mitchell - BNC Camera (1941) - PROS: Perfect for cinematic studio films, excellent sharpness - CONS: Very heavy and not portable",
"PROFESSIONAL - Arriflex - 35 II (1946) - PROS: Versatile, great for handheld work and dynamic movement - CONS: Complex to operate, best for professionals",
"PROFESSIONAL - Bell & Howell - Filmo 70 (1940) - PROS: Portable for news and documentary, great for on-the-go shooting - CONS: Limited lens options for creative work",
"CONSUMER - Kodak - Ektra (1942) - PROS: Compact and accessible for amateurs - CONS: Limited control over creative aspects",
"CONSUMER - Kodak - Brownie 8mm (1946) - PROS: Affordable, great for family movies - CONS: Poor resolution, lacks features",
"PROFESSIONAL - Bolex - H16 (1941) - PROS: Excellent for indie films with great image quality - CONS: Manual controls require expertise",
"CONSUMER - Kodak - Cine-Kodak Special II (1948) - PROS: Popular for amateur 16mm films, good image quality - CONS: Limited to basic film features",
"CONSUMER - Bell & Howell - 16mm Magazine Camera (1940) - PROS: Compact for consumer use, accessible - CONS: Limited manual control",
"PROFESSIONAL - Mitchell - BNC Reflex (1947) - PROS: Best for studio films requiring crisp visuals and precision - CONS: Bulky and hard to transport",
"PROFESSIONAL - De Vry - Sound Camera (1945) - PROS: Excellent for synchronized sound recording - CONS: Heavy and requires stationary setup",
"CONSUMER - Revere - Model 99 8mm (1949) - PROS: Compact and easy to use - CONS: Basic image quality, limited control",
"PROFESSIONAL - Eclair - Cameflex (1947) - PROS: Compact and flexible, perfect for handheld shots - CONS: Limited creative control",
"CONSUMER - Kodak - Brownie Movie Camera (1946) - PROS: Affordable and accessible to beginners - CONS: Limited to basic snapshots and home movies",
"PROFESSIONAL - Pathé - WEBO M 16mm (1942) - PROS: Durable for professional use, great for fieldwork - CONS: Limited in creative lens options",
"PROFESSIONAL - Wilart - 16mm Camera (1948) - PROS: Good for amateur 16mm films, portable - CONS: Basic image quality, limited features",
"PROFESSIONAL - Mitchell - BNC Sound Camera (1949) - PROS: Ideal for capturing synchronized sound with sharp visuals - CONS: Best for controlled studio environments",
"CONSUMER - Kodak - Cine-Kodak 8 (1946) - PROS: First consumer 8mm movie camera - CONS: Basic features, limited image resolution",
"CONSUMER - Bell & Howell - 70DR (1944) - PROS: Compact and easy to carry for documentaries - CONS: Limited control over image quality",
"PROFESSIONAL - Revere - 16mm Cine Camera (1943) - PROS: Perfect for early amateur films, easy to use - CONS: Limited lens options for professional work",
"PROFESSIONAL - De Vry - 35mm Studio Camera (1947) - PROS: Known for its rugged build, used in harsh environments - CONS: Limited creative controls"
],
"1950s": [
"PROFESSIONAL - Mitchell - BNCR (1952) - PROS: Delivers extremely sharp, high-quality images, ideal for Hollywood-style productions - CONS: Very heavy and static, not suited for location work",
"PROFESSIONAL - Arriflex - 16ST (1953) - PROS: Great for handheld shots with smooth motion, perfect for dynamic filmmaking - CONS: Complex to use, best for professionals",
"CONSUMER - Kodak - Brownie 8mm Movie Camera II (1956) - PROS: Affordable for home movies, simple to use - CONS: Basic image quality, limited controls",
"PROFESSIONAL - Bell & Howell - 70DR (1951) - PROS: Known for durability and field use, great for outdoor scenes - CONS: Limited flexibility in creative shots",
"PROFESSIONAL - Eclair - NPR (1958) - PROS: Lightweight and portable, perfect for indie films - CONS: Limited lens options, not suited for high-end productions",
"CONSUMER - Bolex - H16 Reflex (1956) - PROS: Great for amateur filmmakers looking for 16mm film aesthetics - CONS: Requires manual knowledge of film cameras",
"PROFESSIONAL - Beaulieu - R16 (1959) - PROS: Excellent image quality with the ability to shoot in low-light - CONS: Expensive and complex for amateurs",
"CONSUMER - Canon - Cine 8T (1957) - PROS: First compact 8mm consumer camera, perfect for home movies - CONS: Poor resolution and limited manual control",
"CONSUMER - Kodak - Brownie Movie Camera (1952) - PROS: Affordable and accessible for beginners - CONS: Very basic functionality, limited creative control",
"PROFESSIONAL - Mitchell - BNCR (1955) - PROS: Great for detailed cinematic shots in a studio setting - CONS: Requires a large team to operate, very heavy",
"PROFESSIONAL - Pathé - WEBO M 16mm (1954) - PROS: Durable and reliable for professional 16mm film work - CONS: Limited for handheld or dynamic shots",
"CONSUMER - Revere - Eye-Matic EE127 (1958) - PROS: Compact, great for home movies - CONS: Limited resolution, low professional appeal",
"PROFESSIONAL - Bolex - H16 M (1959) - PROS: Reliable for small film productions with manual controls - CONS: Heavy and complex for amateurs",
"CONSUMER - Minolta - Autopak-8 D6 (1959) - PROS: Affordable and easy to use, ideal for consumer-grade films - CONS: Limited features, basic image quality",
"CONSUMER - Kodak - 16mm Magazine Camera (1951) - PROS: Affordable and simple for amateur filmmakers - CONS: Limited image resolution and control",
"PROFESSIONAL - Beaulieu - R16 (1959) - PROS: Excellent for low-light situations, sharp 16mm film results - CONS: Expensive and challenging for beginners",
"CONSUMER - Canon - Cine 8T (1957) - PROS: Compact and affordable for consumer use - CONS: Poor image quality compared to professional cameras",
"PROFESSIONAL - Mitchell - 70DR (1950) - PROS: Rugged and durable for location shooting - CONS: Not ideal for complex, indoor studio shots",
"PROFESSIONAL - Eclair - NPR (1958) - PROS: Perfect for handheld and location shots with dynamic movement - CONS: Limited for high-end cinematic work",
"CONSUMER - Kodak - Brownie Movie Camera II (1953) - PROS: Easy to use and accessible for beginners - CONS: Poor resolution, limited creative control"
],
"1960s": [
"PROFESSIONAL - Panavision - Silent Reflex (1962) - PROS: Silent operation, ideal for professional productions with sound - CONS: Expensive, heavy",
"PROFESSIONAL - Arriflex - 35 IIC (1964) - PROS: Compact and lightweight, ideal for handheld work - CONS: Complex, requires professional operation",
"PROFESSIONAL - Bolex - H16 EBM (1965) - PROS: Electric motor, great for longer takes - CONS: Requires extensive manual adjustments",
"PROFESSIONAL - Canon - Scoopic 16mm (1965) - PROS: Built-in light meter, lightweight - CONS: Limited manual control",
"PROFESSIONAL - Beaulieu - R16 Electric (1965) - PROS: Excellent image quality, ideal for indie filmmakers - CONS: Expensive, hard to maintain",
"PROFESSIONAL - Eclair - ACL (1967) - PROS: Lightweight and versatile, ideal for documentaries - CONS: Fragile, difficult for action shots",
"PROFESSIONAL - Mitchell - Mark II (1967) - PROS: Excellent sharpness, studio quality - CONS: Bulky, difficult to transport",
"CONSUMER - Kodak - Instamatic M2 (1963) - PROS: Affordable, easy to use - CONS: Low image quality, limited controls",
"CONSUMER - Various - Super 8 Cameras (1965) - PROS: Affordable, compact, accessible for amateurs - CONS: Basic image quality",
"CONSUMER - Nizo - S8 (1968) - PROS: High-quality Super 8 camera for enthusiasts - CONS: Limited features for professionals",
"CONSUMER - Bell & Howell - Filmosonic XL (1969) - PROS: Affordable, accessible to home movie makers - CONS: Low resolution, simple features",
"CONSUMER - Minolta - Autopak-8 D12 (1969) - PROS: Compact, easy to use - CONS: Limited manual controls",
"CONSUMER - Yashica - Super-8 Electro (1969) - PROS: Affordable Super 8, good for beginners - CONS: Limited manual control",
"CONSUMER - Chinon - Super 8 (1969) - PROS: Affordable, compact - CONS: Poor image quality compared to professional gear",
"CONSUMER - Agfa - Movexoom 6 (1968) - PROS: Affordable, compact - CONS: Basic features, limited resolution"
],
"1970s": [
"PROFESSIONAL - Panavision - Panaflex (1972) - PROS: Quiet, ideal for location shooting - CONS: Expensive, heavy",
"PROFESSIONAL - Arriflex - 35BL (1972) - PROS: Lightweight for handheld, great for dynamic work - CONS: Noisy, requires careful sound setups",
"PROFESSIONAL - Aaton - 7 LTR (1978) - PROS: Quiet, reliable for location sound recording - CONS: Expensive, limited lens options",
"CONSUMER - Beaulieu - 4008 ZM II Super 8 (1971) - PROS: High-quality Super 8, ideal for enthusiasts - CONS: Limited manual controls",
"CONSUMER - Canon - Auto Zoom 814 Electronic (1973) - PROS: Excellent for home movies, good image quality - CONS: Limited for professional use",
"PROFESSIONAL - Eclair - NPR (1971) - PROS: Lightweight, great for handheld work - CONS: Limited for high-end productions",
"PROFESSIONAL - Mitchell - VistaVision (1976) - PROS: Superb image quality, great for special effects - CONS: Heavy, not suitable for handheld work",
"PROFESSIONAL - Bolex - Pro 16mm (1970) - PROS: Durable, ideal for small film productions - CONS: Expensive, limited for complex shots",
"CONSUMER - Elmo - Super 110R (1974) - PROS: Affordable Super 8 for home movies - CONS: Limited features, basic controls",
"CONSUMER - Nikon - R10 Super 8 (1973) - PROS: Great image quality for home use - CONS: Limited for professional projects",
"CONSUMER - Minolta - XL-601 Super 8 (1976) - PROS: Compact, good for home movies - CONS: Limited manual control",
"CONSUMER - Chinon - Pacific 200/12 SMR (1975) - PROS: Affordable, easy to use - CONS: Limited image quality",
"CONSUMER - Pathé - DS8 Reflex (1973) - PROS: Good for amateur filmmaking - CONS: Limited creative options",
"CONSUMER - Leicina - Super RT1 (1974) - PROS: Compact and portable - CONS: Basic functionality",
"CONSUMER - Sankyo - XL 620 Super 8 (1978) - PROS: Affordable, accessible for beginners - CONS: Low image quality, limited control"
],
"1980s": [
"PROFESSIONAL - Arriflex - 765 (1989) - PROS: Perfect for large-scale cinematic productions - CONS: Heavy, expensive",
"PROFESSIONAL - Panavision - Panaflex Gold (1981) - PROS: Quiet operation, ideal for studio work - CONS: Bulky, expensive",
"PROFESSIONAL - Aaton - XTR Prod (1985) - PROS: Great for handheld shooting, perfect for indie films - CONS: Complex to operate",
"PROFESSIONAL - Sony - Betacam SP (1986) - PROS: First professional video format, excellent image quality - CONS: Expensive, complex",
"CONSUMER - Canon - L2 Hi8 (1988) - PROS: Affordable, good for home video - CONS: Limited manual control",
"CONSUMER - JVC - GR-C1 (1984) - PROS: Iconic compact video camera, great for home use - CONS: Basic image quality",
"PROFESSIONAL - Arriflex - SRII (1982) - PROS: Ideal for high-quality 16mm productions - CONS: Expensive, complex",
"PROFESSIONAL - Panasonic - WV-F250 (1988) - PROS: Great for early video news gathering - CONS: Heavy, expensive",
"PROFESSIONAL - Bolex - H16 SBM (1980) - PROS: Perfect for small productions with 16mm film - CONS: Heavy, manual operation",
"CONSUMER - Canon - T70 SLR with Video Back (1984) - PROS: Affordable for hybrid video and photography - CONS: Limited video quality",
"CONSUMER - Sony - Video8 Handycam CCD-M8U (1985) - PROS: Compact, great for home video - CONS: Low image quality",
"CONSUMER - Hitachi - VK-C820 (1989) - PROS: Affordable video camera for home use - CONS: Poor resolution compared to modern standards",
"PROFESSIONAL - JVC - KY-1900 (1983) - PROS: Excellent for early ENG (electronic news gathering) - CONS: Heavy, outdated by modern standards",
"PROFESSIONAL - Ikegami - HL-79 (1985) - PROS: Great for broadcast TV, excellent image quality - CONS: Complex to operate",
"CONSUMER - Bell & Howell - 2146 XL Super 8 (1981) - PROS: Affordable, great for home movies - CONS: Limited creative controls"
],
"1990s": [
"PROFESSIONAL - Panavision - Millennium (1997) - PROS: Industry-standard for high-budget productions - CONS: Extremely expensive, heavy",
"PROFESSIONAL - Arriflex - 435 (1995) - PROS: Excellent for fast-paced, dynamic shooting - CONS: Heavy, best for professionals",
"PROFESSIONAL - Aaton - 35-III (1991) - PROS: Great for handheld 35mm work, ideal for indie films - CONS: Expensive, complex to use",
"PROFESSIONAL - Sony - Digital Betacam DVW-700WS (1993) - PROS: Broadcast-standard video quality - CONS: Heavy, outdated by modern digital standards",
"CONSUMER - Canon - XL1 MiniDV (1997) - PROS: First affordable digital camcorder for indie filmmakers - CONS: Low resolution by modern standards",
"CONSUMER - Sony - DCR-VX1000 MiniDV (1995) - PROS: Affordable digital video for consumer use - CONS: Limited manual control",
"PROFESSIONAL - Arriflex - 235 (1999) - PROS: Lightweight for handheld 35mm film - CONS: Expensive, complex",
"PROFESSIONAL - Panasonic - AG-DVX100 (1999) - PROS: Excellent for indie films, affordable digital camcorder - CONS: Low resolution compared to modern cameras",
"PROFESSIONAL - Ikegami - HL-V73 (1994) - PROS: Excellent for TV broadcast, sharp video quality - CONS: Heavy, outdated by modern standards",
"PROFESSIONAL - JVC - GY-DV500 (1999) - PROS: Great for early digital productions - CONS: Low resolution, outdated",
"PROFESSIONAL - Sony - Betacam SX DNW-9WS (1996) - PROS: Industry-standard for video production - CONS: Outdated by modern digital video",
"PROFESSIONAL - Canon - EOS D2000 (1998) - PROS: First professional digital SLR - CONS: Low resolution by today’s standards",
"PROFESSIONAL - Nikon - D1 (1999) - PROS: First DSLR for professionals, fast autofocus - CONS: Low resolution",
"CONSUMER - Sony - DSC-F1 (1996) - PROS: Affordable, early digital photography - CONS: Poor image quality",
"CONSUMER - Casio - QV-10 (1995) - PROS: First consumer digital camera with an LCD - CONS: Low resolution, basic functionality"
],
"2000s": [
"PROFESSIONAL - RED - One (2007) - PROS: First digital cinema camera with 4K resolution - CONS: Expensive, requires complex setup",
"PROFESSIONAL - Panavision - Genesis (2005) - PROS: Hollywood-standard for digital cinema - CONS: Heavy, expensive",
"PROFESSIONAL - Arriflex - D-20 (2005) - PROS: Great for early digital cinematography, excellent image quality - CONS: Expensive, outdated by newer models",
"PROFESSIONAL - Sony - CineAlta F900 (2000) - PROS: First HD digital cinema camera - CONS: Low resolution by today’s standards",
"PROFESSIONAL - Canon - EOS 5D Mark II (2008) - PROS: First DSLR to shoot full HD video, great for indie filmmakers - CONS: Limited dynamic range for professional cinema",
"PROFESSIONAL - Panasonic - AG-HVX200 (2005) - PROS: Affordable, great for indie filmmakers - CONS: Limited resolution by modern standards",
"PROFESSIONAL - Sony - PMW-EX1 (2007) - PROS: Excellent for handheld HD shooting - CONS: Low-light performance could be better",
"PROFESSIONAL - RED - Epic (2009) - PROS: Capable of shooting 5K resolution, modular design - CONS: Expensive, complex to operate",
"PROFESSIONAL - ARRI - Alexa Classic (2009) - PROS: Industry-standard for digital cinema, beautiful color science - CONS: Expensive, bulky",
"PROFESSIONAL - Blackmagic - Cinema Camera (2008) - PROS: Affordable, great for indie filmmakers - CONS: Limited dynamic range and low-light performance",
"PROFESSIONAL - Canon - EOS 7D (2009) - PROS: Affordable DSLR with video capabilities, great for indie projects - CONS: Limited low-light performance",
"CONSUMER - Nikon - D90 (2008) - PROS: First DSLR with video recording - CONS: Limited manual control for video",
"CONSUMER - GoPro - HD Hero (2004) - PROS: Compact, affordable, great for action sports - CONS: Limited image quality compared to larger cameras",
"PROFESSIONAL - Sony - α7S (2008) - PROS: Excellent low-light performance, compact form factor - CONS: Rolling shutter issues with fast motion",
"PROFESSIONAL - Panasonic - Lumix GH1 (2009) - PROS: First mirrorless camera with HD video - CONS: Limited for professional use compared to full-frame cameras"
],
"2010s": [
"PROFESSIONAL - ARRI - Alexa Mini (2015) - PROS: Lightweight, ideal for handheld and drone use, beautiful image quality - CONS: Expensive",
"PROFESSIONAL - RED - Weapon (2016) - PROS: Capable of shooting up to 8K resolution, modular design - CONS: Expensive, complex to operate",
"PROFESSIONAL - Sony - Venice (2017) - PROS: Full-frame digital cinema, excellent dynamic range - CONS: Expensive, bulky",
"PROFESSIONAL - Blackmagic - URSA Mini Pro (2017) - PROS: Affordable digital cinema camera, great for indie films - CONS: Limited low-light performance",
"PROFESSIONAL - Canon - EOS C300 Mark II (2015) - PROS: Superb color science, great for documentary filmmaking - CONS: Expensive, complex to use",
"PROFESSIONAL - Panasonic - VariCam LT (2016) - PROS: Excellent low-light performance, versatile camera - CONS: Expensive",
"PROFESSIONAL - DJI - Inspire 2 with X7 Camera (2017) - PROS: Perfect for aerial cinematography, high-quality footage - CONS: Expensive, limited ground use",
"CONSUMER - GoPro - HERO7 Black (2018) - PROS: Great for action shots, waterproof, 4K video - CONS: Small sensor, limited dynamic range",
"PROFESSIONAL - Sony - α7 III (2018) - PROS: Excellent low-light performance, compact, great for video and photography - CONS: Limited for high-end cinema",
"PROFESSIONAL - Canon - EOS R (2018) - PROS: Excellent autofocus and image quality for video - CONS: Limited lens options",
"PROFESSIONAL - Nikon - Z6 (2018) - PROS: Excellent video quality for the price, great autofocus - CONS: Limited dynamic range compared to cinema cameras",
"PROFESSIONAL - Panasonic - Lumix GH5 (2017) - PROS: Great for video production, 4K video, small form factor - CONS: Micro four-thirds sensor limits low-light performance",
"PROFESSIONAL - Blackmagic - Pocket Cinema Camera 4K (2018) - PROS: Affordable digital cinema camera, excellent image quality - CONS: Limited battery life and low-light performance",
"PROFESSIONAL - Sony - FS7 II (2016) - PROS: Great for documentary and TV production, excellent dynamic range - CONS: Heavy, expensive",
"PROFESSIONAL - RED - Raven (2016) - PROS: Compact and affordable for RED standards, 4.5K resolution - CONS: Limited dynamic range compared to higher-end models"
],
"2020s": [
"PROFESSIONAL - ARRI - Alexa LF (2020) - PROS: Large format digital cinema, beautiful image quality - CONS: Expensive, heavy",
"PROFESSIONAL - RED - Komodo 6K (2020) - PROS: Compact, 6K resolution, affordable for RED standards - CONS: Limited dynamic range compared to higher-end models",
"PROFESSIONAL - Sony - FX9 (2020) - PROS: Full-frame digital cinema, excellent autofocus - CONS: Expensive, complex to use",
"PROFESSIONAL - Canon - EOS C500 Mark II (2020) - PROS: Full-frame digital cinema, superb color science - CONS: Expensive",
"PROFESSIONAL - Blackmagic - URSA Mini Pro 12K (2020) - PROS: 12K resolution, affordable for high-end cinema - CONS: Large files, requires powerful hardware for editing",
"PROFESSIONAL - Panasonic - Lumix S1H (2020) - PROS: Full-frame mirrorless camera with excellent video features - CONS: Expensive for mirrorless, limited for high-end cinema",
"PROFESSIONAL - Sony - α7S III (2020) - PROS: Excellent low-light performance, compact, 4K video - CONS: Limited for high-end cinema",
"PROFESSIONAL - Canon - EOS R5 (2020) - PROS: 8K video, excellent autofocus - CONS: Overheating issues with extended use",
"PROFESSIONAL - DJI - Ronin 4D (2021) - PROS: Integrated gimbal and cinema camera, excellent stabilization - CONS: Expensive, limited lens options",
"CONSUMER - GoPro - HERO10 Black (2021) - PROS: Great for action shots, 5.3K video, waterproof - CONS: Limited low-light performance",
"PROFESSIONAL - Sony - FX6 (2020) - PROS: Compact full-frame cinema camera, great for indie filmmakers - CONS: Limited dynamic range compared to larger cinema cameras",
"PROFESSIONAL - RED - V-Raptor ST (2021) - PROS: 8K resolution, excellent image quality - CONS: Expensive, complex to operate",
"PROFESSIONAL - Canon - EOS C70 (2020) - PROS: Compact digital cinema camera, excellent image quality - CONS: Limited lens options",
"PROFESSIONAL - Nikon - Z9 (2021) - PROS: 8K video, excellent image quality - CONS: Expensive, limited lens options for cinema",
"PROFESSIONAL - Panasonic - Lumix GH6 (2022) - PROS: Great for video production, 4K 120fps - CONS: Micro four-thirds sensor limits low-light performance"
],
"Future": [
"EXPERIMENTAL - Sony - A1Z - PROS: Projected 16K capabilities, AI-driven autofocus - CONS: Theoretical, potential high cost",
"EXPERIMENTAL - ARRI - Nova - PROS: AI-powered image processing, fully modular design - CONS: Theoretical, potential high complexity",
"EXPERIMENTAL - RED - Orion 12K - PROS: Hypothetical 12K+ resolution with deep color range - CONS: Projected massive data storage needs",
"EXPERIMENTAL - Canon - CinemaX 16K - PROS: Hypothetical 16K resolution, fully integrated with next-gen HDR - CONS: Requires next-gen hardware",
"EXPERIMENTAL - Blackmagic - Infinity Pro - PROS: Speculative affordable 8K+ cinema camera with AI editing tools - CONS: Hypothetical, unproven",
"EXPERIMENTAL - DJI - Phantom Cinema - PROS: Future drone with integrated 8K+ HDR video - CONS: Hypothetical, limited ground use",
"EXPERIMENTAL - Nikon - X12 Vision - PROS: Predicted to feature AI-driven sensor with real-time editing - CONS: Theoretical, unknown performance",
"EXPERIMENTAL - Panasonic - Genesis 32K - PROS: Speculative future 32K resolution camera with next-gen AI processing - CONS: Projected to have high cost and file size",
"EXPERIMENTAL - GoPro - FutureGo 12K - PROS: Hypothetical future action camera with 12K resolution, waterproof - CONS: Limited real-world use in extreme conditions",
"EXPERIMENTAL - Canon - EOS A6 - PROS: Future integration of computational photography with 10K video - CONS: Theoretical, high processing power needed",
"EXPERIMENTAL - RED - Helium Infinity - PROS: Hypothetical 10K resolution with 240fps capabilities - CONS: Data-intensive, requires next-gen storage",
"EXPERIMENTAL - Sony - α9 Super 8K - PROS: Speculative 8K camera with compact form factor, AI image enhancement - CONS: Theoretical, high cost",
"EXPERIMENTAL - Panasonic - Lumix VR - PROS: Projected to feature VR integration with 8K video - CONS: Hypothetical, unknown market readiness",
"EXPERIMENTAL - DJI - Inspire FutureX - PROS: Future drone integration with 16K aerial video - CONS: Theoretical, high data storage needs",
"EXPERIMENTAL - ARRI - AI Max - PROS: Predicted to feature AI-driven 12K+ image processing - CONS: Theoretical, unknown practicality in smaller productions"
]
}
selected_decade = "2020s"
print(CAMERAS[selected_decade])

# Decades list for UI or sorting purposes
DECADES = sorted(CAMERAS.keys())

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

        # Initialize video and audio save folders
        self.video_save_folder = ""  # Set to default or empty until populated
        self.audio_save_folder = ""  # Set to default or empty until populated

        # Initialize video_prompt_number_var here
        self.video_prompt_number_var = tk.IntVar(value=DEFAULT_PROMPTS)
        
        self.build_gui()

        # Initialize settings.json if it doesn't exist
        self.initialize_settings()

        # Check for Hugging Face API token
        self.check_huggingface_token()


        
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

# Utility function to save data to a file with error handling
    def save_to_file(self, content, file_path):
        """
        Saves the given content to the specified file path.
        
        Args:
            content (str): The content to save.
            file_path (str): The path to the file where content will be saved.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Content saved to {file_path}")
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            messagebox.showerror("Save Error", f"Failed to save file {file_path}:\n{e}")

        
    def validate_prompts(self, generated_prompts, expected_count):
        """
        Validates that each prompt contains exactly one positive and one negative section.
        
        Args:
            generated_prompts (str): The concatenated prompts.
            expected_count (int): The number of prompts expected.
        
        Returns:
            bool: True if all prompts are valid, False otherwise.
        """
        prompt_sets = [ps.strip() for ps in generated_prompts.strip().split("--------------------") if ps.strip()]
        if len(prompt_sets) != expected_count:
            print(f"Expected {expected_count} prompt sets, but got {len(prompt_sets)}.")
            return False

        for prompt_set in prompt_sets:
            positive_match = re.search(r"^positive:\s*(.+)$", prompt_set, re.IGNORECASE | re.DOTALL)
            negative_match = re.search(r"^negative:\s*(.+)$", prompt_set, re.IGNORECASE | re.DOTALL)
            if not positive_match or not negative_match:
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

    
    def check_huggingface_token(self):
        """
        Checks if the Huggingface API token is stored either in environment variables or settings file.
        If not found, prompts the user to enter it.
        """
        # Try to get from environment variable
        token = os.environ.get('HUGGINGFACE_API_TOKEN')
        if token:
            self.huggingface_api_token = token
            print("Huggingface API token loaded from environment variable.")
            return
        
        # Try to get from settings.json
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                try:
                    settings = json.load(f)
                except json.JSONDecodeError:
                    settings = {}
            token = settings.get('huggingface_api_token')
            if token:
                self.huggingface_api_token = token
                print("Huggingface API token loaded from settings file.")
                return
        
        # If not found, prompt the user to enter it
        self.prompt_huggingface_token()
    
    def prompt_huggingface_token(self):
        """
        Displays a popup window to prompt the user to enter their Huggingface API token.
        """
        popup = tk.Toplevel(self.root)
        popup.title("Huggingface API Token Required")
        popup.configure(bg='#0A2239')
        popup.geometry("500x200")
        popup.grab_set()  # Make it modal

        label = tk.Label(
            popup,
            text="Huggingface API token is required for Ollama.\nPlease enter your Huggingface API token below:",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12),
            justify='left'
        )
        label.pack(pady=20, padx=20)

        token_entry = tk.Entry(
            popup,
            show='*',
            width=50,
            font=('Helvetica', 12)
        )
        token_entry.pack(pady=10)

        def save_token():
            token = token_entry.get().strip()
            if not token:
                messagebox.showerror("Input Error", "Huggingface API token cannot be empty.")
                return
            # Save to settings.json
            settings = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    try:
                        settings = json.load(f)
                    except json.JSONDecodeError:
                        settings = {}
            settings['huggingface_api_token'] = token
            try:
                with open(SETTINGS_FILE, 'w') as f:
                    json.dump(settings, f, indent=4)
                self.huggingface_api_token = token
                messagebox.showinfo("Token Saved", "Huggingface API token has been saved successfully.")
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save the token: {e}")

        save_button = tk.Button(
            popup,
            text="Save Token",
            command=save_token,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 12, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=15
        )
        save_button.pack(pady=10)

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
        settings_menu.add_command(label="Set Output Directory", command=set_output_directory)
        settings_menu.add_command(label="Set ComfyUI Prompts Folder", command=set_comfyui_prompts_folder)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        self.root.config(menu=menubar)

        # Instructions
        self.instructions = tk.Label(
            self.root,
            text="Enter the desired prompt concept (max 400 characters):",
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
            text="Video Options - REQUIRED",
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
            text="Audio Options",
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

        # Buttons Row for Generate and Combine
        button_row_frame = tk.Frame(self.root, bg='#0A2239')
        button_row_frame.grid(row=5, column=0, pady=5, padx=5, sticky='ew')
        button_row_frame.columnconfigure((0,1,2), weight=1)

        # Generate Videos Button
        self.generate_videos_button = tk.Button(
            button_row_frame,
            text="Generate Videos",
            command=self.generate_videos,
            bg="#FF5733",
            fg="white",
            font=('Helvetica', 14, 'bold'),
            activebackground="#C70039",
            activeforeground='white',
            cursor="hand2",
            width=25,
            height=2
        )
        self.generate_videos_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Generate Sound Effects Button
        self.generate_sound_button = tk.Button(
            button_row_frame,
            text="Generate Sound Effects (BETA)",
            command=self.generate_sound_effects,
            bg="#28a745",
            fg="white",
            font=('Helvetica', 14, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=25,
            height=2
        )
        self.generate_sound_button.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

        # COMBINE button
        self.combine_button = tk.Button(
            button_row_frame,
            text="COMBINE (BETA)",
            command=self.combine_media,
            bg="#FFC107",
            fg="white",
            font=('Helvetica', 16, 'bold'),
            activebackground="#e0a800",
            activeforeground='white',
            cursor="hand2",
            width=30,
            height=3
        )
        self.combine_button.grid(row=0, column=2, padx=10, pady=10, sticky='ew')

        # Footer Frame containing Logo
        self.footer_frame = tk.Frame(self.root, bg='#0A2239')
        self.footer_frame.grid(row=6, column=0, sticky='ew', padx=20, pady=10)
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
            logo_label.grid(row=0, column=1, sticky='e', padx=20, pady=5)
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
        contact_label.grid(row=7, column=0, sticky='s', pady=5)

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
            

    def generate_videos(self):
        # Prompt the user for the desired version
        version = simpledialog.askstring("Select Version", "Which version would you like to use? (2b or 5b)")
        if version not in ["2b", "5b"]:
            messagebox.showerror("Invalid Selection", "Please enter either '2b' or '5b'.")
            return

        # Paths to required folders and scripts
        cogvideo_path = os.path.join(os.getcwd(), "CogVideo", "inference", "gradio_composite_demo")
        temporal_script = f"TemporalCog-{version}.py"
        cogvx_venv_path = os.path.join(cogvideo_path, "CogVx")

        try:
            # Check if CogVideo repository is cloned, if not, clone it
            if not os.path.exists(os.path.join(os.getcwd(), "CogVideo")):
                messagebox.showinfo("Cloning CogVideo Repository", "Cloning CogVideo repository into the project directory.")
                subprocess.run(["git", "clone", "https://github.com/THUDM/CogVideo"], check=True)
                messagebox.showinfo("Clone Complete", "CogVideo repository successfully cloned.")

            # Check if the virtual environment exists
            if not os.path.exists(os.path.join(cogvx_venv_path, "Scripts", "activate")):
                # Create Virtual Environment and Install Dependencies
                self.create_cogvx_environment(cogvideo_path, cogvx_venv_path)

            # Activate Environment and Run the Utility
            self.activate_and_run_utility(cogvideo_path, temporal_script)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def create_cogvx_environment(self, cogvideo_path, cogvx_venv_path):
        try:
            messagebox.showinfo("Creating Virtual Environment", "Setting up CogVx environment with Python 3.12.")

            # Ensure Python 3.12 is used for this virtual environment
            subprocess.run(["py", "-3.12", "-m", "venv", cogvx_venv_path], check=True)

            # Install the necessary dependencies for CogVideo
            pip_executable = os.path.join(cogvx_venv_path, "Scripts", "pip.exe")
            requirements_file = os.path.join(cogvideo_path, "requirements.txt")

            subprocess.run([pip_executable, "install", "-r", requirements_file], check=True)
            subprocess.run([pip_executable, "install", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "torchaudio==2.0.2+cu118", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
            subprocess.run([pip_executable, "install", "moviepy==2.0.0.dev2"], check=True)

            messagebox.showinfo("Environment Setup Complete", "CogVx environment has been successfully created and dependencies installed.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create virtual environment or install dependencies: {e}")

    def activate_and_run_utility(self, cogvideo_path, temporal_script):
        try:
            # Clear any residual memory before running the script
            import torch
            import gc

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

            # Path to the virtual environment's Python executable
            if platform.system() == "Windows":
                python_executable = os.path.join(cogvideo_path, "CogVx", "Scripts", "python.exe")
            else:
                python_executable = os.path.join(cogvideo_path, "CogVx", "bin", "python")

            # Full path to the TemporalCog script (2b or 5b)
            temporal_script_path = os.path.join(cogvideo_path, temporal_script)

            # Build the command
            command = [python_executable, temporal_script_path]

            # Run the script in a new console window and ensure output is visible
            if platform.system() == "Windows":
                # Use CREATE_NEW_CONSOLE to open a new console window
                subprocess.Popen(command, cwd=cogvideo_path, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # For Unix-like systems, run the script normally
                subprocess.Popen(command, cwd=cogvideo_path)

            messagebox.showinfo("Script Execution", f"{temporal_script} has been started successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute the script: {e}")


    def create_smart_directory_and_filenames(self, input_concept):
        """
        Creates directories and filenames based on the input concept.
        
        Args:
            input_concept (str): The concept input by the user.
        
        Returns:
            tuple: (directory, video_folder, audio_folder, video_filename, audio_filename)
        """
        # Sanitize the input concept for use in file/directory names
        sanitized_concept = re.sub(r'[^\w\s-]', '', input_concept).strip().replace(' ', '_')[:30]  # Reduced to 30 chars
        unique_id = uuid.uuid4().hex[:8]  # 8-character unique identifier
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a shorter directory name to prevent exceeding path length
        directory_name = f"{sanitized_concept}"
        directory = os.path.join(self.output_folder, directory_name)
        
        video_folder = os.path.join(directory, "Video")
        audio_folder = os.path.join(directory, "Audio")
        
        # Use shorter filenames
        video_filename = f"{sanitized_concept}_video_prompts.txt"
        audio_filename = f"{sanitized_concept}_audio_prompts.txt"
        
        # Ensure filenames are not too long
        max_filename_length = 100  # Adjust as needed
        video_filename = video_filename[:max_filename_length]
        audio_filename = audio_filename[:max_filename_length]
        
        # Create directories if they don't exist
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        
        return directory, video_folder, audio_folder, video_filename, audio_filename
    def show_video_prompt_options(self):
        self.video_options_window = tk.Toplevel(self.root)
        self.video_options_window.title("Video Prompt Options")
        self.video_options_window.configure(bg='#0A2239')

        self.build_video_options(self.video_options_window)

        
    def clean_prompt_text(self, prompt_text):
        """
        Cleans the prompt text by extracting only the positive and negative sections in the required format.
        
        Args:
            prompt_text (str): The raw prompt text from the model.
        
        Returns:
            str: The cleaned and formatted prompts.
        """
        # Split the prompt sets based on the separator
        prompt_sets = prompt_text.strip().split("--------------------")

        cleaned_prompts = []
        for prompt_set in prompt_sets:
            prompt_set = prompt_set.strip()
            if not prompt_set:
                continue

            # Match the exact format required (case-insensitive)
            prompt_match = re.search(r"^positive:\s*(.*?)\nnegative:\s*(.*)", prompt_set, re.DOTALL | re.IGNORECASE)
            if prompt_match:
                positive_section = prompt_match.group(1).strip()
                negative_section = prompt_match.group(2).strip()

                # Ensure negative_section starts with 'Avoid' and ends with a period
                negative_section = re.sub(r"^-*\s*Avoid\s*", "Avoid ", negative_section, flags=re.MULTILINE | re.IGNORECASE)
                negative_section = re.sub(r"^\s*\*", "", negative_section)  # Remove any leading asterisks

                if not negative_section.lower().startswith("avoid"):
                    negative_section = f"Avoid {negative_section}"

                if not negative_section.endswith('.'):
                    negative_section += '.'

                # Reconstruct the prompt in the exact required format
                cleaned_prompt = f"positive: {positive_section}\nnegative: {negative_section}"
                cleaned_prompts.append(cleaned_prompt)
            else:
                # If the format doesn't match, skip this prompt set or handle accordingly
                print(f"Skipping malformed prompt set:\n{prompt_set}")
                continue

        # Join the cleaned prompts with the separator
        return "\n--------------------\n".join(cleaned_prompts)

        
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

    def save_to_file(self, content, file_path):
        """
        Saves the given content to the specified file path.
        
        Args:
            content (str): The content to save.
            file_path (str): The path to the file where content will be saved.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Content saved to {file_path}")
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            messagebox.showerror("Save Error", f"Failed to save file {file_path}:\n{e}")


    def validate_prompts(self, generated_prompts, expected_count):
        """
        Validates that the number of prompt sets matches the expected count and 
        that each set contains exactly one 'positive:' section and one 'negative:' section.
        The 'negative:' section can be empty.

        Args:
            generated_prompts (str): The concatenated prompts string.
            expected_count (int): The expected number of prompt sets.

        Returns:
            bool: True if all prompt sets are valid, False otherwise.
        """
        # Split the prompts based on the separator
        prompt_sets = [p.strip() for p in generated_prompts.strip().split("--------------------") if p.strip()]
        
        # Validate the number of prompt sets
        if len(prompt_sets) != expected_count:
            print(f"Expected {expected_count} prompt sets, but got {len(prompt_sets)}.")
            return False

        for idx, prompt_set in enumerate(prompt_sets, start=1):
            prompt_set = prompt_set.strip()
            if not prompt_set:
                print(f"Prompt set {idx} is empty.")
                return False

            # Use regex to find 'positive:' and 'negative:' sections
            positive_match = re.search(r"positive:\s*(.*)", prompt_set, re.IGNORECASE | re.DOTALL)
            negative_match = re.search(r"negative:\s*(.*)", prompt_set, re.IGNORECASE | re.DOTALL)
            
            if not positive_match or negative_match is None:
                print(f"Prompt set {idx} is missing 'positive:' or 'negative:' sections.")
                return False
            
            positive_content = positive_match.group(1).strip()
            negative_content = negative_match.group(1).strip()
            
            if not positive_content:
                print(f"Prompt set {idx} has an empty 'positive:' section.")
                return False
            
            # 'negative:' section can be empty, so no need to check its content
            
        return True
    # Function to open audio prompt options
    def show_audio_prompt_options(self):
        self.audio_options_window = tk.Toplevel(self.root)
        self.audio_options_window.title("Audio Prompt Options")
        self.audio_options_window.configure(bg='#0A2239')

        self.build_audio_options(self.audio_options_window)


    def contains_year(self, text):
        """
        Checks if the provided text contains a four-digit year between 1900 and 2099.

        Args:
            text (str): The text to check for a year.

        Returns:
            bool: True if a year is found, False otherwise.
        """
        return bool(re.search(r'\b(19|20)\d{2}\b', text))
        
    def get_setting_value(self, var, options_list, random_var, custom_entry_var=None):
        """
        Retrieves the value for a setting, applying randomization if enabled.

        Args:
            var (tk.Variable): The tkinter variable associated with the dropdown.
            options_list (list): The list of predefined options for the dropdown.
            random_var (tk.BooleanVar): The variable indicating if randomization is enabled.
            custom_entry_var (tk.Variable, optional): The variable for custom user entries.

        Returns:
            str: The selected or randomized value.
        """
        if random_var.get():
            # Start with the predefined options
            choices = options_list.copy()

            # If there's a custom entry (for animals), split and add to choices
            if custom_entry_var:
                custom_text = custom_entry_var.get().strip()
                if custom_text:
                    custom_options = [item.strip() for item in custom_text.split(',') if item.strip()]
                    choices.extend(custom_options)

            if choices:
                return random.choice(choices)
            else:
                return ""  # Return empty string if no options are available
        else:
            return var.get()

    def get_randomized_setting(self, var, options_list, random_var, custom_entry_var=None):
        """
        Retrieves the value for a setting, applying randomization if enabled.

        Args:
            var (tk.Variable): The tkinter variable associated with the dropdown.
            options_list (list): The list of predefined options for the dropdown.
            random_var (tk.BooleanVar): The variable indicating if randomization is enabled.
            custom_entry_var (tk.Variable, optional): The variable for custom user entries.

        Returns:
            str: The selected or randomized value.
        """
        if random_var.get():
            # Start with the predefined options
            choices = options_list.copy()

            # If there's a custom entry (for animals), split and add to choices
            if custom_entry_var:
                custom_text = custom_entry_var.get().strip()
                if custom_text:
                    custom_options = [item.strip() for item in custom_text.split(',') if item.strip()]
                    choices.extend(custom_options)

            if choices:
                return random.choice(choices)
            else:
                # Handle empty choices gracefully
                messagebox.showwarning("Randomization Warning", f"No available options to randomize for this setting. Using selected value.")
                return var.get()
        else:
            return var.get()

    def generate_video_prompts(self):
        """
        Generate video prompts optimized for CogVideoX Prompting Standards.
        In 'Story Mode', first generate a coherent story outline and then create detailed prompts for each scene.
        In 'Non-Story Mode', generate prompts individually without any overlap.
        Incorporates best prompting practices for enhanced prompt quality.
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

        # Ensure Characters directory exists
        characters_dir = os.path.join(self.output_folder, 'Characters')
        os.makedirs(characters_dir, exist_ok=True)

        # Determine the number of prompts to generate
        num_prompts = self.video_prompt_number_var.get()

        generated_prompts = []
        accumulated_story = []  # To store generated scene prompts for context

        # Adjusted System Prompt for Story Mode
        sys_prompt_story = f"""
    Intuitively and creatively turn {input_concept} into a series of deeply descriptive, cohesive narrative paragraphs for a set of natural language video prompts. Each prompt set in the prompt list represents a prompt set within the overall story. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY

    PROMPT RULES:
    - Write in narrative paragraph form, starting with date and location. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY
    - Include specific character names in bold (e.g., **John Smith**), with detailed descriptions of their appearance, attire, and actions. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY
    - Maximize detail and space, aiming for up to 226 tokens per paragraph. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY
    - Integrate real-world physics with subtle details (e.g., debris scattering, fluid dynamics, light refraction).
    - Enhance the environment with realistic particles respecting natural laws. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY
    - Provide specific details for anatomy and avoid generalized concepts. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY
    - Reinforce keywords in different parts of the prompt for output consistency. 
    - Focus on the content that should be in the video.
    - Ensure narrative continuity and coherency between scenes.
    - Do not use bullets, lists, or any formatting other than narrative paragraphs.
    - Do not mention video durations or phrases like 'generate a video' or 'create a clip'.
    - At the end of each paragraph, include negative prompts in the following format:
        Negative: [list of negative terms]
    - Do not word the negative prompts negatively; simply list things to avoid.

    Ensure that the output includes all 7 markers (Camera Language, Framing Angle, Lighting, Subject Description, Subject Movement, Scene Description, Atmosphere) but weave them naturally into the narrative, without explicitly mentioning them or using parentheses.
    
    Each prompt should be a three sentence description maximizing the token space for conveying the most information to the model as efficiently as possible.

    """

        # Adjusted System Prompt for Non-Story Mode
        sys_prompt_non_story = f"""
    Transform {input_concept} into specifically detailed and immersive narrative paragraphs for video prompts. Always make sure content does not get censored by Llama 3.2 or triggeer and safety guards for content. AVOID VIOLENCE, SEX AND OTHER CENSORED CONTENT AT ALL COSTS WHEN CREATING PROMPTS.

    PROMPT RULES:
    - Write in narrative paragraph form, starting with date and location.
    - Include specific character names in bold (e.g., **John Smith**), with detailed descriptions of their appearance, attire, and actions.
    - Maximize detail and space, aiming for up to 226 tokens per paragraph.
    - Integrate real-world physics with subtle details (e.g., debris scattering, fluid dynamics, light refraction).
    - Enhance the environment with realistic particles respecting natural laws.
    - Provide specific details for anatomy and avoid generalized concepts.
    - Reinforce keywords in different parts of the prompt for output consistency.
    - Focus on the content that should be in the video.
    - Do not use bullets, lists, or any formatting other than narrative paragraphs.
    - Do not mention video durations or phrases like 'generate a video' or 'create a clip'.
    - At the end of each paragraph, include negative prompts in the following format:
        Negative: [list of negative terms]
    - Do not word the negative prompts negatively; simply list things to avoid.

    Ensure that the output includes all 7 markers (Camera Language, Framing Angle, Lighting, Subject Description, Subject Movement, Scene Description, Atmosphere) but weave them naturally into the narrative, without explicitly mentioning them or using parentheses.
    
    Each prompt should be a three sentence description maximizing the token space for conveying the most information to the model as efficiently as possible.


    """

        if self.video_story_mode_var.get():
            # Story Mode: Generate a story outline first
            outline_generated = False
            max_outline_retries = 7
            outline_retry_count = 0

            while not outline_generated and outline_retry_count < max_outline_retries:
                try:
                    video_options = {
                        "theme": self.get_randomized_setting(
                            self.video_theme_var, THEMES, self.video_randomize_theme_var
                        ),
                        "art_style": self.get_randomized_setting(
                            self.video_art_style_var, ART_STYLES, self.video_randomize_art_style_var
                        ),
                        "lighting": self.get_randomized_setting(
                            self.video_lighting_var, LIGHTING_OPTIONS, self.video_randomize_lighting_var
                        ),
                        "framing": self.get_randomized_setting(
                            self.video_framing_var, FRAMING_OPTIONS, self.video_randomize_framing_var
                        ),
                        "camera_movement": self.get_randomized_setting(
                            self.video_camera_movement_var, CAMERA_MOVEMENTS, self.video_randomize_camera_movement_var
                        ),
                        "shot_composition": self.get_randomized_setting(
                            self.video_shot_composition_var, SHOT_COMPOSITIONS, self.video_randomize_shot_composition_var
                        ),
                        "time_of_day": self.get_randomized_setting(
                            self.video_time_of_day_var, TIME_OF_DAY_OPTIONS, self.video_randomize_time_of_day_var
                        ),
                        "camera": self.get_randomized_setting(
                            self.video_camera_var, CAMERAS.get(self.video_decade_var.get(), []), self.video_randomize_camera_var
                        ),
                        "lens": self.get_randomized_setting(
                            self.video_lens_var, LENSES, self.video_randomize_lens_var
                        ),
                        "resolution": self.get_randomized_setting(
                            self.video_resolution_var, RESOLUTIONS.get(self.video_decade_var.get(), RESOLUTIONS[DECADES[0]]), self.video_randomize_resolution_var
                        ),
                        "wildlife_animal": self.get_randomized_setting(
                            self.wildlife_animal_var, WILDLIFE_ANIMALS, self.video_randomize_wildlife_animal_var, self.wildlife_animal_entry_var
                        ),
                        "domesticated_animal": self.get_randomized_setting(
                            self.domesticated_animal_var, DOMESTICATED_ANIMALS, self.video_randomize_domesticated_animal_var, self.domesticated_animal_entry_var
                        ),
                        "soundscape_mode": self.video_soundscape_mode_var.get(),
                        "holiday_mode": self.video_holiday_mode_var.get(),
                        "selected_holidays": self.video_holidays_var.get(),
                        "specific_modes": [mode for mode, var in self.video_specific_modes_vars.items() if var.get()],
                        "no_people_mode": self.video_no_people_mode_var.get(),
                        "chaos_mode": self.video_chaos_mode_var.get(),
                        "remix_mode": self.video_remix_mode_var.get(),
                        "decade": self.get_randomized_setting(
                            self.video_decade_var, DECADES, self.video_randomize_decade_var
                        )
                    }

                    # Build the base options context for this prompt
                    current_options_context = [
                        f"Theme: {video_options['theme']}",
                        f"Art Style: {video_options['art_style']}",
                        f"Lighting: {video_options['lighting']}",
                        f"Framing: {video_options['framing']}",
                        f"Camera Movement: {video_options['camera_movement']}",
                        f"Shot Composition: {video_options['shot_composition']}",
                        f"Time of Day: {video_options['time_of_day']}",
                        f"Camera: {video_options['camera']}, Lens: {video_options['lens']}",
                        f"Resolution: {video_options['resolution']}"
                    ]                                     
                    # Step 1: Generate a story outline with system prompt
                    outline_prompt = (
                        f"{sys_prompt_story} in {num_prompts} prompt seeds, WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY and leaned towards american consumers, unless otherwise specificed direct prompts towards american actors and settings.\n"
                        f"Based on the concept '{input_concept}', generate a professionally ordered, detailed outline for a family-friendly story divided into exactly {num_prompts} prompt seeds, no more and no less than {num_prompts}."
                        f"Provide the outline as a numbered list, with each scene on a new line. Do not include any additional text before or after the outline. WITH ANY EXPLICIT CONTENT REPHRASED TO BE FAMILY FRIENDLY. The CogVideoX Model is chinese based and leans towards asian people generations, always inject key terms to direct it towards American audiences and generations by default unless otherwise specified within '{input_concept}'\n"
                        f"Do not create a scene that might promote or glorify illegal activities. Do not promote or glorify illegal activities EVER. Each positive prompt should be a four sentence description maximizing the token space for conveying the most information to the model as efficiently as possible.\n"
                        f"IF it is older silent film era then inject extra terms to guide it towards that, If it's 60s then guide it towards panavision and technicolor, etc. Don't reiterate the {input_concept} directly here.\n"
                        f"Ensure that the outline remains cohesive and masterfully decides on and incorporates one of these three story arcs, The Coming-of-Age Arc (Bildungsroman), The Transformation Arc (Positive Change Arc), or the The Hero's Journey (Monomyth).\n"
                        
                    )

                    # Call the model to generate the outline
                    raw_outline = self.generate_prompts_via_ollama(outline_prompt, 'text', 1)

                    # Parse the outline into scenes
                    scene_descriptions = self.parse_outline(raw_outline, num_prompts)

                    if scene_descriptions and len(scene_descriptions) == num_prompts:
                        outline_generated = True
                        print("Temporal Story Outline Generation Complete.")
                    else:
                        outline_retry_count += 1
                        print(f"Temporal Story Outline still generating. Please be patient while I continue putting everything together for you... ({outline_retry_count}/{max_outline_retries})")
                except Exception as e:
                    outline_retry_count += 1
                    print(f"It looks like there has been an error your generating Temporal Story Outline: {e}. This is not common. Let me go ahead and retry that for you... ({outline_retry_count}/{max_outline_retries})")

            if not outline_generated:
                messagebox.showerror("Temporal Story Outline FAILED", "I am sorry! It looks like I've failed to generate your Temporal Story Outline after multiple attempts. Please go ahead and start it again. This is pretty rare.")
                # Fallback: Proceed without story mode
                self.video_story_mode_var.set(False)
                print("Proceeding without 'Story Mode' due to outline generation failure.")

        if self.video_story_mode_var.get():
            # Story Mode: Generate detailed prompts for each scene
            for prompt_index, scene_description in enumerate(scene_descriptions, start=1):
                retry_count = 0
                max_retries = 42  # Set a maximum number of retries

                while retry_count < max_retries:
                    try:
                        # Gather settings for this specific prompt
                        video_options = {
                            "theme": self.get_randomized_setting(
                                self.video_theme_var, THEMES, self.video_randomize_theme_var
                            ),
                            "art_style": self.get_randomized_setting(
                                self.video_art_style_var, ART_STYLES, self.video_randomize_art_style_var
                            ),
                            "lighting": self.get_randomized_setting(
                                self.video_lighting_var, LIGHTING_OPTIONS, self.video_randomize_lighting_var
                            ),
                            "framing": self.get_randomized_setting(
                                self.video_framing_var, FRAMING_OPTIONS, self.video_randomize_framing_var
                            ),
                            "camera_movement": self.get_randomized_setting(
                                self.video_camera_movement_var, CAMERA_MOVEMENTS, self.video_randomize_camera_movement_var
                            ),
                            "shot_composition": self.get_randomized_setting(
                                self.video_shot_composition_var, SHOT_COMPOSITIONS, self.video_randomize_shot_composition_var
                            ),
                            "time_of_day": self.get_randomized_setting(
                                self.video_time_of_day_var, TIME_OF_DAY_OPTIONS, self.video_randomize_time_of_day_var
                            ),
                            "camera": self.get_randomized_setting(
                                self.video_camera_var, CAMERAS.get(self.video_decade_var.get(), []), self.video_randomize_camera_var
                            ),
                            "lens": self.get_randomized_setting(
                                self.video_lens_var, LENSES, self.video_randomize_lens_var
                            ),
                            "resolution": self.get_randomized_setting(
                                self.video_resolution_var, RESOLUTIONS.get(self.video_decade_var.get(), RESOLUTIONS[DECADES[0]]), self.video_randomize_resolution_var
                            ),
                            "wildlife_animal": self.get_randomized_setting(
                                self.wildlife_animal_var, WILDLIFE_ANIMALS, self.video_randomize_wildlife_animal_var, self.wildlife_animal_entry_var
                            ),
                            "domesticated_animal": self.get_randomized_setting(
                                self.domesticated_animal_var, DOMESTICATED_ANIMALS, self.video_randomize_domesticated_animal_var, self.domesticated_animal_entry_var
                            ),
                            "soundscape_mode": self.video_soundscape_mode_var.get(),
                            "holiday_mode": self.video_holiday_mode_var.get(),
                            "selected_holidays": self.video_holidays_var.get(),
                            "specific_modes": [mode for mode, var in self.video_specific_modes_vars.items() if var.get()],
                            "no_people_mode": self.video_no_people_mode_var.get(),
                            "chaos_mode": self.video_chaos_mode_var.get(),
                            "remix_mode": self.video_remix_mode_var.get(),
                            "decade": self.get_randomized_setting(
                                self.video_decade_var, DECADES, self.video_randomize_decade_var
                            )
                        }

                        # Build the base options context for this prompt
                        current_options_context = [
                            f"Theme: {video_options['theme']}",
                            f"Art Style: {video_options['art_style']}",
                            f"Lighting: {video_options['lighting']}",
                            f"Framing: {video_options['framing']}",
                            f"Camera Movement: {video_options['camera_movement']}",
                            f"Shot Composition: {video_options['shot_composition']}",
                            f"Time of Day: {video_options['time_of_day']}",
                            f"Camera: {video_options['camera']}, Lens: {video_options['lens']}",
                            f"Resolution: {video_options['resolution']}"
                        ]

                        # Add optional elements dynamically
                        if video_options["wildlife_animal"]:
                            current_options_context.append(f"Feature a {video_options['wildlife_animal']}.")
                        
                        if video_options["domesticated_animal"]:
                            current_options_context.append(f"Include a {video_options['domesticated_animal']}.")
                        
                        if video_options["soundscape_mode"]:
                            current_options_context.append("Incorporate soundscapes relevant to the scene.")
                        
                        if video_options["holiday_mode"]:
                            current_options_context.append(f"Apply holiday themes: {video_options['selected_holidays']}.")
                        
                        if video_options["no_people_mode"]:
                            current_options_context.append("Focus on the environment or animals, without human figures.")
                        
                        if video_options["chaos_mode"]:
                            current_options_context.append("Introduce chaotic elements that create tension or contrast in the visuals.")
                        
                        if video_options["remix_mode"]:
                            current_options_context.append("Add creative variations in visual styles or thematic choices.")

                        # Prepare a summary of previous scenes
                        if prompt_index > 1:
                            # Summarize previous prompts without exceeding token limit
                            previous_scenes = "\n".join(
                                [f"Scene {i}: {self.extract_scene_summary(desc)}" for i, desc in enumerate(generated_prompts[:prompt_index-1], start=1)]
                            )
                            previous_scenes_summary = f"The story so far:\n{previous_scenes}\n\n"
                        else:
                            previous_scenes_summary = ""

                        # Construct the detailed prompt with system prompt and user instructions
                        detailed_prompt = (
                            f"Take {prompt_index}:{scene_description} and create a specific, pg-13 friendly, description focusing on the visual scene for it including all specific details and aspects of the scene. Do not focus on exposition or the soundscape, DO NOT MENTION ANY SOUNDS IN THE SCENE. You are focusing on the visual aspects scene as if describing it to a blind person. Avoid abstract or emotional descriptions. It must be leaned towards american consumers, unless otherwise specificed direct prompts towards american actors and settings. Every prompt must be set in {video_options['decade']}\n"
                            f"Always start with this exact phraseing 'Positive: Set in {video_options['decade']}, shot on a {video_options['camera']}...' DO NOT INCLUDE PROS AND CONS FOR THE CAMERA MODEL. AND SUMMARIZE IT BETTER TO FIT THE PROMPT ALWAYS.\n"
                            f" ALWAYS INCLUDE (Camera Language, Framing Angle, Lighting, Subject Description, Subject Movement, Scene Description, Atmosphere) integrated seamlessly into the narrative, traditional exposition is not required, only visual description of the 6 second visual scene is required in temporal order. \n"
                            f"- Include specific character names in bold (e.g., **John Smith**), with a detailed description of the character's appearance, attire, and actions focusing on the visual aspects.\n"
                            f"- Do not use bullets, lists, or any formatting other than narrative paragraphs. Generate exactly one Positive and one Negative for this scene. There should never be more than one set per story generation. One set = One Positive Prompt + One Negative Prompt"
                            f"- Integrate an expert awareness of real-world physics to influence the subtle environmental details.\n"
                            f"- Provide specific details and avoid generalized concepts, do not convey abstract ideas and only provide descriptions of the visual aspects of the scene starting with the main subject and ensuring all details are coherent with the {video_options['decade']} standards and expectations.\n"
                            f"- All content must be within PG-13 guidelines and always family-friendly. Nothing explicit should be considered and should be replaced with cleaner phrasing. Each prompt should be a three sentence description maximizing the token space for conveying the most information to the model as efficiently as possible.\n"
                            f"- Maximize visual description detail, aiming for up to 220 tokens and always maximizing each prompt set. Do not just do short and easy ones and avoid exposition like 'a testament to X's resourcefulness and ingenuity' or '.\n"
                            f"Generate exactly one Positive Prompt and one Negative Prompt as a Prompt Set for [idx] using FORMAT Example below:\n"
                            f"Positive: [ALWAYS START WITH THE DECADE AND CAMERA LIKE 'Positive: Set in ['decade'], shot on a ['camera']...' DO NOT INCLUDE PROS AND CONS FOR THE CAMERA MODEL. If the camera were 'PROFESSIONAL - Sony - Digital Betacam DVW-700WS (1993) - ' DO NOT PRESENT AS 'PROFESSIONAL - Sony - Digital Betacam DVW-700WS (1993) - ' ALWAYS PRESENT ANY CAMERA AS FOLLOWING 'Sony Digital Betacam DVW-700WS from 1993'. The actual camera, model and date may differ. IF it is older silent film era then inject extra terms to guide it towards that, If it's 60s then guide it towards panavision and technicolor, etc. Don't reiterate the {input_concept} directly here. This is for finalized content prompts. This positive prompt should be 5 or 6 sentences in detail and never shorter than 3 long sentences. Optimize the prompt output to a token count of 220 for {input_concept}]"
                            f"Negative: [A masterfully crafted negative prompt to compliment {prompt_index}:{scene_description}. ONLY PRESENT IN STRAIGHT-FORWARD LIST FORMAT LIKE 'Blurry background figures, misaligned or awkward features, deformed limbs, distracting backgrounds, cluttered scenes' without exposition or reasoning or explantion. Just provide an optimized list with commas between each, never use dashes or dots or anything else besides commas. YOU CAN NOT mention names, titles or any specific character information or give full english directions in any way. Do not express or write anything that should be. Make sure the list takes into account specifics to {prompt_index}:{scene_description}. Most women don't have facial hair for example, etc. DO NOT EVER PROVIDE IN BULLETED FORM.]\n"

                        )

                        # Call the model to generate the detailed video prompt
                        raw_video_prompt = self.generate_prompts_via_ollama(detailed_prompt, 'video', 1)

                        if not raw_video_prompt:
                            raise Exception(f"No video prompt generated for prompt {prompt_index}. Retrying...")

                        # Clean and format the prompt
                        cleaned_prompt = self.clean_prompt_text(raw_video_prompt)
                        formatted_prompt = self.remove_unwanted_headers(cleaned_prompt)

                        # Extract character names from the prompt
                        character_names = self.extract_character_names(formatted_prompt)

                        # Manage character profiles
                        for character in character_names:
                            # Load existing profile or create a new one
                            profile = self.load_character_profile(character, characters_dir)
                            if not profile:
                                description = self.extract_character_description(formatted_prompt, character)
                                profile = self.create_character_profile(character, characters_dir, description)
                                self.update_character_history(profile, "Character introduced.", characters_dir)
                            else:
                                # Update history with new prompt details
                                self.update_character_history(profile, formatted_prompt, characters_dir)

                        # Validate the generated prompt
                        if self.validate_prompts(formatted_prompt, 1):
                            generated_prompts.append(formatted_prompt)
                            accumulated_story.append(formatted_prompt)  # Store for context in next scenes
                            print(f"Prompt {prompt_index} generated successfully.")
                            break  # Move to the next prompt set
                        else:
                            retry_count += 1
                            print(f"Validation failed for prompt {prompt_index}. Retrying... ({retry_count}/{max_retries})")
                            time.sleep(1)  # Optional: wait before retrying
                    except KeyError as ke:
                        retry_count += 1
                        print(f"Error generating video prompt {prompt_index}: {ke}. Retrying... ({retry_count}/{max_retries})")
                        time.sleep(1)  # Optional: wait before retrying
                    except Exception as e:
                        retry_count += 1
                        print(f"Error generating video prompt {prompt_index}: {e}. Retrying... ({retry_count}/{max_retries})")
                        time.sleep(1)  # Optional: wait before retrying
                else:
                    print(f"Failed to generate a valid prompt after {max_retries} attempts for prompt {prompt_index}.")
                    messagebox.showerror("Prompt Generation Error", f"Failed to generate a valid prompt after {max_retries} attempts for prompt {prompt_index}.")
                    return  # Exit the function if unable to generate valid prompts

        else:
            # Non-Story Mode
            for prompt_index in range(1, num_prompts + 1):
                retry_count = 0
                max_retries = 12  # Set a maximum number of retries

                while retry_count < max_retries:
                    try:
                        # Gather settings for this specific prompt
                        video_options = {
                            "theme": self.get_randomized_setting(
                                self.video_theme_var, THEMES, self.video_randomize_theme_var
                            ),
                            "art_style": self.get_randomized_setting(
                                self.video_art_style_var, ART_STYLES, self.video_randomize_art_style_var
                            ),
                            "lighting": self.get_randomized_setting(
                                self.video_lighting_var, LIGHTING_OPTIONS, self.video_randomize_lighting_var
                            ),
                            "framing": self.get_randomized_setting(
                                self.video_framing_var, FRAMING_OPTIONS, self.video_randomize_framing_var
                            ),
                            "camera_movement": self.get_randomized_setting(
                                self.video_camera_movement_var, CAMERA_MOVEMENTS, self.video_randomize_camera_movement_var
                            ),
                            "shot_composition": self.get_randomized_setting(
                                self.video_shot_composition_var, SHOT_COMPOSITIONS, self.video_randomize_shot_composition_var
                            ),
                            "time_of_day": self.get_randomized_setting(
                                self.video_time_of_day_var, TIME_OF_DAY_OPTIONS, self.video_randomize_time_of_day_var
                            ),
                            "camera": self.get_randomized_setting(
                                self.video_camera_var, CAMERAS.get(self.video_decade_var.get(), []), self.video_randomize_camera_var
                            ),
                            "lens": self.get_randomized_setting(
                                self.video_lens_var, LENSES, self.video_randomize_lens_var
                            ),
                            "resolution": self.get_randomized_setting(
                                self.video_resolution_var, RESOLUTIONS.get(self.video_decade_var.get(), RESOLUTIONS[DECADES[0]]), self.video_randomize_resolution_var
                            ),
                            "decade": self.get_randomized_setting(
                                self.video_decade_var, DECADES, self.video_randomize_decade_var
                            ),
                            "wildlife_animal": self.get_randomized_setting(
                                self.wildlife_animal_var, WILDLIFE_ANIMALS, self.video_randomize_wildlife_animal_var, self.wildlife_animal_entry_var
                            ),
                            "domesticated_animal": self.get_randomized_setting(
                                self.domesticated_animal_var, DOMESTICATED_ANIMALS, self.video_randomize_domesticated_animal_var, self.domesticated_animal_entry_var
                            ),
                            "soundscape_mode": self.video_soundscape_mode_var.get(),
                            "holiday_mode": self.video_holiday_mode_var.get(),
                            "selected_holidays": self.video_holidays_var.get(),
                            "specific_modes": [mode for mode, var in self.video_specific_modes_vars.items() if var.get()],
                            "no_people_mode": self.video_no_people_mode_var.get(),
                            "chaos_mode": self.video_chaos_mode_var.get(),
                            "remix_mode": self.video_remix_mode_var.get(),
                        }

                        # Build the base options context for this prompt
                        current_options_context = [
                            f"Theme: {video_options['theme']}",
                            f"Art Style: {video_options['art_style']}",
                            f"Lighting: {video_options['lighting']}",
                            f"Framing: {video_options['framing']}",
                            f"Camera Movement: {video_options['camera_movement']}",
                            f"Shot Composition: {video_options['shot_composition']}",
                            f"Time of Day: {video_options['time_of_day']}",
                            f"Camera: {video_options['camera']}, Lens: {video_options['lens']}",
                            f"Resolution: {video_options['resolution']}",
                            f"Decade: {video_options['decade']}"
                        ]

                        # Add optional elements dynamically
                        if video_options["wildlife_animal"]:
                            current_options_context.append(f"Feature a {video_options['wildlife_animal']}.")
                        
                        if video_options["domesticated_animal"]:
                            current_options_context.append(f"Include a {video_options['domesticated_animal']}.")
                        
                        if video_options["soundscape_mode"]:
                            current_options_context.append("Incorporate soundscapes relevant to the scene.")
                        
                        if video_options["holiday_mode"]:
                            current_options_context.append(f"Apply holiday themes: {video_options['selected_holidays']}.")
                        
                        if video_options["no_people_mode"]:
                            current_options_context.append("Focus on the environment or animals, without human figures.")
                        
                        if video_options["chaos_mode"]:
                            current_options_context.append("Introduce chaotic elements that create tension or contrast in the visuals.")
                        
                        if video_options["remix_mode"]:
                            current_options_context.append("Add creative variations in visual styles or thematic choices.")

                        # Construct the detailed prompt with system prompt and user instructions
                        detailed_prompt = (
                            f"{sys_prompt_non_story}\n"
                            f"Create a detailed, visuals focused, pg-13 friendly, video prompt formatted for a COGVIDEOX VIDEO OUTPUT based on the concept '{input_concept} leaned towards american consumers, unless otherwise specificed direct prompts towards american actors and settings.'.\n"
                            f"The COGVIDEOX VIDEO OUTPUT scene should be unique, self-contained, and optimized for COGVIDEOX VIDEO OUTPUT.\n"
                            f"Mention the camera and decade naturally in the narrative. ALWAYS INCLUDE (Camera Language, Framing Angle, Lighting, Subject Description, Subject Movement, Scene Description, Atmosphere) integrated seamlessly into the narrative, traditional exposition is not required, only visual description of the 6 second visual scene is required in temporal order. \n"
                            f"- Include specific character names in bold (e.g., **John Smith**), with a detailed description of the character's appearance, attire, and actions focusing on the visual aspects.\n"
                            f"- Do not use bullets, lists, or any formatting other than narrative paragraphs. Generate exactly one Positive and one Negative for this scene. There should never be more than one set per story generation. One set = One Positive Prompt + One Negative Prompt"
                            f"- Integrate an expert awareness of real-world physics to influence the subtle environmental details.\n"
                            f"- Provide specific details and avoid generalized concepts, do not convey abstract ideas and only provide descriptions of the visual aspects of the scene starting with the main subject.\n"
                            f"- All content must be within PG-13 guidelines and always family-friendly. Nothing explicit should be considered and should be replaced with cleaner phrasing. Each prompt should be a three sentence description maximizing the token space for conveying the most information to the model as efficiently as possible.\n"
                            f"- Maximize visual description detail, aiming for up to 220 tokens and always maximizing each prompt set. Do not just do short and easy ones and avoid exposition like 'a testament to X's resourcefulness and ingenuity' or '.\n"
                            f"Always start with this exact phraseing 'Positive: Set in {video_options['decade']}, shot on a {video_options['camera']}...' DO NOT INCLUDE PROS AND CONS FOR THE CAMERA MODEL.\n"
                            f"Generate exactly one Positive Prompt and one Negative Prompt as a Prompt Set for {prompt_index} using FORMAT Example below:\n"
                            f"Positive: [ALWAYS START WITH THE DECADE AND CAMERA LIKE 'Positive: Set in {video_options['decade']}, shot on a {video_options['camera']}...' DO NOT INCLUDE PROS AND CONS FOR THE CAMERA MODEL. If the camera were 'PROFESSIONAL - Sony - Digital Betacam DVW-700WS (1993) - ' DO NOT PRESENT AS 'PROFESSIONAL - Sony - Digital Betacam DVW-700WS (1993) - ' ALWAYS PRESENT ANY CAMERA AS FOLLOWING 'Sony Digital Betacam DVW-700WS from 1993'. The actual camera, model and date may differ. IF it is older silent film era then inject extra terms to guide it towards that, If it's 60s then guide it towards panavision and technicolor, etc. Don't reiterate the {input_concept} directly here. This is for finalized content prompts. This positive prompt should be 5 or 6 sentences in detail and never shorter than 3 long sentences. Optimize the prompt output to a token count of 220 for {input_concept}]"
                            f"Negative: [A masterfully crafted negative prompt to compliment {input_concept}. ONLY PRESENT IN STRAIGHT-FORWARD LIST FORMAT LIKE 'Blurry background figures, misaligned or awkward features, deformed limbs, distracting backgrounds, cluttered scenes' without exposition or reasoning or explantion. Just provide an optimized list. YOU CAN NOT mention names, titles or any specific character information or give full english directions in any way. Do not express or write anything that should be. Make sure the list takes into account specifics to {input_concept}. Most women don't have facial hair for example, etc. DO NOT EVER PROVIDE IN BULLETED FORM.]\n"

                        )

                        # Call the model to generate the detailed video prompt
                        raw_video_prompt = self.generate_prompts_via_ollama(detailed_prompt, 'video', 1)

                        if not raw_video_prompt:
                            raise Exception(f"No video prompt generated for prompt {prompt_index}. Retrying...")

                        # Clean and format the prompt
                        cleaned_prompt = self.clean_prompt_text(raw_video_prompt)
                        formatted_prompt = self.remove_unwanted_headers(cleaned_prompt)

                        # Extract character names from the prompt
                        character_names = self.extract_character_names(formatted_prompt)

                        # Manage character profiles
                        for character in character_names:
                            # Load existing profile or create a new one
                            profile = self.load_character_profile(character, characters_dir)
                            if not profile:
                                description = self.extract_character_description(formatted_prompt, character)
                                profile = self.create_character_profile(character, characters_dir, description)
                                self.update_character_history(profile, "Character introduced.", characters_dir)
                            else:
                                # Update history with new prompt details
                                self.update_character_history(profile, formatted_prompt, characters_dir)

                        # Validate the generated prompt
                        if self.validate_prompts(formatted_prompt, 1):
                            generated_prompts.append(formatted_prompt)
                            accumulated_story.append(formatted_prompt)  # Store for context in next scenes
                            print(f"Prompt {prompt_index} generated successfully.")
                            break  # Move to the next prompt set
                        else:
                            retry_count += 1
                            print(f"Validation failed for prompt {prompt_index}. Retrying... ({retry_count}/{max_retries})")
                            time.sleep(1)  # Optional: wait before retrying
                    except KeyError as ke:
                        retry_count += 1
                        print(f"Error generating video prompt {prompt_index}: {ke}. Retrying... ({retry_count}/{max_retries})")
                        time.sleep(1)  # Optional: wait before retrying
                    except Exception as e:
                        retry_count += 1
                        print(f"Error generating video prompt {prompt_index}: {e}. Retrying... ({retry_count}/{max_retries})")
                        time.sleep(1)  # Optional: wait before retrying
                else:
                    print(f"Failed to generate a valid prompt after {max_retries} attempts for prompt {prompt_index}.")
                    messagebox.showerror("Prompt Generation Error", f"Failed to generate a valid prompt after {max_retries} attempts for prompt {prompt_index}.")
                    return  # Exit the function if unable to generate valid prompts

        # After generating all prompts, save them
        try:
            directory, video_folder, audio_folder, video_filename, _ = self.create_smart_directory_and_filenames(input_concept)
            video_save_path = os.path.join(video_folder, video_filename)

            # Combine all prompts into a single string separated by the delimiter
            formatted_prompts = "\n--------------------\n".join(generated_prompts)

            self.save_to_file(formatted_prompts, video_save_path)

            # Initialize both save folders
            self.video_save_folder = video_folder
            self.audio_save_folder = audio_folder  # Ensure this is also set

            # Store the prompts in the class-level attribute `self.video_prompts`
            self.video_prompts = formatted_prompts  # Store for later use

            # Enable the button to generate audio prompts
            self.enable_button(self.generate_audio_prompts_button)

            # Display formatted prompts in the output text box
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Generated Video Prompts:\n\n" + formatted_prompts)

            # Optionally, log the save paths for verification
            print(f"Video prompts saved to: {video_save_path}")
            print(f"Audio prompts will be saved to: {self.audio_save_folder}")

        except Exception as e:
            messagebox.showerror("Prompt Generation Error", f"Failed to save video prompts: {e}")
            print(f"Error saving video prompts: {e}")

    def extract_character_names(self, prompt):
        """
        Extract character names from the prompt.
        Assumes that character names are in bold, e.g., **John Smith**
        """
        return re.findall(r'\*\*(.*?)\*\*', prompt)

    def extract_character_description(self, prompt, character_name):
        """
        Extract the description of the character from the prompt.
        Assumes that the description follows the character name.
        """
        pattern = re.escape(f"**{character_name}**") + r",\s*(.*?)\."
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).strip()
        return ""

    def extract_scene_summary(self, prompt):
        """
        Extract a concise summary of a scene from the generated prompt.
        This function should parse the narrative paragraph and extract key elements to summarize the scene.
        Implement this based on the structure of your prompts.
        """
        # Example implementation: Extract the first sentence or key details
        # This needs to be tailored to your prompt structure
        try:
            # Split the prompt into sentences
            sentences = prompt.split('. ')
            # Return the first sentence as a summary
            return sentences[0] + '.'
        except Exception as e:
            print(f"Error extracting scene summary: {e}")
            return ""

    def ensure_model_available(self, model_name):
        try:
            # List available models
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if model_name not in result.stdout:
                print(f"Model '{model_name}' is not available locally. Pulling model...")
                subprocess.run(['ollama', 'pull', model_name], check=True)
            else:
                print(f"Model '{model_name}' is already available.")
        except subprocess.CalledProcessError as e:
            print(f"Error ensuring model availability: {e}")
            messagebox.showerror("Ollama Model Error", f"Failed to ensure model '{model_name}' is available.\nError: {e}")

            
    def ensure_prompt_count_update(self):
        """
        Forces the prompt count to update at least once to avoid blank generations.
        """
        self.video_prompt_number_var.set(DEFAULT_PROMPTS + 1)  # Increment once
        self.video_prompt_number_var.set(DEFAULT_PROMPTS)  # Reset to the original value

    def ask_for_prompt_list(self):
        """
        Asks the user if they want to use the current session's video prompts or load a prompt list from a file.
        Returns the prompts as a string.
        """
        response = messagebox.askyesno(
            "Load Video Prompt List",
            "Do you want to use the current session's video prompts?\nClick 'No' to load a prompt list from a file."
        )
        if response:
            # Use current session's video prompts
            return self.video_prompts
        else:
            # Open file dialog to select a prompt list file
            file_path = filedialog.askopenfilename(
                title="Select Video Prompt List File",
                filetypes=[("Text Files", "*.txt")]
            )
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompts = f.read()
                return prompts
            else:
                # User cancelled the file dialog
                return None


    def generate_audio_prompts(self):
        """
        Generate audio prompts based on video prompts.
        """
        # Ask the user if they want to use the current session's video prompts or load an existing list
        video_prompts, prompt_file_path = self.ask_for_prompt_list()
        if not video_prompts:
            # User cancelled or no prompts available
            return

        try:
            # Split the video prompts into individual prompts using the separator
            video_prompt_sets = [p.strip() for p in video_prompts.strip().split('--------------------') if p.strip()]
            print(f"Number of video prompts: {len(video_prompt_sets)}")

            sonic_descriptions = []

            for i, video_prompt_set in enumerate(video_prompt_sets, start=1):
                # Extract 'positive:' section using regular expressions
                positive_prompt = ''
                positive_match = re.search(r'positive:\s*(.*?)\s*(?=negative:|$)', video_prompt_set, re.DOTALL | re.IGNORECASE)
                if positive_match:
                    positive_prompt = positive_match.group(1).strip()

                # Ensure positive prompt is available
                if not positive_prompt:
                    print(f"No 'positive:' section found in prompt {i}. Skipping this prompt.")
                    continue  # Skip to the next prompt

                # Build the prompt to send to the language model, instructing it to not include negative prompts
                sound_prompt_template = (
                    f"Based on {positive_prompt}, list specific sounds that would make up the soundscape for the scene. Never use full sentence structure or reference characters or names. Include specific details like the camera model, diagetic sounds known to the region being represented, etc. Seperate each sound with a comma and only a comma. ABSOLUTELY first with the camera itself and then move outwards as you build the sonic landscape prompt set. You are only describing things that make noise and can be heard in two or three words. You are not referencing characters or story moments or details. You are not describing abstract ideas or conveying any visual information. Never use terms like whine, hiss or other harmonic resonant type things unless EXPLICTLY called for in the prompt exactly. You are creating absolutely diagetic soundscapes and avoiding anything to indicate musical tones unless specifically asked. Do not describe scents, light, colors or non-sonic aspects of the scene here in any form. Do not give any character or story information here. You are crafting the perfect diagetic soundscape for {positive_prompt}"
                    f"Focus solely on listing positive sounds without any negative descriptions, labels, separators, or explanations. "
                    f"Provide only the sounds separated by commas. The output should have two sections: 'positive:' followed by the list of sounds, no repeats or similiar sounds of any kind, aim for 4 to 9, followed by 'negative:' with no content or comments of any kind Negative prompt should ALWAYS BE BLANK FOR SOUNDSCAPES.\n\n"

                )

                retry_attempt = 0
                max_retries = 3  # Maximum number of retries per audio prompt
                success = False

                while retry_attempt < max_retries and not success:
                    try:
                        print(f"Attempting to generate audio prompt {i}, Retry {retry_attempt + 1}/{max_retries}")

                        # Send the sound prompt to your language model to generate the sonic description
                        translated_sonic_prompt = self.generate_prompts_via_ollama(sound_prompt_template, 'audio', 1)

                        if not translated_sonic_prompt:
                            raise Exception("No sonic description generated. Retrying...")

                        # Log the raw response for debugging
                        print(f"Raw response for audio prompt {i}:\n{translated_sonic_prompt}")

                        # Parse the model's response to extract positive sounds and set negative empty
                        formatted_sonic_prompt = parse_model_response(translated_sonic_prompt)
                        if not formatted_sonic_prompt.startswith("positive:"):
                            raise Exception("Formatted sonic prompt is missing 'positive:' section.")

                        # Validate the generated audio prompt
                        if self.validate_prompts(formatted_sonic_prompt, 1):
                            sonic_descriptions.append(formatted_sonic_prompt)
                            print(f"Audio prompt {i} generated successfully.")
                            success = True
                        else:
                            retry_attempt += 1
                            print(f"Validation failed for audio prompt {i}. Retrying... ({retry_attempt}/{max_retries})")
                    except Exception as e:
                        retry_attempt += 1
                        print(f"Error generating audio prompt {i}: {e}. Retrying... ({retry_attempt}/{max_retries})")

                if not success:
                    print(f"Failed to generate a valid audio prompt after {max_retries} attempts for prompt {i}.")
                    messagebox.showerror(
                        "Audio Prompt Generation Error",
                        f"Failed to generate a valid audio prompt after {max_retries} attempts for prompt {i}."
                    )
                    # Optionally, continue with the next prompt instead of stopping
                    continue

            if not sonic_descriptions:
                messagebox.showerror("No Audio Prompts Generated", "No valid audio prompts were generated.")
                return

            # Join the sonic descriptions together with the same format as video prompts
            formatted_sonic_prompts = "\n--------------------\n".join(sonic_descriptions)

            # Determine the directory to save audio prompts
            if prompt_file_path:
                # Use the root directory from the loaded video prompt file
                video_prompt_dir = os.path.dirname(prompt_file_path)
                root_dir = os.path.dirname(video_prompt_dir)  # Go up one level
                video_prompt_filename = os.path.basename(prompt_file_path)
            elif hasattr(self, 'video_prompt_save_path') and self.video_prompt_save_path:
                # Use the root directory from the current session's video prompts
                video_prompt_dir = os.path.dirname(self.video_prompt_save_path)
                root_dir = os.path.dirname(video_prompt_dir)
                video_prompt_filename = os.path.basename(self.video_prompt_save_path)
            else:
                # Default to a standard location or ask the user
                default_root = os.path.join(os.getcwd(), 'TemporalPromptEngineOutputs')
                root_dir = default_root
                if not os.path.exists(root_dir):
                    os.makedirs(root_dir, exist_ok=True)
                video_prompt_filename = 'video_prompts.txt'  # Default name if not available

            # Debug statements to verify paths
            print(f"Prompt File Path: {prompt_file_path}")
            print(f"Video Prompt Directory: {video_prompt_dir}")
            print(f"Root Directory: {root_dir}")

            # Create the Audio directory under the root directory
            audio_folder = os.path.join(root_dir, 'Audio')
            if not os.path.exists(audio_folder):
                os.makedirs(audio_folder, exist_ok=True)

            # Extract the base name without '_video_prompts'
            video_prompt_base_name = os.path.splitext(video_prompt_filename)[0].replace('_video_prompts', '')

            # Define the audio prompt filename to match the desired format
            audio_filename = f"{video_prompt_base_name}_audio_prompts.txt"

            self.audio_save_folder = audio_folder
            audio_save_path = os.path.join(self.audio_save_folder, audio_filename)
            self.save_to_file(formatted_sonic_prompts, audio_save_path)

            # Store the audio prompts for later use
            self.audio_prompts = formatted_sonic_prompts
            self.enable_button(self.generate_sound_button)

            # Display the formatted audio prompts
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Generated Audio Prompts:\n\n" + formatted_sonic_prompts)

            print(f"Data successfully saved to {audio_save_path}")

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
            # Ensure that 'huggingface_api_token' exists in settings
            with open(SETTINGS_FILE, 'r') as f:
                try:
                    settings = json.load(f)
                except json.JSONDecodeError:
                    settings = {}
            
            if "huggingface_api_token" not in settings:
                settings["huggingface_api_token"] = ""
                with open(SETTINGS_FILE, 'w') as f:
                    json.dump(settings, f, indent=4)
                print("Added 'huggingface_api_token' to settings.json.")
            else:
                print("settings.json already contains 'huggingface_api_token'.")
            
    def select_video_file(self, frame, index):
        """
        Allows the user to select a video file for a specific video prompt.

        Args:
            frame (ttk.LabelFrame): The frame containing the video prompt.
            index (int): The index of the video prompt.
        """
        file_path = filedialog.askopenfilename(
            title=f"Select Video File for Prompt Set {index}",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            frame.video_file_var.set(file_path)
            
    from playsound import playsound

    def play_audio(self, audio_prompt):
        """
        Plays the given audio prompt. Assumes that audio prompts are stored as MP3 files in the audio folder.

        Args:
            audio_prompt (str): The audio prompt text to be played.
        """
        try:
            # Map the audio prompt text to its corresponding MP3 file.
            # This assumes that audio files are named uniquely based on the prompt.
            # Adjust this mapping based on your actual audio file naming convention.
            sanitized_prompt = re.sub(r'[^\w\s-]', '', audio_prompt).strip().replace(' ', '_')[:50]
            audio_filename = f"{sanitized_prompt}.mp3"
            audio_file_path = os.path.join(self.audio_save_folder, audio_filename)

            if os.path.exists(audio_file_path):
                playsound(audio_file_path)
            else:
                messagebox.showerror("Audio Playback Error", f"Audio file not found: {audio_file_path}")
                print(f"Audio file not found: {audio_file_path}")
        except Exception as e:
            messagebox.showerror("Audio Playback Error", f"Failed to play audio prompt: {e}")
            print(f"Error playing audio: {e}")

    def associate_media(self):
        """
        Opens a popup window to associate video files with their respective audio prompts.
        Allows users to preview and select audio prompts for each video prompt set.
        """
        if not self.video_prompts or not self.audio_prompts:
            messagebox.showerror("Media Association Error", "Please generate both video and audio prompts before associating media.")
            return

        # Create the popup window
        popup = tk.Toplevel(self.root)
        popup.title("Associate Media")
        popup.geometry("900x700")
        popup.grab_set()  # Make the popup modal

        # Instructions
        instructions = tk.Label(
            popup,
            text="Associate each video prompt with its corresponding video file and select an audio prompt variant.",
            wraplength=850,
            justify="left",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        instructions.pack(pady=10)

        # Create a frame with a scrollbar to hold all associations
        canvas = tk.Canvas(popup, bg='#0A2239')
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # List to store user selections
        self.media_associations = []

        # Iterate over each video prompt set
        for idx, (video_prompt, audio_prompts) in enumerate(zip(
            [p.strip() for p in self.video_prompts.split("--------------------") if p.strip()],
            self.audio_prompts
        ), start=1):
            # Frame for each association
            frame = ttk.LabelFrame(scrollable_frame, text=f"Prompt Set {idx}", padding=10)
            frame.pack(fill="x", padx=10, pady=5)

            # Display the video prompt
            prompt_label = tk.Label(
                frame,
                text=f"Prompt:\n{video_prompt}",
                wraplength=800,
                justify="left",
                bg='#0A2239',
                fg='white',
                font=('Helvetica', 10)
            )
            prompt_label.pack(anchor='w')

            # Select Video File
            select_video_btn = tk.Button(
                frame,
                text="Select Video File",
                command=lambda f=frame, i=idx: self.select_video_file(f, i),
                bg="#007bff",
                fg='white',
                font=('Helvetica', 10, 'bold'),
                cursor="hand2"
            )
            select_video_btn.pack(anchor='w', pady=5)

            # Label to show selected video file path
            video_file_var = tk.StringVar()
            video_file_label = tk.Label(
                frame,
                textvariable=video_file_var,
                wraplength=800,
                justify="left",
                bg='#0A2239',
                fg='light gray',
                font=('Helvetica', 10, 'italic')
            )
            video_file_label.pack(anchor='w')

            # Audio Prompt Selection
            audio_selection_label = tk.Label(
                frame,
                text="Select Audio Prompt Variant:",
                bg='#0A2239',
                fg='white',
                font=('Helvetica', 10, 'bold')
            )
            audio_selection_label.pack(anchor='w', pady=(10, 0))

            audio_prompts_frame = ttk.Frame(frame)
            audio_prompts_frame.pack(fill="x", padx=10, pady=5)

            # Variable to store selected audio index
            selected_audio_var = tk.IntVar()
            selected_audio_var.set(-1)  # No selection by default

            for a_idx, audio_prompt in enumerate(audio_prompts, start=1):
                # Frame for each audio variant
                audio_frame = tk.Frame(audio_prompts_frame, bg='#0A2239')
                audio_frame.pack(fill="x", padx=5, pady=2)

                # Radio button for selection
                radio_btn = tk.Radiobutton(
                    audio_frame,
                    text=f"Variant {a_idx}",
                    variable=selected_audio_var,
                    value=a_idx-1,  # Zero-based index
                    bg='#0A2239',
                    fg='white',
                    selectcolor='#0A2239',
                    activebackground='#0A2239',
                    activeforeground='white'
                )
                radio_btn.pack(side="left")

                # Button to play audio (implements playback)
                play_btn = tk.Button(
                    audio_frame,
                    text="Play",
                    command=lambda ap=audio_prompt: self.play_audio(ap),
                    bg="#28a745",
                    fg='white',
                    font=('Helvetica', 8, 'bold'),
                    cursor="hand2"
                )
                play_btn.pack(side="left", padx=5)

                # Display the audio prompt text
                audio_text = tk.Label(
                    audio_frame,
                    text=f"{audio_prompt}",
                    wraplength=600,
                    justify="left",
                    bg='#0A2239',
                    fg='light gray',
                    font=('Helvetica', 10)
                )
                audio_text.pack(side="left", padx=5)

            # Store the selection variables in the frame for later access
            frame.audio_var = selected_audio_var
            frame.video_file_var = video_file_var

        # Save Associations Button
        save_btn = tk.Button(
            popup,
            text="Save Associations",
            command=lambda: self.save_media_associations(popup),
            bg="#28a745",
            fg='white',
            font=('Helvetica', 12, 'bold'),
            cursor="hand2",
            width=20,
            height=2
        )
        save_btn.pack(pady=10)


    def select_video_file(self, frame, index):
        """
        Allows the user to select a video file for a specific video prompt.

        Args:
            frame (ttk.LabelFrame): The frame containing the video prompt.
            index (int): The index of the video prompt.
        """
        file_path = filedialog.askopenfilename(
            title=f"Select Video File for Prompt {index}",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            frame.video_file_var.set(file_path)

    def play_audio(self, audio_prompt):
        """
        Plays the given audio prompt. Placeholder implementation.
        You need to implement actual audio playback based on your environment and requirements.

        Args:
            audio_prompt (str): The audio prompt text to be played.
        """
        # Placeholder: Print a message or implement actual audio playback if audio files are available
        print(f"Playing audio prompt: {audio_prompt}")
        # Example using simple text-to-speech (optional, requires additional packages)
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(audio_prompt)
        # engine.runAndWait()

    def save_media_associations(self, popup):
        """
        Saves the user's media associations and closes the popup.

        Args:
            popup (tk.Toplevel): The popup window.
        """
        associations = []
        for idx, (video_prompt, audio_prompts) in enumerate(zip(
            [p.strip() for p in self.video_prompts.split("--------------------") if p.strip()],
            self.audio_prompts
        ), start=1):
            frame = self.get_frame_by_index(popup, idx)
            if not frame:
                continue

            video_file = frame.video_file_var.get()
            selected_audio_idx = frame.audio_var.get()

            if not video_file:
                messagebox.showwarning("Association Incomplete", f"No video file selected for Prompt Set {idx}. Skipping.")
                continue

            if selected_audio_idx == -1 or selected_audio_idx >= len(audio_prompts):
                messagebox.showwarning("Association Incomplete", f"No audio prompt selected for Prompt Set {idx}. Skipping.")
                continue

            selected_audio_prompt = audio_prompts[selected_audio_idx]

            # Assume that audio files are named based on sanitized prompts
            sanitized_audio_prompt = re.sub(r'[^\w\s-]', '', selected_audio_prompt).strip().replace(' ', '_')[:50]
            audio_filename = f"{sanitized_audio_prompt}.mp3"
            audio_file_path = os.path.join(self.audio_save_folder, audio_filename)

            if not os.path.exists(audio_file_path):
                messagebox.showwarning("Audio File Missing", f"Audio file not found for Prompt Set {idx}: {audio_file_path}. Skipping.")
                continue

            associations.append({
                "video_prompt": video_prompt,
                "video_file": video_file,
                "audio_file": audio_file_path
            })

        if not associations:
            messagebox.showwarning("No Associations", "No valid media associations were saved.")
            popup.destroy()
            return

        self.media_associations = associations
        messagebox.showinfo("Associations Saved", "Media associations have been saved successfully.")
        popup.destroy()


    def get_frame_by_index(self, popup, index):
        """
        Retrieves the frame associated with a given video prompt index within the popup.

        Args:
            popup (tk.Toplevel): The popup window.
            index (int): The index of the video prompt.

        Returns:
            ttk.LabelFrame or None: The corresponding frame or None if not found.
        """
        # Iterate through the children of the scrollable_frame to find the matching frame
        for child in popup.winfo_children():
            if isinstance(child, tk.Canvas):
                canvas_children = child.winfo_children()
                for canvas_child in canvas_children:
                    if isinstance(canvas_child, ttk.Frame):
                        for label_frame in canvas_child.winfo_children():
                            if isinstance(label_frame, ttk.LabelFrame) and label_frame.cget("text") == f"Prompt Set {index}":
                                return label_frame
        return None


    def play_audio(self, audio_prompt):
        """
        Plays the given audio prompt. Placeholder implementation.
        You need to implement actual audio playback based on your environment and requirements.

        Args:
            audio_prompt (str): The audio prompt text to be played.
        """
        # Placeholder: Print a message or implement actual audio playback if audio files are available
        print(f"Playing audio prompt: {audio_prompt}")
        # Example using simple text-to-speech (optional, requires additional packages)
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(audio_prompt)
        # engine.runAndWait()


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

    def enable_button(self, button):
        """
        Enables the given button widget.
        
        Args:
            button (tk.Button): The button widget to enable.
        """
        button.config(state=tk.NORMAL)

    def disable_button(self, button):
        """
        Disables the given button widget.
        """
        button.config(state=tk.DISABLED)
    
    
    def generate_prompts_via_ollama(self, input_concept, prompt_type, number_of_prompts, options=None):
        system_prompt = f"""
        You are an AI assistant tasked with generating a single, detailed and intuitive set of prompts, one positive and one matching negative prompt set for {prompt_type} generation models. Each prompt must strictly follow the format below, with no additional information or explanation:

        Example format:

        positive: Describe the positive aspects of the scene or shot in masterful {prompt_type} detail including specific features.
        negative: Describe what to avoid in the scene or shot in detail to maintain consistent coherency.
        --------------------

        Generate a single set of prompts, one positive and one complimentary negative, of {prompt_type} prompts based on the following concept: '{input_concept}'. Ensure that each prompt set strictly follows the Correct Examples with both positive and negative format.
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

    def clean_prompt_text(self, prompt_text):
        """
        Cleans the prompt text by extracting only the positive and negative sections in the required format.
        
        Args:
            prompt_text (str): The raw prompt text from the model.
        
        Returns:
            str: The cleaned and formatted prompt.
        """
        # Remove any unwanted prefixes or suffixes
        cleaned_text = prompt_text.strip()

        # Match the exact format required
        prompt_match = re.search(r"^positive:\s*(.*?)\nnegative:\s*(.*)", cleaned_text, re.DOTALL | re.IGNORECASE)
        if prompt_match:
            positive_section = prompt_match.group(1).strip()
            negative_section = prompt_match.group(2).strip()

            # Ensure negative_section is in sentence form
            # Remove any bullet points or list formatting
            negative_section = re.sub(r"^-*\s*Avoid\s*", "Avoid ", negative_section, flags=re.MULTILINE | re.IGNORECASE)
            negative_section = re.sub(r"^\s*\*", "", negative_section)  # Remove any leading asterisks

            # Ensure negative_section ends with a period
            if not negative_section.endswith('.'):
                negative_section += '.'

            # Reconstruct the prompt in the exact required format
            cleaned_text = f"positive: {positive_section}\nnegative: {negative_section}"
        else:
            # If the format doesn't match, return an empty string to trigger a retry
            cleaned_text = ""

        return cleaned_text

    def clean_audio_prompt_text(self, prompt_text):
        """
        Cleans the audio prompt text by extracting only the list of sounds in sentence form.
        
        Args:
            prompt_text (str): The raw audio prompt text from the model.
        
        Returns:
            str: The cleaned and formatted audio prompt.
        """
        # Remove any unwanted prefixes or suffixes
        cleaned_text = prompt_text.strip()

        # Remove any extra text before or after the list
        # Assuming the model returns only the list, but to be safe:
        cleaned_text = re.sub(r'^.*?(?=[\w\s,]+$)', '', cleaned_text, flags=re.DOTALL).strip()
        cleaned_text = re.sub(r'[^a-zA-Z0-9,\s]', '', cleaned_text)

        # Ensure it's a comma-separated list
        cleaned_text = ', '.join([s.strip() for s in cleaned_text.split(',') if s.strip()])

        return cleaned_text

    def parse_outline(self, raw_outline, num_prompts):
        """
        Parses the outline returned by the model into a list of scene descriptions.

        Args:
            raw_outline (str): The raw outline text from the model.
            num_prompts (int): The expected number of scenes.

        Returns:
            list: A list of scene descriptions, or None if parsing fails.
        """
        scene_descriptions = []
        lines = raw_outline.strip().split('\n')
        for line in lines:
            # Match lines that start with a number followed by a period or parenthesis
            match = re.match(r'^\s*(\d+)\s*[\.\)]\s*(.*)', line)
            if match:
                scene_number = int(match.group(1))
                scene_description = match.group(2).strip()
                if scene_description:
                    scene_descriptions.append(scene_description)
        if len(scene_descriptions) != num_prompts:
            print(f"Expected {num_prompts} scenes in the outline, but got {len(scene_descriptions)}.")
            return None
        return scene_descriptions
        
    def sanitize_filename(self, name):
        """
        Sanitize the character name to create a valid filename.
        """
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    def load_character_profile(self, character_name, characters_dir):
        """
        Load an existing character profile if it exists.
        """
        filename = self.sanitize_filename(character_name) + '.json'
        filepath = os.path.join(characters_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def create_character_profile(self, character_name, characters_dir, description=''):
        """
        Create a new character profile.
        """
        profile = {
            'name': character_name,
            'description': description,
            'history': []
        }
        filename = self.sanitize_filename(character_name) + '.json'
        filepath = os.path.join(characters_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=4)
        return profile

    def update_character_history(self, character_profile, new_entry, characters_dir):
        """
        Update the character's history and save the profile.
        """
        character_profile['history'].append(new_entry)
        filename = self.sanitize_filename(character_profile['name']) + '.json'
        filepath = os.path.join(characters_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(character_profile, f, indent=4)

    def extract_character_names(self, prompt):
        """
        Extract character names from the prompt.
        Assumes that character names are in bold, e.g., **John Smith**
        """
        return re.findall(r'\*\*(.*?)\*\*', prompt)

    def extract_character_description(self, prompt, character_name):
        """
        Extract the description of the character from the prompt.
        Assumes that the description follows the character name.
        """
        pattern = re.escape(f"**{character_name}**") + r",\s*(.*?)(?:\.|$)"
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).strip()
        return ""


            
    def parse_raw_response(self, raw_data):
        """
        Parses the raw JSON response from the API and extracts the 'response' field which contains the raw prompts.

        Args:
            raw_data (str): The raw JSON response from the API.

        Returns:
            str: The raw prompts extracted from the 'response' field.
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


    def ask_for_prompt_list(self, prompt_type='video'):
        """
        Opens a file dialog to select a prompt list based on the prompt type.

        :param prompt_type: 'audio' or 'video' to specify the type of prompt list.
        :return: Tuple containing the prompts as a string and the file path.
        """
        if prompt_type == 'audio':
            initial_dir = os.path.join("C:\\", "Users", "Shadow", "Documents", "TemporalPromptEngineOutputs", "Mythical_creatures_as_two_year", "Audio")
            title = "Select Audio Prompt List"
        else:
            initial_dir = os.path.join("C:\\", "Users", "Shadow", "Documents", "TemporalPromptEngineOutputs", "Mythical_creatures_as_two_year", "Video")
            title = "Select Video Prompt List"

        prompt_file_path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=[("Text Files", "*.txt")]
        )
        if not prompt_file_path:
            return None, None
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                prompts = file.read()
            print(f"Loaded prompts from: {prompt_file_path}")
            return prompts, prompt_file_path
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load prompt list:\n{e}")
            print(f"Failed to load prompt list from {prompt_file_path}: {e}")
            return None, None



    def sanitize_filename(self, sound_text):
        """
        Sanitize the sound_text to create a valid filename.
        Removes or replaces invalid characters, including newlines.
        """
        # Replace newline characters with a space
        sound_text = sound_text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove all characters except word characters, spaces, and hyphens
        sanitized = re.sub(r'[^\w\s-]', '', sound_text)
        
        # Replace spaces with underscores
        sanitized = sanitized.strip().replace(' ', '_')
        
        # Truncate to 50 characters to prevent excessively long filenames
        sanitized = sanitized[:50]
        
        # If the sanitized string is empty, assign a default name
        if not sanitized:
            sanitized = "sound"
        
        return sanitized

    def generate_sound_effects(self):
        logging.info("generate_sound_effects called")
        
        # Unpack the tuple returned by ask_for_prompt_list() with prompt_type='audio'
        prompts, prompt_file_path = self.ask_for_prompt_list(prompt_type='audio')
        if not prompts:
            messagebox.showwarning("Prompt Error", "No prompts available for sound effect generation.")
            return

        # Set the audio_save_folder to the directory of prompt_file_path
        audio_save_folder = os.path.dirname(prompt_file_path)
        self.audio_save_folder = audio_save_folder

        duration = self.audio_length_var.get()
        if not duration:
            messagebox.showwarning("Duration Error", "Please provide a valid duration.")
            return

        try:
            duration = float(duration)
        except ValueError:
            messagebox.showwarning("Duration Error", "Duration must be a number.")
            return

        self.duration = duration
        logging.info(f"Audio Duration: {self.duration} seconds")

        if not audio_save_folder or not os.path.exists(audio_save_folder):
            messagebox.showerror("Invalid Audio Save Folder", "The audio save folder is invalid or does not exist.")
            logging.error("Invalid audio save folder path.")
            return

        os.makedirs(audio_save_folder, exist_ok=True)
        logging.info(f"Audio save folder: {audio_save_folder}")

        prompt_sets = [re.sub(r'negative:\s*', '', p.strip()) for p in prompts.strip().split('--------------------') if p.strip()]
        logging.info(f"Number of audio prompt sets to process: {len(prompt_sets)}")

        try:
            inference_steps = int(self.audio_inference_steps_var.get())
        except ValueError:
            messagebox.showwarning("Inference Steps Error", "Inference steps must be an integer.")
            return

        try:
            seed = int(self.audio_seed_var.get())
        except ValueError:
            messagebox.showwarning("Seed Error", "Seed must be an integer.")
            return

        # Include StepsX and CfgX in filenames
        steps_str = f"Steps{inference_steps}_"
        cfg_str = f"Cfg{inference_steps}_"  # Assuming cfg is related to inference_steps

        prompt_sets_sounds = []
        for i, prompt_set in enumerate(prompt_sets, start=1):
            sounds = [sound.strip() for sound in prompt_set.split(',') if sound.strip()]
            prompt_sets_sounds.append(sounds)
            logging.info(f"Prompt set {i}: {len(sounds)} sounds extracted.")

        total_sounds = sum(len(sounds) for sounds in prompt_sets_sounds)
        logging.info(f"Total number of individual sounds to generate: {total_sounds}")

        number_of_prompts = total_sounds
        logging.info(f"Generating {number_of_prompts} unique sound effects.")

        self.ensure_audioldm2_installed()

        try:
            repo_id = "cvssp/audioldm2-large"
            pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
            device = "cuda" if self.detect_gpu() else "cpu"
            pipe = pipe.to(device)
            logging.info(f"AudioLDM2 pipeline loaded on {device}.")
        except Exception as e:
            messagebox.showerror("Pipeline Error", f"Failed to load AudioLDM2 pipeline:\n{e}")
            logging.error(f"Failed to load AudioLDM2 pipeline: {e}")
            return

        # Create a progress window
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Generating Sound Effects")
        self.progress_window.geometry("400x100")
        self.progress_window.grab_set()

        progress_label = tk.Label(
            self.progress_window,
            text="Generating sound effects...",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        progress_label.pack(pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("custom.Horizontal.TProgressbar", troughcolor='#0A2239', background='white')

        self.progress_bar = ttk.Progressbar(
            self.progress_window,
            orient='horizontal',
            length=300,
            mode='determinate',
            style="custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar['maximum'] = number_of_prompts

        # Create a queue for thread communication
        self.queue = queue.Queue()

        def run_all_generations():
            logging.info("run_all_generations started")
            generated_audio_files_per_set = [[] for _ in prompt_sets_sounds]
            ratings_per_set = [[] for _ in prompt_sets_sounds]

            for set_idx, sounds in enumerate(prompt_sets_sounds, start=1):
                logging.info(f"Processing Prompt Set {set_idx}")

                audio_folder = os.path.join(audio_save_folder, f"Video_{set_idx}")
                os.makedirs(audio_folder, exist_ok=True)
                logging.info(f"Created/Using folder: {audio_folder}")

                for sound_idx, sound_text in enumerate(sounds, start=1):
                    if sound_text:
                        logging.info(f"Generating audio for set {set_idx}, sound {sound_idx}: {sound_text}")
                        attempt = 0
                        max_attempts = 3
                        while attempt < max_attempts:
                            try:
                                generator = torch.Generator(device=device).manual_seed(seed + set_idx * 1000 + sound_idx)
                                audio_samples = pipe(
                                    sound_text,
                                    negative_prompt="Low quality.",
                                    num_inference_steps=inference_steps,
                                    audio_length_in_s=duration,
                                    num_waveforms_per_prompt=1,
                                    generator=generator
                                ).audios

                                audio = audio_samples[0]

                                sanitized_sound = self.sanitize_filename(sound_text)
                                logging.info(f"Sanitized sound text: '{sanitized_sound}'")

                                output_filename = os.path.join(
                                    audio_folder,
                                    f"{steps_str}{cfg_str}Audio{set_idx}_SonicLayer{sound_idx}_{sanitized_sound}.wav"
                                )

                                scipy.io.wavfile.write(output_filename, rate=16000, data=audio)
                                logging.info(f"Saved sound effect to {output_filename}")
                                generated_audio_files_per_set[set_idx - 1].append(output_filename)

                                # Update progress
                                self.queue.put(("progress", 1))

                                break  # Success

                            except Exception as e:
                                attempt += 1
                                logging.error(f"[AudioLDM2 Error] Attempt {attempt} failed for set {set_idx}, sound {sound_idx}: {e}")
                                if attempt < max_attempts:
                                    logging.info(f"Retrying set {set_idx}, sound {sound_idx} (Attempt {attempt + 1}/{max_attempts})...")
                                else:
                                    self.queue.put(("error", f"Failed to generate sound effect for set {set_idx}, sound {sound_idx} after {max_attempts} attempts.\nError: {e}"))
                                    logging.error(f"[AudioLDM2 Error] Giving up on set {set_idx}, sound {sound_idx} after {max_attempts} attempts.")

            # Signal completion
            self.queue.put(("complete", "All sound effects have been generated successfully."))

        def process_queue():
            while not self.queue.empty():
                msg_type, content = self.queue.get()
                if msg_type == "progress":
                    self.progress_bar['value'] += content
                elif msg_type == "error":
                    messagebox.showerror("Error", content)
                elif msg_type == "complete":
                    self.progress_window.destroy()
                    messagebox.showinfo("Success", content)
            self.progress_window.after(100, process_queue)

        # Start the thread for generation
        threading.Thread(target=run_all_generations, daemon=True).start()
        logging.info("Audio generation thread started.")

        # Start processing the queue
        self.progress_window.after(100, process_queue)

        
    def get_corresponding_audio_prompt(self, video_prompt_path):
        """
        Finds and loads the corresponding audio prompt list based on the video prompt path.

        :param video_prompt_path: File path of the loaded video prompt list.
        :return: Tuple containing the audio prompts as a string and the file path.
        """
        try:
            base_dir = os.path.dirname(video_prompt_path)
            # Assuming the structure replaces 'Video' with 'Audio' and uses the same filename
            audio_dir = base_dir.replace("\\Video\\", "\\Audio\\")
            audio_filename = os.path.basename(video_prompt_path)
            audio_prompt_path = os.path.join(audio_dir, audio_filename)

            if os.path.exists(audio_prompt_path):
                with open(audio_prompt_path, 'r', encoding='utf-8') as file:
                    prompts = file.read()
                print(f"Automatically loaded corresponding audio prompts from: {audio_prompt_path}")
                return prompts, audio_prompt_path
            else:
                messagebox.showwarning("Audio Prompt Not Found", f"Corresponding audio prompt list not found for {video_prompt_path}.")
                return None, None
        except Exception as e:
            messagebox.showerror("Path Error", f"Error locating audio prompt list:\n{e}")
            print(f"Error locating audio prompt list for {video_prompt_path}: {e}")
            return None, None


        
    def generate_dynamic_audio_prompts(self, prompts, num_variations=3):
        """
        Generate unique audio prompts by introducing slight variations in the sound descriptions
        to avoid repeating the same prompts across different audio sets.
        """
        soundscape_variations = [
            "the gentle rustling of alien foliage",
            "the distant hum of geothermal activity",
            "the soft chirping of insectoid creatures",
            "subtle water droplets on rocky terrain",
            "faint wind brushing against bioluminescent structures"
        ]
        
        dynamic_prompts = []
        
        for prompt in prompts:
            for _ in range(num_variations):
                variation = random.choice(soundscape_variations)
                # Add the variation to the original prompt
                new_prompt = f"{prompt.strip()} with ambient sounds of {variation}."
                dynamic_prompts.append(new_prompt)
        
        return dynamic_prompts


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
            
            # Load Randomizer Settings
            self.video_randomize_theme_var.set(video_options.get("randomize_theme", False))
            self.video_randomize_art_style_var.set(video_options.get("randomize_art_style", False))
            self.video_randomize_lighting_var.set(video_options.get("randomize_lighting", False))
            self.video_randomize_framing_var.set(video_options.get("randomize_framing", False))
            self.video_randomize_camera_movement_var.set(video_options.get("randomize_camera_movement", False))
            self.video_randomize_shot_composition_var.set(video_options.get("randomize_shot_composition", False))
            self.video_randomize_time_of_day_var.set(video_options.get("randomize_time_of_day", False))
            self.video_randomize_decade_var.set(video_options.get("randomize_decade", False))
            self.video_randomize_camera_var.set(video_options.get("randomize_camera", False))
            self.video_randomize_lens_var.set(video_options.get("randomize_lens", False))
            self.video_randomize_resolution_var.set(video_options.get("randomize_resolution", False))
            self.video_randomize_wildlife_animal_var.set(video_options.get("randomize_wildlife_animal", False))
            self.video_randomize_domesticated_animal_var.set(video_options.get("randomize_domesticated_animal", False))
            
            # Load User-Defined Options for Animals
            self.wildlife_animal_entry_var.set(video_options.get("wildlife_animal_custom", ""))
            self.domesticated_animal_entry_var.set(video_options.get("domesticated_animal_custom", ""))
        
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
            self.audio_guidance_scale_var.set(audio_options.get("guidance_scale", 8))
            self.audio_ddim_steps_var.set(audio_options.get("ddim_steps", 40))
            self.audio_n_candidate_var.set(audio_options.get("n_candidate_gen_per_text", 8))
            self.audio_seed_var.set(audio_options.get("seed", 1990))


    def combine_media(self):
        """
        Combines video files with generated sound effects from respective Video_X folders.
        Dynamically layers sound effects and optimizes the audio mix.
        Outputs individual combined video-audio files and combines all into FINAL_VIDEO.mp4.
        """
        
        import os
        import re
        from pydub import AudioSegment
        from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
        import tkinter as tk
        from tkinter import filedialog, messagebox
        import numpy as np
        from scipy.fft import rfft, rfftfreq

        # Helper function for natural sorting without external libraries
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
        
        # Helper function to extract number from filenames based on a given prefix
        def extract_number(filename, prefix):
            """
            Extracts the first occurrence of a number in the filename following the specified prefix.
            Example: extract_number('Video_10.mp4', 'Video') returns 10
            If no match is found, returns a large number to sort unmatched files last.
            """
            pattern = rf'{prefix}_(\d+)'
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            else:
                return float('inf')  # Assign a high number to sort unmatched files last

        # Function to combine all audio layers (WAV files) from a Video_X folder into a single soundscape
        def combine_sound_effects_layers(video_folder):
            SAMPLE_RATE = 16000
            DURATION_MS = 6000  # All sounds are trimmed/padded to 6 seconds
            
            def analyze_sound(audio_segment):
                samples = np.array(audio_segment.get_array_of_samples())
                if len(samples) == 0:
                    return "unknown"
                freqs = rfftfreq(len(samples), 1 / SAMPLE_RATE)
                fft_values = np.abs(rfft(samples))
                dominant_frequency = freqs[np.argmax(fft_values)]
                
                # Classification based on dominant frequency
                if dominant_frequency < 200:
                    return "low"
                elif 200 <= dominant_frequency < 1000:
                    return "mid"
                elif 1000 <= dominant_frequency < 3000:
                    return "high_mid"
                else:
                    return "high"

            def get_dynamic_volume_and_effects(sound_classification):
                # Define volume adjustments and possible effects
                adjustments = {
                    'low': {'volume': -2, 'pan': (-5, 5)},
                    'mid': {'volume': -3, 'pan': (0, 0)},
                    'high_mid': {'volume': -4, 'pan': (3, -3)},
                    'high': {'volume': -5, 'pan': (5, -5)},
                    'unknown': {'volume': -3, 'pan': (0, 0)}
                }
                return adjustments.get(sound_classification, {'volume': -3, 'pan': (0, 0)})

            print(f"Processing folder: {video_folder}")

            # List all .wav files in the folder
            try:
                all_files = os.listdir(video_folder)
            except Exception as e:
                print(f"Error accessing folder {video_folder}: {e}")
                messagebox.showerror("Folder Access Error", f"Cannot access folder: {video_folder}\nError: {e}")
                return None
            print(f"Files in folder: {all_files}")

            sound_effect_files = sorted([f for f in all_files if f.lower().endswith('.wav')], key=natural_sort_key)
            if not sound_effect_files:
                print(f"No .wav files found in {video_folder}.")
                messagebox.showwarning("No Audio Files", f"No .wav files found in {video_folder}. Skipping this folder.")
                return None

            # Start with a 6-second silent audio segment
            combined_audio = AudioSegment.silent(duration=DURATION_MS)

            # Process each sound file
            for sound_file in sound_effect_files:
                sound_file_path = os.path.join(video_folder, sound_file)
                print(f"Loading sound file: {sound_file_path}")

                try:
                    # Load and trim the sound file to 6 seconds
                    sound = AudioSegment.from_wav(sound_file_path)[:DURATION_MS]
                except Exception as e:
                    print(f"Error loading {sound_file_path}: {e}")
                    messagebox.showwarning("Audio Load Error", f"Failed to load audio file: {sound_file_path}\nError: {e}\nSkipping this file.")
                    continue

                # Analyze the sound to classify its frequency range
                classification = analyze_sound(sound)
                print(f"Sound {sound_file} classified as '{classification}' frequency.")

                # Get volume and panning adjustments based on classification
                adjustments = get_dynamic_volume_and_effects(classification)
                dynamic_volume = adjustments['volume']
                pan_left, pan_right = adjustments['pan']
                print(f"Applying {dynamic_volume} dB adjustment and panning ({pan_left}, {pan_right}) to {sound_file}.")

                # Apply volume adjustment
                sound = sound + dynamic_volume

                # Apply panning if necessary
                if pan_left != 0 or pan_right != 0:
                    channels = sound.split_to_mono()
                    if len(channels) == 2:
                        left = channels[0] + pan_left
                        right = channels[1] + pan_right
                        sound = AudioSegment.from_mono_audiosegments(left, right)
                    else:
                        # If mono, duplicate and apply panning
                        sound = AudioSegment.from_mono_audiosegments(sound + pan_left, sound + pan_right)

                # Apply fade in/out for smoothness
                sound = sound.fade_in(50).fade_out(50)

                # Overlay the adjusted sound onto the combined audio
                combined_audio = combined_audio.overlay(sound)

            # Apply master gain to prevent overall loudness
            master_gain = -6  # Reduce overall volume by 6 dB
            combined_audio = combined_audio + master_gain

            # Export the combined soundscape
            combined_audio_path = os.path.join(video_folder, "combined_soundscape.wav")
            try:
                combined_audio.export(combined_audio_path, format="wav")
                print(f"Combined soundscape saved at {combined_audio_path}")
            except Exception as e:
                print(f"Error exporting combined soundscape: {e}")
                messagebox.showerror("Audio Export Error", f"Failed to export combined soundscape:\n{combined_audio_path}\nError: {e}")
                return None

            return combined_audio_path

        # Custom dialog class for sorting method selection using radio buttons
        class SortingDialog(tk.Toplevel):
            def __init__(self, parent):
                super().__init__(parent)
                self.title("Select Sorting Method")
                self.geometry("400x250")
                self.resizable(False, False)
                self.grab_set()  # Make the dialog modal
                self.sorting_method = tk.StringVar(value="default")
                self.result = None

                # Branding: Add a title and description
                title_label = tk.Label(self, text="Sorting Method Selection", font=("Helvetica", 16, "bold"), fg="#333")
                title_label.pack(pady=(20, 10))

                desc_label = tk.Label(self, text="Please choose how you want your videos to be sorted:", font=("Arial", 12), fg="#555")
                desc_label.pack(pady=(0, 20))

                # Radio buttons for sorting options
                rb_steps = tk.Radiobutton(self, text="Steps", variable=self.sorting_method, value="steps", font=("Arial", 12))
                rb_guidance = tk.Radiobutton(self, text="Guidance Scale", variable=self.sorting_method, value="guidance_scale", font=("Arial", 12))
                rb_default = tk.Radiobutton(self, text="Default (Prompt Set Order)", variable=self.sorting_method, value="default", font=("Arial", 12))

                rb_steps.pack(anchor='w', padx=40, pady=5)
                rb_guidance.pack(anchor='w', padx=40, pady=5)
                rb_default.pack(anchor='w', padx=40, pady=5)

                # Frame for OK and Cancel buttons
                button_frame = tk.Frame(self)
                button_frame.pack(pady=20)

                ok_button = tk.Button(button_frame, text="OK", width=10, command=self.on_ok, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
                cancel_button = tk.Button(button_frame, text="Cancel", width=10, command=self.on_cancel, bg="#f44336", fg="white", font=("Arial", 10, "bold"))

                ok_button.pack(side='left', padx=10)
                cancel_button.pack(side='right', padx=10)

            def on_ok(self):
                self.result = self.sorting_method.get()
                self.destroy()

            def on_cancel(self):
                self.result = "default"
                self.destroy()

        # Function to get sorting method via dialog
        def get_sorting_method():
            dialog = SortingDialog(None)
            dialog.wait_window()  # Wait until the dialog is closed
            return dialog.result

        # Ensure video and audio folders are valid
        if not hasattr(self, 'video_save_folder') or not self.video_save_folder or not os.path.exists(self.video_save_folder):
            messagebox.showwarning("No Video Folder Selected", "Please select a valid Video folder.")
            self.video_save_folder = filedialog.askdirectory(title="Select Video Save Folder")
            if not self.video_save_folder:
                messagebox.showerror("Operation Cancelled", "No Video folder selected. Operation cancelled.")
                return  # Exit if no folder is selected

        if not hasattr(self, 'audio_save_folder') or not self.audio_save_folder or not os.path.exists(self.audio_save_folder):
            messagebox.showwarning("No Audio Folder Selected", "Please select a valid Audio folder.")
            self.audio_save_folder = filedialog.askdirectory(title="Select Audio Save Folder")
            if not self.audio_save_folder:
                messagebox.showerror("Operation Cancelled", "No Audio folder selected. Operation cancelled.")
                return  # Exit if no folder is selected

        # Ask the user for sorting preference using radio buttons
        sorting_method = get_sorting_method()
        print(f"User selected sorting method: {sorting_method}")

        # Define sorting key
        def sorting_key(filename):
            if sorting_method == "steps":
                return extract_number(filename, 'Step')
            elif sorting_method == "guidance_scale":
                return extract_number(filename, 'Scale')
            else:
                return extract_number(filename, 'Video')

        # Retrieve and sort video files
        video_files = [f for f in os.listdir(self.video_save_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        video_files = sorted(video_files, key=lambda x: sorting_key(x))
        video_files = [os.path.join(self.video_save_folder, f) for f in video_files]

        print(f"Sorted video files: {video_files}")  # Debugging print

        if not video_files:
            messagebox.showwarning("No Video Files", "No video files found in the Video folder.")
            return

        final_clips = []

        # Process each video and match it with its corresponding audio
        for video_file in video_files:
            # Extract the video number (e.g., Video_1, Video_2, etc.)
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            video_number = extract_number(base_name, 'Video')
            video_folder = os.path.join(self.audio_save_folder, f"Video_{video_number}")

            if not os.path.exists(video_folder):
                messagebox.showwarning("Missing Audio Folder", f"No matching audio folder found for {video_file}. Skipping.")
                continue

            # Combine sound effects layers into a soundscape for this video
            soundscape_path = combine_sound_effects_layers(video_folder)
            if not soundscape_path:
                continue

            try:
                # Load the video clip and trim to 6 seconds
                video_clip = VideoFileClip(video_file).subclip(0, 6)
            except Exception as e:
                print(f"Error loading video {video_file}: {e}")
                messagebox.showwarning("Video Load Error", f"Failed to load video {video_file}.\nError: {e}\nSkipping.")
                continue

            try:
                # Load the combined audio (soundscape)
                audio_clip = AudioFileClip(soundscape_path)
            except Exception as e:
                print(f"Error loading audio {soundscape_path}: {e}")
                messagebox.showwarning("Audio Load Error", f"Failed to load audio for {video_file}.\nError: {e}\nSkipping.")
                continue

            try:
                # Set the combined audio to the video
                final_video = video_clip.set_audio(audio_clip)

                # Save the combined video to the output folder
                output_filename = f"{base_name}_combined.mp4"
                output_path = os.path.join(self.video_save_folder, output_filename)
                final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                print(f"Generated combined video for {video_file} and saved to {output_path}")

                # Add the final video clip to the list for final combination later
                final_clips.append(final_video)
            except Exception as e:
                print(f"Error combining video and audio for {video_file}: {e}")
                messagebox.showwarning("Combine Error", f"Failed to combine video and audio for {video_file}.\nError: {e}\nSkipping.")
                continue

        # Combine all final clips into one video (if more than one)
        if final_clips:
            try:
                final_combined_video = concatenate_videoclips(final_clips, method="compose")
                final_output_path = os.path.join(self.video_save_folder, "FINAL_VIDEO.mp4")
                final_combined_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
                print(f"\nFinal combined video saved to: {final_output_path}")
                messagebox.showinfo("Combine Successful", f"Final combined video saved to:\n{final_output_path}")
            except Exception as e:
                print(f"Error combining final videos: {e}")
                messagebox.showerror("Final Combine Error", f"Failed to create FINAL_VIDEO.mp4.\nError: {e}")
        else:
            messagebox.showwarning("No Videos Selected", "No videos were selected for the final compilation.")

        # Final message to notify the user of completion
        if final_clips:
            messagebox.showinfo("Combine Successful", "The videos and audio have been successfully combined.")
        else:
            messagebox.showinfo("Combine Completed", "The combine process has finished, but no videos were processed.")

    def select_audio_for_video(self, video_file, matching_audio_files):
        """
        Displays a popup for selecting an audio file if multiple matching audio files are found.
        Allows the user to preview the audio before selection.
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        selected_audio_file = [None]

        def play_audio(audio_path):
            audio_clip = AudioFileClip(audio_path)
            audio_clip.preview()

        audio_selection_window = tk.Toplevel(root)
        audio_selection_window.title(f"Select Audio for Video: {os.path.basename(video_file)}")
        tk.Label(audio_selection_window, text=f"Select an audio file for {os.path.basename(video_file)}").pack()

        audio_file_var = tk.StringVar()
        audio_file_var.set(matching_audio_files[0])

        for audio_file in matching_audio_files:
            audio_button = tk.Button(
                audio_selection_window,
                text=os.path.basename(audio_file),
                command=lambda af=audio_file: play_audio(af)
            )
            audio_button.pack()

        def select_audio():
            selected_audio_file[0] = audio_file_var.get()
            audio_selection_window.destroy()

        select_button = tk.Button(audio_selection_window, text="Select", command=select_audio)
        select_button.pack()

        root.wait_window(audio_selection_window)

        return selected_audio_file[0] if selected_audio_file[0] else None


    def create_srt_file(video_file_path, subtitle_text, duration):
        """
        Creates an .srt file with the given subtitle text for the video.
        The subtitle spans the entire video duration.
        """
        srt_file_path = video_file_path.replace('.mp4', '.srt')
        with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write(f"1\n00:00:00,000 --> {format_duration(duration)}\n{subtitle_text}\n\n")


    def format_duration(duration):
        """
        Formats the video duration into SRT-compatible timestamp.
        """
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration - int(duration)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    def build_video_options(self, parent):
        options_label_frame = tk.LabelFrame(
            parent,
            text="Video Prompt Options",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 14, 'bold')
        )
        options_label_frame.pack(fill='both', expand=True, padx=10, pady=10)
        options_label_frame.columnconfigure((0, 1, 2, 3), weight=1)

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
        
        # Initialize Randomizer Variables
        self.video_randomize_theme_var = tk.BooleanVar()
        self.video_randomize_art_style_var = tk.BooleanVar()
        self.video_randomize_lighting_var = tk.BooleanVar()
        self.video_randomize_framing_var = tk.BooleanVar()
        self.video_randomize_camera_movement_var = tk.BooleanVar()
        self.video_randomize_shot_composition_var = tk.BooleanVar()
        self.video_randomize_time_of_day_var = tk.BooleanVar()
        self.video_randomize_decade_var = tk.BooleanVar()
        self.video_randomize_camera_var = tk.BooleanVar()
        self.video_randomize_lens_var = tk.BooleanVar()
        self.video_randomize_resolution_var = tk.BooleanVar()
        self.video_randomize_wildlife_animal_var = tk.BooleanVar()
        self.video_randomize_domesticated_animal_var = tk.BooleanVar()
        
        # Initialize Entry Variables for Animals
        self.wildlife_animal_entry_var = tk.StringVar()
        self.domesticated_animal_entry_var = tk.StringVar()

        # Create dropdowns and other widgets with Randomizer Checkbuttons
        self.create_dropdown_with_randomizer(options_label_frame, "Theme:", THEMES, 0, 0, self.video_theme_var, self.video_randomize_theme_var)
        self.create_dropdown_with_randomizer(options_label_frame, "Art Style:", ART_STYLES, 1, 0, self.video_art_style_var, self.video_randomize_art_style_var)
        self.create_dropdown_with_randomizer(options_label_frame, "Lighting:", LIGHTING_OPTIONS, 2, 0, self.video_lighting_var, self.video_randomize_lighting_var)
        self.create_dropdown_with_randomizer(options_label_frame, "Framing:", FRAMING_OPTIONS, 3, 0, self.video_framing_var, self.video_randomize_framing_var)
        self.create_dropdown_with_randomizer(options_label_frame, "Camera Movement:", CAMERA_MOVEMENTS, 4, 0, self.video_camera_movement_var, self.video_randomize_camera_movement_var)
        self.create_dropdown_with_randomizer(options_label_frame, "Shot Composition:", SHOT_COMPOSITIONS, 5, 0, self.video_shot_composition_var, self.video_randomize_shot_composition_var)
        self.create_dropdown_with_randomizer(options_label_frame, "Time of Day:", TIME_OF_DAY_OPTIONS, 6, 0, self.video_time_of_day_var, self.video_randomize_time_of_day_var)

        # Decade Dropdown with Randomizer
        self.create_dropdown_with_randomizer(options_label_frame, "Decade:", DECADES, 7, 0, self.video_decade_var, self.video_randomize_decade_var)
        self.video_decade_var.trace('w', self.update_video_camera_options)
        self.video_decade_var.trace('w', self.update_resolution_options)

        # Camera Dropdown with Randomizer
        self.video_camera_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.video_camera_var,
            state="readonly",
            values=CAMERAS[DECADES[0]],
            font=('Helvetica', 12)
        )
        self.video_camera_combobox.set(CAMERAS[DECADES[0]][0])  # Set default camera
        self.video_camera_combobox.grid(row=8, column=1, padx=10, pady=10, sticky='ew')

        # Randomize Camera Checkbutton
        self.create_randomizer_checkbutton(options_label_frame, 8, 2, self.video_randomize_camera_var)

        self.create_label(options_label_frame, "Camera:", 8, 0)

        # Lens Dropdown with Randomizer
        self.create_dropdown_with_randomizer(options_label_frame, "Lens:", LENSES, 9, 0, self.video_lens_var, self.video_randomize_lens_var)

        # Resolution Dropdown with Randomizer (Initialize with resolutions from the default decade)
        self.resolution_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.video_resolution_var,
            state="readonly",
            values=RESOLUTIONS[DECADES[0]],
            font=('Helvetica', 12)
        )
        self.resolution_combobox.set(RESOLUTIONS[DECADES[0]][0])  # Set default resolution
        self.resolution_combobox.grid(row=10, column=1, padx=10, pady=10, sticky='ew')

        # Randomize Resolution Checkbutton
        self.create_randomizer_checkbutton(options_label_frame, 10, 2, self.video_randomize_resolution_var)

        self.create_label(options_label_frame, "Resolution:", 10, 0)

        # Wildlife and Domesticated Animal Dropdowns with Randomizers and Entry Boxes
        self.create_dropdown_with_randomizer_and_entry(
            options_label_frame,
            "Wildlife Animal:",
            WILDLIFE_ANIMALS,
            11,
            0,
            self.wildlife_animal_var,
            self.video_randomize_wildlife_animal_var,
            self.wildlife_animal_entry_var
        )

        self.create_dropdown_with_randomizer_and_entry(
            options_label_frame,
            "Domesticated Animal:",
            DOMESTICATED_ANIMALS,
            12,
            0,
            self.domesticated_animal_var,
            self.video_randomize_domesticated_animal_var,
            self.domesticated_animal_entry_var
        )

        # Prompt Count Selection
        prompt_number_label = tk.Label(
            options_label_frame,
            text="Prompt Count - REQUIRED",
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

        # No People Mode Checkbox
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
            text="Story Mode - BETA",
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

    def create_randomizer_checkbutton(self, parent, row, column, var):
        """
        Creates a Randomize checkbutton within the specified parent widget.

        Args:
            parent (tk.Widget): The parent widget.
            row (int): The row position in the grid.
            column (int): The column position in the grid.
            var (tk.Variable): The tkinter variable associated with the checkbutton.
        """
        randomize_cb = ttk.Checkbutton(
            parent,
            text="Randomize",
            variable=var,
            style='TCheckbutton'
        )
        randomize_cb.grid(row=row, column=column, padx=10, pady=10, sticky='w')
        return randomize_cb

    def create_dropdown_with_randomizer(self, parent, label_text, values_list, row, column, var, random_var):
        """
        Creates a labeled dropdown (Combobox) with a Randomize checkbutton.

        Args:
            parent (tk.Widget): The parent widget.
            label_text (str): The text for the label.
            values_list (list): The list of values for the dropdown.
            row (int): The row position in the grid.
            column (int): The column position in the grid.
            var (tk.Variable): The tkinter variable associated with the dropdown.
            random_var (tk.Variable): The tkinter variable for the randomizer checkbutton.
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

        # Create Randomize Checkbutton
        self.create_randomizer_checkbutton(parent, row, column+2, random_var)

    def create_dropdown_with_randomizer_and_entry(self, parent, label_text, values_list, row, column, var, random_var, entry_var):
        """
        Creates a labeled dropdown (Combobox) with a Randomize checkbutton and a text entry box.

        Args:
            parent (tk.Widget): The parent widget.
            label_text (str): The text for the label.
            values_list (list): The list of values for the dropdown.
            row (int): The row position in the grid.
            column (int): The column position in the grid.
            var (tk.Variable): The tkinter variable associated with the dropdown.
            random_var (tk.Variable): The tkinter variable for the randomizer checkbutton.
            entry_var (tk.Variable): The tkinter variable for the text entry box.
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

        # Create Randomize Checkbutton
        self.create_randomizer_checkbutton(parent, row, column+2, random_var)

        # Create Entry Box
        entry = tk.Entry(
            parent,
            textvariable=entry_var,
            font=('Helvetica', 12),
            width=20
        )
        entry.grid(row=row, column=column+3, padx=10, pady=10, sticky='w')
        entry.insert(0, "Add custom options")
            
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
                    "chaos_mode": False,
                    
                    # Randomizer Settings
                    "randomize_theme": False,
                    "randomize_art_style": False,
                    "randomize_lighting": False,
                    "randomize_framing": False,
                    "randomize_camera_movement": False,
                    "randomize_shot_composition": False,
                    "randomize_time_of_day": False,
                    "randomize_decade": False,
                    "randomize_camera": False,
                    "randomize_lens": False,
                    "randomize_resolution": False,
                    "randomize_wildlife_animal": False,
                    "randomize_domesticated_animal": False,
                    
                    # User-Defined Options for Animals
                    "wildlife_animal_custom": "",
                    "domesticated_animal_custom": ""
                },
                "audio_options": {
                    "exclude_music": False,
                    "holiday_mode": False,
                    "selected_holidays": [],
                    "specific_modes": [],
                    "open_source_mode": True,
                    "model_name": "audioldm2-full-large-1150k",
                    "guidance_scale": 8,
                    "ddim_steps": 80,
                    "n_candidate_gen_per_text": 8,
                    "seed": 1990
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
            
            # Load Randomizer Settings
            self.video_randomize_theme_var.set(video_options.get("randomize_theme", False))
            self.video_randomize_art_style_var.set(video_options.get("randomize_art_style", False))
            self.video_randomize_lighting_var.set(video_options.get("randomize_lighting", False))
            self.video_randomize_framing_var.set(video_options.get("randomize_framing", False))
            self.video_randomize_camera_movement_var.set(video_options.get("randomize_camera_movement", False))
            self.video_randomize_shot_composition_var.set(video_options.get("randomize_shot_composition", False))
            self.video_randomize_time_of_day_var.set(video_options.get("randomize_time_of_day", False))
            self.video_randomize_decade_var.set(video_options.get("randomize_decade", False))
            self.video_randomize_camera_var.set(video_options.get("randomize_camera", False))
            self.video_randomize_lens_var.set(video_options.get("randomize_lens", False))
            self.video_randomize_resolution_var.set(video_options.get("randomize_resolution", False))
            self.video_randomize_wildlife_animal_var.set(video_options.get("randomize_wildlife_animal", False))
            self.video_randomize_domesticated_animal_var.set(video_options.get("randomize_domesticated_animal", False))
            
            # Load User-Defined Options for Animals
            self.wildlife_animal_entry_var.set(video_options.get("wildlife_animal_custom", ""))
            self.domesticated_animal_entry_var.set(video_options.get("domesticated_animal_custom", ""))
        
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
        self.audio_guidance_scale_var = tk.DoubleVar(value=8)
        self.audio_ddim_steps_var = tk.IntVar(value=15)
        self.audio_n_candidate_var = tk.IntVar(value=8)
        self.audio_seed_var = tk.IntVar(value=1990)
        self.audio_device_var = tk.StringVar(value="cpu")  # Initialize audio_device_var with default 'cpu'
        self.audio_inference_steps_var = tk.IntVar(value=15)  # Default inference steps
        self.audio_length_var = tk.DoubleVar(value=6.0)  # Default audio length in seconds
        self.audio_waveforms_var = tk.IntVar(value=1)  # Default number of waveforms per prompt

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
            to=50,
            textvariable=self.audio_waveforms_var,
            font=('Helvetica', 12)
        )
        waveforms_spinbox.grid(row=18, column=1, padx=10, pady=5, sticky='w')

        # ---------------------
        # Save Button
        # ---------------------
        save_button = tk.Button(
            parent,
            text="Save Audio Options - Does NOT Close Window",
            command=self.save_audio_options,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 12, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=40,
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
            self.audio_ddim_steps_var.set(audio_options.get("ddim_steps", 15))
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
        
    def remove_unwanted_headers(self, cleaned_prompt):
        """
        Removes any unwanted headers or metadata from the cleaned prompt.
        
        Args:
            cleaned_prompt (str): The prompt text after initial cleaning.
        
        Returns:
            str: The prompt text without unwanted headers.
        """
        # Assuming headers like {"done":true,...} are present, remove them
        # This regex removes JSON-like structures
        cleaned_prompt = re.sub(r'\{.*?\}', '', cleaned_prompt, flags=re.DOTALL)
        
        # Also remove any remaining unwanted text after negative section
        cleaned_prompt = re.split(r'\n--------------------\n', cleaned_prompt)[0]
        
        # Trim any leading/trailing whitespace
        cleaned_prompt = cleaned_prompt.strip()
        
        return cleaned_prompt

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
