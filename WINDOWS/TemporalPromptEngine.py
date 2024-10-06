import os
import sys
import datetime
import json
import subprocess
from dotenv import load_dotenv
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
"Adventure - Exciting journeys",
"Romance - Love stories",
"Horror - Scary elements",
"Sci-Fi - Futuristic concepts",
"Fantasy - Magical realms",
"Thriller - Suspenseful plots",
"Comedy - Humorous content",
"Drama - Emotional narratives",
"Documentary - Real-life stories",
"Mystery - Intriguing puzzles",
"Action - High-energy scenes",
"Animation - Animated visuals",
"Crime - Criminal activities",
"Family - Family-oriented stories",
"Historical - Past events",
"Musical - Music-driven",
"Sport - Athletic themes",
"War - Battle scenarios",
"Western - Frontier settings",
"Biographical - Life stories",
"Political Anarchy - Chaos politics",
"Psychological - Mind-focused",
"Satire - Humorous criticism",
"Noir - Dark and gritty",
"Epic - Grand narratives",
"Superhero - Heroic adventures",
"Dystopian - Dark futures",
"Utopian - Ideal societies",
"Cyberpunk - High-tech low-life",
"Steampunk - Victorian tech",
"Post-Apocalyptic - After disaster",
"Coming of Age - Growth stories",
"Crime Thriller - Suspenseful crimes",
"Spy - Espionage tales",
"Heist - Robbery plans",
"Survival - Endurance stories",
"Mythology - Ancient myths",
"Fairy Tale - Classic tales",
"Noir Romance - Dark love",
"Gothic - Dark romanticism",
"Magical Realism - Real with magic",
"Sword and Sorcery - Fantasy battles",
"Space Opera - Galactic adventures",
"Time Travel - Temporal journeys",
"Urban - City life",
"Rural - Countryside settings",
"Political Thriller - Political suspense",
"Legal Drama - Courtroom stories",
"Medical Drama - Healthcare narratives",
"Espionage - Secret missions",
"Biographical Drama - Life-focused",
"Psychological Thriller - Mind suspense",
"Dark Comedy - Grim humor",
"Romantic Comedy - Love and laughs",
"Slapstick - Physical humor",
"Parody - Imitative humor",
"Black Comedy - Dark humor",
"Romantic Drama - Love and emotion",
"Legal Thriller - Law and suspense",
"Military - Armed forces",
"Espionage Thriller - Spy suspense",
"Disaster - Catastrophe themes",
"High Fantasy - Elaborate magic",
"Low Fantasy - Subtle magic",
"Sword and Sandal - Ancient adventures",
"Biopunk - Biological tech",
"Eco - Environmental themes",
"Heist Comedy - Robbery humor",
"Space Western - Galactic frontier",
"Spy Comedy - Espionage humor",
"Martial Arts - Combat focus",
"Historical Fantasy - Magic in history",
"Urban Fantasy - Magic in cities",
"Science Fantasy - Sci-Fi and magic",
"Romantic Fantasy - Love and magic",
"Mystery Thriller - Puzzle suspense",
"Crime Drama - Criminal narratives",
"Action Comedy - High-energy humor",
"Supernatural - Beyond natural",
"Occult - Hidden mysteries",
"Fable - Moral stories",
"Legend - Traditional tales",
"Tall Tale - Exaggerated stories",
"Folklore - Cultural myths",
"Western Comedy - Frontier humor",
"Neo-Noir - Modern dark",
"Psychological Horror - Mind scares",
"Eco Thriller - Environmental suspense",
"Legal Comedy - Law humor",
"Medical Comedy - Healthcare humor",
"Historical Romance - Past love",
"Steampunk Fantasy - Victorian magic",
"Cyberpunk Noir - Dark tech",
"Biographical Comedy - Life humor",
"Adventure Comedy - Journey humor",
"Romantic Thriller - Love suspense",
"Mystery Comedy - Puzzle humor",
"Horror Comedy - Scary humor",
"Fantasy Adventure - Magical journeys",
"Science Fantasy Adventure - Sci-Fi magic",
"Urban Mystery - City puzzles",
"Rural Mystery - Countryside puzzles",
"Space Mystery - Galactic puzzles",
"Time Travel Mystery - Temporal puzzles",
"Superhero Thriller - Hero suspense",
"Spy Action - Espionage action",
"Heist Thriller - Robbery suspense",
"Survival Horror - Endurance scares",
"Mythological Fantasy - Ancient magic",
"Fairy Tale Fantasy - Classic magic",
"Noir Mystery - Dark puzzle",
"Gothic Horror - Dark scares",
"Magical Realism Drama - Real and magic",
"Space Opera Adventure - Galactic journey",
"Time Travel Adventure - Temporal journey",
"Urban Drama - City emotions",
"Rural Drama - Countryside emotions",
"Political Drama - Political narratives",
"Legal Drama - Courtroom stories",
"Medical Drama - Healthcare narratives",
"Espionage Drama - Spy stories",
"Biographical Drama - Life-focused",
"Psychological Drama - Mind-focused",
"Dark Drama - Grim emotions",
"Romantic Drama - Love and emotion",
"Adventure Drama - Journey emotions"
]

ART_STYLES = [
"Realism - True-to-life depiction",
"Surrealism - Dream-like imagery",
"Impressionism - Light and color focus",
"Expressionism - Emotional expression",
"Minimalism - Simplified forms",
"Abstract - Non-representational",
"Noir - Dark and moody",
"Art Deco - Decorative elegance",
"Cyberpunk - High-tech aesthetics",
"Steampunk - Victorian machinery",
"Retro-Futurism - Past's future",
"Modernism - Contemporary styles",
"Baroque - Ornate details",
"Cubism - Geometric shapes",
"Pop Art - Popular culture",
"Gothic - Dark romanticism",
"Vaporwave - Nostalgic synth",
"Biopunk - Biological tech",
"Pixel Art - Retro digital",
"Anime - Japanese animation",
"Watercolor - Soft pigment washes",
"Low Poly - Simple 3D models",
"High Fantasy - Elaborate magic",
"Comic Book - Illustrated storytelling",
"Art Nouveau - Flowing lines",
"Futurism - Dynamic movement",
"Constructivism - Industrial style",
"Dadaism - Anti-art movement",
"Op Art - Optical illusions",
"Art Brut - Raw art",
"Photorealism - Photo-like art",
"Symbolism - Symbolic imagery",
"Fresco - Mural painting",
"Graffiti - Street art",
"Mosaic - Piecewise composition",
"Collage - Mixed media",
"Pointillism - Dot technique",
"Ink Wash - Monochrome gradients",
"Chiaroscuro - Light and shadow",
"Monochrome - Single color tones",
"Color Field - Large color areas",
"Neo-Expressionism - Modern emotion",
"Post-Impressionism - Beyond impressionism",
"Transavantgarde - Italian neo-expressionism",
"Digital Art - Computer-generated",
"Conceptual Art - Idea-driven",
"Installation Art - Immersive setups",
"Performance Art - Live expression",
"Land Art - Nature integration",
"Fiber Art - Textile-based",
"Ceramic Art - Clay sculpting",
"Glass Art - Glass sculpting",
"Metalwork - Metal sculpting",
"Woodwork - Wood sculpting",
"Sculpture - 3D art",
"Tattoo Art - Body ink",
"Pixel Art - Retro gaming",
"Vector Art - Scalable graphics",
"Typography - Lettering design",
"Calligraphy - Artistic writing",
"Digital Painting - Digital brushes",
"Mixed Media - Combined materials",
"3D Modeling - Three-dimensional design",
"Virtual Reality Art - Immersive art",
"Augmented Reality Art - Overlayed art",
"Interactive Art - User engagement",
"Light Art - Light-based creations",
"Kinetic Art - Movement-based",
"Assemblage - Three-dimensional collage",
"Stencil Art - Repeated patterns",
"Spray Paint Art - Aerosol techniques",
"Stained Glass - Colored glass pieces",
"Encaustic - Wax-based painting",
"Tempera - Egg-based paint",
"Sgraffito - Scratch techniques",
"Digital Collage - Computer mixed media",
"Paper Cutting - Intricate designs",
"Found Object - Recycled materials",
"Neo-Classical - Modern classicism",
"Futuristic - Forward-looking designs",
"Symbolic - Representational symbols",
"Decorative Art - Ornamentation",
"Environmental Art - Eco-friendly",
"Geometric - Shape-focused",
"Biomorphic - Organic shapes",
"Fractal Art - Mathematical patterns",
"Glitch Art - Digital errors",
"Pixel Sorting - Digital manipulation",
"Algorithmic Art - Code-driven",
"Generative Art - Autonomously created",
"Procedural Art - Process-based",
"Data Visualization - Informative graphics",
"Infographic Art - Data and design",
"Memphis Design - Bold patterns",
"Brutalism - Raw simplicity",
"De Stijl - Abstract minimalism",
"Suprematism - Basic geometric forms",
"Action Painting - Expressive brushwork",
"Colorism - Emphasis on color",
"Hard-Edge Painting - Sharp boundaries",
"Lyrical Abstraction - Free-form expression",
"Arte Povera - Use of poor materials",
"New Media Art - Digital technologies",
"Postmodernism - Diverse styles",
"Street Art - Urban expression",
"Neo-Pop - Modern pop culture",
"Lowbrow Art - Pop surrealism",
"Transgressive Art - Challenging norms",
"Digital Graffiti - Virtual street art",
"Projection Mapping - Light displays",
"Holographic Art - 3D projections",
"Interactive Installation - User participation",
"Bio Art - Living materials",
"Sound Art - Audio-based",
"Light Sculpture - Illuminated forms",
"Mixed Reality Art - Blended realities",
"Algorithmic Painting - Code-created art",
"Biomorphic Abstraction - Organic abstraction",
"Constructive Art - Building-focused",
"Deco-Futurism - Elegant future",
"Eco Art - Environmental themes",
"Expressive Abstraction - Emotional shapes",
"Figurative - Representing figures",
"Gesture Drawing - Quick sketches",
"Hard-Edge Abstraction - Clean lines",
"Illusionism - Optical tricks",
"Interior Design Art - Space aesthetics",
"Landscape Painting - Natural scenes",
"Marine Art - Ocean themes",
"Portraiture - Human figures",
"Still Life - Inanimate objects",
"Urban Sketching - City drawings",
"Vibrant Art - Bright colors",
"Whimsical Art - Playful designs",
"Zen Art - Minimalist aesthetics",
"Abstract Expressionism - Emotional abstraction",
"Bauhaus - Functional design",
"Constructivist Art - Industrial themes",
"Dystopian Art - Dark futures",
"Elegant Art - Graceful designs",
"Fantasy Art - Imaginary scenes",
"Geometric Abstraction - Shape-focused",
"Heritage Art - Cultural legacy",
"Iconography - Symbolic images",
"Juxtaposed Art - Placed together",
"Kitsch - Tacky aesthetics",
"Liminal Art - Transitional themes",
"Modernist Sculpture - Contemporary forms",
"Narrative Art - Storytelling visuals",
"Organic Art - Nature-inspired",
"Psychedelic Art - Mind-bending designs",
"Quirky Art - Unusual styles",
"Regional Art - Local themes",
"Symbolic Abstraction - Meaningful shapes",
"Textural Art - Surface focus",
"Urban Decay Art - Rust and ruins",
"Visual Metaphor - Symbolic imagery",
"Whimsical Illustration - Playful drawings",
"Zen Minimalism - Simple peace",
"Acrylic Painting - Fast-drying medium",
"Charcoal Drawing - Dark sketches",
"Digital Illustration - Computer art",
"Enamel Art - Glassy finish",
"Faux Finishes - Imitated textures",
"Glassblowing - Shaped glass",
"Hand Lettering - Artistic writing",
"Ink Drawing - Pen-based art",
"Jewelry Design - Wearable art",
"Kiln Fired - Ceramic art",
"Leatherworking - Crafted leather",
"Metal Casting - Molded metal",
"Nature Art - Outdoor creations",
"Oil Painting - Rich pigments",
"Paper Mache - Moldable paper",
"Quilling - Rolled paper art",
"Screen Printing - Layered prints",
"Textile Art - Fabric-based",
"Upcycled Art - Recycled materials",
"Vinyl Art - Record-based",
"Wax Art - Molded wax",
"Xerography Art - Photocopy techniques",
"Yarn Bombing - Knit street art",
"Zentangle - Structured doodles"
]

LIGHTING_OPTIONS = [
"Natural Light - Sunlit scenes",
"High Key - Bright and airy",
"Low Key - Dark and moody",
"Rembrandt Lighting - Classic portrait",
"Chiaroscuro - Strong contrasts",
"Backlighting - Silhouette effect",
"Soft Lighting - Gentle illumination",
"Hard Lighting - Sharp shadows",
"Silhouette - Outline only",
"Motivated Lighting - Contextual light",
"Ambient Light - Overall illumination",
"Side Lighting - Dramatic side shadows",
"Golden Hour - Warm sunset tones",
"Blue Hour - Cool twilight hues",
"Candlelight - Warm, flickering light",
"Moonlight - Soft nocturnal glow",
"Studio Lighting - Controlled environment",
"Neon Lighting - Vibrant colors",
"Spotlight - Focused illumination",
"Underlighting - Below light source",
"Top Lighting - Direct overhead",
"Practical Lighting - On-set sources",
"Fill Lighting - Softens shadows",
"Key Lighting - Main light source",
"Bounce Lighting - Reflected light",
"Directional Light - Specific angle",
"Diffused Light - Softens shadows",
"Ambient Occlusion - Soft global shadows",
"Colored Lighting - Hue variation",
"RGB Lighting - Multi-color sources",
"LED Lighting - Energy-efficient",
"Incandescent Lighting - Warm glow",
"Fluorescent Lighting - Cool tones",
"HMI Lighting - Daylight balanced",
"Tungsten Lighting - Warm and bright",
"Firelight - Natural flame light",
"Flash Lighting - Quick bursts",
"Continuous Lighting - Steady light",
"Motion Tracking Lighting - Dynamic changes",
"Dynamic Lighting - Variable intensity",
"Shadow Play - Creative shadows",
"Silky Lighting - Smooth light transitions",
"Softbox Lighting - Even spread",
"Ring Light - Circular illumination",
"Grid Lighting - Directed beams",
"Barndoors - Shape light spread",
"Gobo Lighting - Patterned light",
"Top Light - Ceiling sources",
"Side Rim Lighting - Highlights edges",
"Under Rim Lighting - Underside highlights",
"Cross Lighting - Two side sources",
"Butterfly Lighting - Butterfly shadow",
"Loop Lighting - Defined cheek shadows",
"Split Lighting - Half light, half shadow",
"Broad Lighting - Wider illuminated face",
"Short Lighting - Narrow illuminated face",
"High Contrast Lighting - Extreme differences",
"Low Contrast Lighting - Subtle differences",
"Colored Gel Lighting - Colored filters",
"Blacklight - Ultraviolet illumination",
"Ambient Rim Lighting - Edge highlights",
"Bounced Rim Lighting - Reflective edges",
"Candle Glow - Flickering warmth",
"Ambient Fill Lighting - Soft background",
"Low Angle Lighting - Ground-up light",
"High Angle Lighting - Overhead light",
"Ambient Spotlight - Combined soft and focused",
"Ambient Backlighting - Background illumination",
"Sidelight with Fill - Side and fill combination",
"Front Lighting - Direct frontal light",
"Kicker Lighting - Edge lighting",
"Hair Light - Highlights hair",
"Accent Lighting - Highlights specific areas",
"Snoot Lighting - Narrow beam light",
"Barn Doors Lighting - Control spread",
"Light Painting - Artistic light trails",
"Strobe Lighting - Flashing bursts",
"Bi-Color Lighting - Dual color temperature",
"RGB LED Lighting - Custom colors",
"Fiber Optic Lighting - Fine point light",
"Laser Lighting - Precise beams",
"Smart Lighting - Programmable controls",
"Tunable White Lighting - Adjustable color temp",
"Ultra Violet Lighting - UV effects",
"Infrared Lighting - IR effects",
"Ambient Colored Lighting - Soft colored background",
"Dimmed Lighting - Low intensity",
"Bright Lighting - High intensity",
"Mixed Lighting - Multiple sources",
"Directional Ambient Lighting - Directed overall light",
"Localized Lighting - Specific area focus",
"Punctuated Lighting - Spotty illumination",
"Reactive Lighting - Changes with motion",
"Timed Lighting - Scheduled changes",
"Transitional Lighting - Smooth changes",
"Layered Lighting - Multiple layers",
"Highlight and Shadow Lighting - Defined contrasts",
"Subdued Lighting - Soft and muted",
"Dynamic Range Lighting - Wide intensity range",
"Matte Lighting - Soft without glare",
"Glossy Lighting - Reflective and shiny",
"Backlit Silhouette - Dark foreground with light background",
"Edge Lighting - Highlights object edges",
"Projected Lighting - Patterns or shapes",
"Diffuse Rim Lighting - Soft edge highlights",
"Integrated Lighting - Seamless light sources",
"Controlled Lighting - Precise adjustments",
"Environmental Lighting - Matches surroundings",
"Atmospheric Lighting - Creates mood",
"Narrative Lighting - Tells a story",
"Symbolic Lighting - Represents ideas",
"Thematic Lighting - Matches theme",
"Mood Lighting - Sets emotional tone",
"Genre-Specific Lighting - Fits genre norms",
"Contextual Lighting - Fits scene context",
"Realistic Lighting - Mimics real life",
"Stylized Lighting - Artistic interpretation",
"High Contrast Noir Lighting - Noir with strong contrasts",
"Soft Romantic Lighting - Gentle and warm",
"Harsh Industrial Lighting - Strong and stark",
"Ethereal Lighting - Light and airy",
"Mystical Lighting - Magical glow",
"Vintage Lighting - Old-fashioned style",
"Modern Lighting - Contemporary style",
"Futuristic Lighting - Advanced and sleek",
"Retro Lighting - Past-inspired style",
"Industrial Lighting - Utilitarian style",
"Artistic Lighting - Creative approaches",
"Technical Lighting - Precision-based",
"Experimental Lighting - Innovative techniques",
"Balanced Lighting - Even distribution",
"Asymmetrical Lighting - Uneven distribution",
"Complementary Lighting - Harmonious colors",
"Analog Lighting - Traditional sources",
"Digital Lighting - Modern sources",
"Multi-Layered Lighting - Several light layers",
"Shadowless Lighting - No shadows",
"Edge-Free Lighting - Smooth transitions",
"Contour Lighting - Highlights shapes",
"Accent Rim Lighting - Enhanced edge highlights",
"Color Gradient Lighting - Smooth color transitions",
"Light Blending - Merging light sources",
"Temperature Controlled Lighting - Adjusted warmth",
"Intensity Controlled Lighting - Adjusted brightness",
"Direction Controlled Lighting - Adjustable angles",
"Focus Controlled Lighting - Sharpened areas",
"Expansion Lighting - Spread wide coverage",
"Contraction Lighting - Narrow focused light",
"Patterned Lighting - Repeating patterns",
"Animated Lighting - Moving light sources",
"Static Lighting - Fixed light sources",
"Continuous Soft Lighting - Ongoing gentle light",
"Intermittent Hard Lighting - Sporadic strong light",
"Overlay Lighting - Multiple layers over each other",
"Integrated Ambient Lighting - Blended overall light",
"Dynamic Ambient Lighting - Changing overall light",
"Selective Ambient Lighting - Chosen overall light",
"Reflected Ambient Lighting - Reflected light sources",
"Primary Ambient Lighting - Main light source",
"Secondary Ambient Lighting - Supporting light source",
"Tertiary Ambient Lighting - Additional light source",
"Localized Ambient Lighting - Specific area light",
"General Ambient Lighting - Overall light",
"Integrated Practical Lighting - On-set light sources",
"Supplemental Practical Lighting - Additional on-set light",
"Soft Practical Lighting - Gentle on-set light",
"Hard Practical Lighting - Strong on-set light",
"Decorative Practical Lighting - Aesthetic on-set light",
"Functional Practical Lighting - Utility on-set light",
"Minimalist Practical Lighting - Simple on-set light",
"Maximalist Practical Lighting - Elaborate on-set light",
"Adaptive Practical Lighting - Responsive on-set light",
"Interactive Practical Lighting - User-controlled on-set light",
"Contextual Practical Lighting - Scene-matching on-set light",
"Artistic Practical Lighting - Creative on-set light",
"Symbolic Practical Lighting - Meaningful on-set light",
"Thematic Practical Lighting - Theme-matching on-set light",
"Mood Practical Lighting - Emotion-setting on-set light",
"Genre-Specific Practical Lighting - Genre-aligned on-set light",
"Narrative Practical Lighting - Story-driven on-set light",
"Technical Practical Lighting - Precision on-set light",
"Experimental Practical Lighting - Innovative on-set light",
"Balanced Practical Lighting - Even on-set light",
"Asymmetrical Practical Lighting - Uneven on-set light",
"Complementary Practical Lighting - Harmonious on-set light",
"Analog Practical Lighting - Traditional on-set light",
"Digital Practical Lighting - Modern on-set light",
"Multi-Layered Practical Lighting - Several on-set light layers",
"Shadowless Practical Lighting - No shadows on set",
"Edge-Free Practical Lighting - Smooth on-set light",
"Contour Practical Lighting - Shape-highlighted on-set light",
"Accent Rim Practical Lighting - Edge-enhanced on-set light",
"Color Gradient Practical Lighting - Smooth color transitions on set",
"Light Blending Practical Lighting - Merged on-set light",
"Temperature Controlled Practical Lighting - Adjusted warmth on set",
"Intensity Controlled Practical Lighting - Adjusted brightness on set",
"Direction Controlled Practical Lighting - Adjustable on-set light angles",
"Focus Controlled Practical Lighting - Sharpened on-set light areas",
"Expansion Practical Lighting - Wide on-set light coverage",
"Contraction Practical Lighting - Focused on-set light",
"Patterned Practical Lighting - Repeating on-set light patterns",
"Animated Practical Lighting - Moving on-set light sources",
"Static Practical Lighting - Fixed on-set light sources"
]

FRAMING_OPTIONS = [
"Wide Shot - Broad view",
"Close-Up - Detailed focus",
"Medium Shot - Waist level",
"Over-the-Shoulder - Perspective view",
"Dutch Angle - Tilted frame",
"Bird's Eye View - Top-down",
"Point of View - First-person",
"Two-Shot - Dual subjects",
"Extreme Close-Up - Minute details",
"Panoramic - Wide landscape",
"Establishing Shot - Scene setting",
"Long Shot - Full body view",
"Medium Close-Up - Upper body focus",
"Full Shot - Complete figure",
"Cowboy Shot - Mid-thigh framing",
"Cut-In - Specific detail",
"Cutaway - Secondary action",
"Master Shot - Continuous action",
"Aerial Shot - Elevated view",
"Insert Shot - Close detail",
"Reverse Angle - Opposite perspective",
"Clean Single - Single subject",
"Dirty Single - Single with distractions",
"Reaction Shot - Subject's response",
"Extreme Wide Shot - Vast surroundings",
"Full Body Shot - Entire figure",
"Headshot - Focus on head",
"High Angle Shot - Looking down",
"Low Angle Shot - Looking up",
"Shoulder Shot - Partial head",
"Tracking Shot - Following movement",
"Static Shot - Fixed position",
"Dynamic Shot - Moving camera",
"Two-Shot Over-the-Shoulder - Dual perspective",
"Triple Shot - Three subjects",
"Quad Shot - Four subjects",
"Quintet Shot - Five subjects",
"Group Shot - Multiple subjects",
"Crowd Shot - Large group",
"Tableau - Static arrangement",
"Oblique Angle - Slanted frame",
"Worm's Eye View - Ground-up",
"Tilt Shot - Vertical movement",
"Pan Shot - Horizontal movement",
"Zoom Shot - Variable distance",
"Handheld Shot - Unstable camera",
"Steadicam Shot - Smooth movement",
"Crane Shot - Elevated movement",
"Dolly Shot - Tracking movement",
"Truck Shot - Lateral movement",
"Arc Shot - Circular movement",
"Whip Pan Shot - Quick horizontal",
"Rack Focus Shot - Depth change",
"Push In Shot - Camera moves closer",
"Pull Back Shot - Camera moves away",
"Roll Shot - Camera rotates",
"Boom Shot - Overhead movement",
"Random Movement Shot - Unpredictable motion",
"Virtual Camera Movement - Digital motion",
"Time-Lapse Shot - Time acceleration",
"Hyperlapse Shot - Time and space",
"Slow Motion Shot - Reduced speed",
"Fast Motion Shot - Increased speed",
"Frozen Time Shot - Static frame",
"Bullet Time Shot - 360-degree view",
"Panoramic Tilt Shot - Wide vertical",
"Dutch Tilt Close-Up - Tilted detail",
"Extreme High Angle Shot - Very top-down",
"Extreme Low Angle Shot - Very ground-up",
"Wide Angle Shot - Broad perspective",
"Telephoto Shot - Narrow focus",
"Macro Shot - Extreme close-up",
"Fisheye Shot - Distorted wide view",
"Prime Lens Shot - Fixed focus",
"Zoom Lens Shot - Variable focus",
"Anamorphic Shot - Wide cinematic",
"Tilt-Shift Shot - Miniature effect",
"Ultra Wide Angle Shot - Extremely broad",
"Standard Lens Shot - Natural view",
"Short Telephoto Shot - Mild zoom",
"Long Telephoto Shot - Deep zoom",
"Catadioptric Shot - Mirror lens effect",
"Soft Focus Shot - Blurred edges",
"Infrared Shot - Heat vision",
"UV Shot - Ultraviolet view",
"Cine Lens Shot - Cinematic quality",
"Portrait Lens Shot - Face-focused",
"Super Telephoto Shot - High zoom",
"Pancake Lens Shot - Compact view",
"Refractive Shot - Light-bending",
"Mirror Lens Shot - Infinity reflections",
"Perspective Control Shot - Depth manipulation",
"Fish-Eye Circular Shot - Circular distortion",
"Wide Pan Shot - Extensive horizontal",
"Narrow Pan Shot - Limited horizontal",
"Vertical Tilt Shot - Up and down",
"Horizontal Tilt Shot - Side to side",
"Diagonal Framing Shot - Angled lines",
"Symmetrical Framing Shot - Balanced sides",
"Asymmetrical Framing Shot - Unbalanced sides",
"Rule of Thirds Shot - Balanced thirds",
"Golden Ratio Shot - Natural proportions",
"Centered Composition Shot - Central focus",
"Off-Center Shot - Side focus",
"Leading Lines Shot - Directing view",
"Framing Within Framing - Nested frames",
"Negative Space Shot - Emptiness focus",
"Depth of Field Shot - Focus layers",
"Shallow Depth Shot - Limited focus",
"Deep Depth Shot - Extensive focus",
"Layered Framing Shot - Multiple layers",
"Foreground Interest Shot - Visible front",
"Background Interest Shot - Visible back",
"Symmetrical Composition Shot - Mirror balance",
"Asymmetrical Composition Shot - Uneven balance",
"Dynamic Composition Shot - Active layout",
"Static Composition Shot - Fixed layout",
"Balanced Composition Shot - Even distribution",
"Unbalanced Composition Shot - Uneven distribution",
"Textured Composition Shot - Surface details",
"Color Contrast Composition Shot - Hue differences",
"Silhouette Composition Shot - Outline focus",
"Reflection Composition Shot - Mirror images",
"Subframing Composition Shot - Partial frames",
"Juxtaposition Composition Shot - Contrast elements",
"Pattern Composition Shot - Repeating motifs",
"Geometric Composition Shot - Shape focus",
"Organic Composition Shot - Natural forms",
"Minimalist Composition Shot - Simple elements",
"Maximalist Composition Shot - Complex elements",
"Diagonal Composition Shot - Slanting lines",
"Circular Composition Shot - Round elements",
"Triangular Composition Shot - Three-point focus",
"Spatial Composition Shot - 3D space",
"Linear Composition Shot - Straight lines",
"Zigzag Composition Shot - Angled lines",
"Radiant Composition Shot - Spreading lines",
"Grid Composition Shot - Structured layout",
"Frame within Frame Composition Shot - Nested frames",
"Overlapping Composition Shot - Layered elements",
"Rule Breaking Composition Shot - Unconventional layout",
"Balanced Asymmetry Composition Shot - Controlled imbalance",
"Centralized Composition Shot - Single point focus",
"Distributed Composition Shot - Spread focus",
"Isolated Composition Shot - Single element focus",
"Clustered Composition Shot - Grouped elements",
"Fractal Composition Shot - Repeating patterns",
"Text-Based Composition Shot - Lettering focus",
"Icon-Based Composition Shot - Symbol focus",
"Minimal Detail Composition Shot - Sparse elements",
"Detailed Composition Shot - Rich elements",
"Transparent Composition Shot - See-through elements",
"Opaque Composition Shot - Solid elements",
"Dimensional Composition Shot - 3D elements",
"Flat Composition Shot - 2D elements",
"Light Composition Shot - Bright elements",
"Dark Composition Shot - Shadowed elements",
"Colorful Composition Shot - Vibrant colors",
"Monochrome Composition Shot - Single color tone",
"High Contrast Composition Shot - Sharp differences",
"Low Contrast Composition Shot - Subtle differences",
"Muted Composition Shot - Soft colors",
"Vivid Composition Shot - Bright colors",
"Highlight Composition Shot - Emphasized areas",
"Shadow Composition Shot - Darkened areas",
"Balanced Framing - Equal elements",
"Dynamic Framing - Active elements",
"Static Framing - Still elements",
"Interactive Framing - Engaging elements",
"Narrative Framing - Story elements",
"Symbolic Framing - Representational elements",
"Thematic Framing - Theme-aligned elements",
"Genre-Specific Framing - Genre-aligned elements",
"Contextual Framing - Scene-matching elements",
"Technical Framing - Precision elements",
"Experimental Framing - Innovative elements",
"Balanced Layering Shot - Even layers",
"Asymmetrical Layering Shot - Uneven layers",
"Colorful Layering Shot - Vibrant layers",
"Monochrome Layering Shot - Single tone layers",
"Textured Layering Shot - Surface layers",
"Patterned Layering Shot - Repeating layers",
"Geometric Layering Shot - Shape layers",
"Organic Layering Shot - Natural layers",
"Minimalist Layering Shot - Simple layers",
"Maximalist Layering Shot - Complex layers",
"Symmetrical Layering Shot - Mirror layers",
"Asymmetrical Layering Shot - Uneven layers",
"Dynamic Layering Shot - Active layers",
"Static Layering Shot - Fixed layers",
"Balanced Element Shot - Even elements",
"Asymmetrical Element Shot - Uneven elements",
"Layered Element Shot - Multiple elements",
"Overlapping Element Shot - Layered elements",
"Isolated Element Shot - Single element focus",
"Grouped Element Shot - Clustered elements",
"Distributed Element Shot - Spread elements",
"Central Element Shot - Single central focus",
"Peripheral Element Shot - Side elements",
"Foreground Element Shot - Front focus",
"Background Element Shot - Back focus",
"Balanced Space Shot - Even space",
"Asymmetrical Space Shot - Uneven space",
"Layered Space Shot - Multiple layers",
"Depth Space Shot - 3D space",
"Flat Space Shot - 2D space",
"Transparent Space Shot - See-through space",
"Opaque Space Shot - Solid space",
"Dynamic Space Shot - Active space",
"Static Space Shot - Still space",
"Symbolic Space Shot - Representational space",
"Narrative Space Shot - Story-driven space",
"Thematic Space Shot - Theme-aligned space",
"Genre-Specific Space Shot - Genre-aligned space",
"Contextual Space Shot - Scene-matching space",
"Technical Space Shot - Precision space",
"Experimental Space Shot - Innovative space",
"Balanced Composition - Even distribution",
"Dynamic Composition - Active layout",
"Symmetrical Composition - Mirror balance",
"Asymmetrical Composition - Uneven balance",
"Rule of Thirds Composition - Balanced thirds",
"Golden Ratio Composition - Natural proportions",
"Centered Composition - Central focus",
"Off-Center Composition - Side focus",
"Leading Lines Composition - Directing view",
"Framing Composition - Nested frames",
"Negative Space Composition - Emptiness focus",
"Depth Composition - Focus layers",
"Pattern Composition - Repeating motifs",
"Juxtaposition Composition - Contrast elements",
"Geometric Composition - Shape focus",
"Organic Composition - Natural forms",
"Minimalist Composition - Simple elements",
"Maximalist Composition - Complex elements",
"Diagonal Composition - Slanting lines",
"Circular Composition - Round elements",
"Triangular Composition - Three-point focus",
"Spatial Composition - 3D space",
"Linear Composition - Straight lines",
"Zigzag Composition - Angled lines",
"Radiant Composition - Spreading lines",
"Grid Composition - Structured layout",
"Frame within Frame Composition - Nested frames",
"Overlapping Composition - Layered elements",
"Rule Breaking Composition - Unconventional layout",
"Balanced Asymmetry Composition - Controlled imbalance",
"Centralized Composition - Single point focus",
"Distributed Composition - Spread focus",
"Isolated Composition - Single element focus",
"Clustered Composition - Grouped elements",
"Fractal Composition - Repeating patterns",
"Text-Based Composition - Lettering focus",
"Icon-Based Composition - Symbol focus",
"Minimal Detail Composition - Sparse elements",
"Detailed Composition - Rich elements",
"Transparent Composition - See-through elements",
"Opaque Composition - Solid elements",
"Dimensional Composition - 3D elements",
"Flat Composition - 2D elements",
"Light Composition - Bright elements",
"Dark Composition - Shadowed elements",
"Colorful Composition - Vibrant colors",
"Monochrome Composition - Single color tone",
"High Contrast Composition - Sharp differences",
"Low Contrast Composition - Subtle differences",
"Muted Composition - Soft colors",
"Vivid Composition - Bright colors",
"Highlight Composition - Emphasized areas",
"Shadow Composition - Darkened areas",
"Balanced Framing - Equal elements",
"Dynamic Framing - Active elements",
"Static Framing - Still elements",
"Interactive Framing - Engaging elements",
"Narrative Framing - Story elements",
"Symbolic Framing - Representational elements",
"Thematic Framing - Theme-aligned elements",
"Genre-Specific Framing - Genre-aligned elements",
"Contextual Framing - Scene-matching elements",
"Technical Framing - Precision elements",
"Experimental Framing - Innovative elements",
"Balanced Layering Shot - Even layers",
"Asymmetrical Layering Shot - Uneven layers",
"Colorful Layering Shot - Vibrant layers",
"Monochrome Layering Shot - Single tone layers",
"Textured Layering Shot - Surface layers",
"Patterned Layering Shot - Repeating layers",
"Geometric Layering Shot - Shape layers",
"Organic Layering Shot - Natural layers",
"Minimalist Layering Shot - Simple layers",
"Maximalist Layering Shot - Complex layers",
"Symmetrical Layering Shot - Mirror layers",
"Asymmetrical Layering Shot - Uneven layers",
"Dynamic Layering Shot - Active layers",
"Static Layering Shot - Fixed layers",
"Balanced Element Shot - Even elements",
"Asymmetrical Element Shot - Uneven elements",
"Layered Element Shot - Multiple elements",
"Overlapping Element Shot - Layered elements",
"Isolated Element Shot - Single element focus",
"Grouped Element Shot - Clustered elements",
"Distributed Element Shot - Spread elements",
"Central Element Shot - Single central focus",
"Peripheral Element Shot - Side elements",
"Foreground Element Shot - Front focus",
"Background Element Shot - Back focus",
"Balanced Space Shot - Even space",
"Asymmetrical Space Shot - Uneven space",
"Layered Space Shot - Multiple layers",
"Depth Space Shot - 3D space",
"Flat Space Shot - 2D space",
"Transparent Space Shot - See-through space",
"Opaque Space Shot - Solid space",
"Dynamic Space Shot - Active space",
"Static Space Shot - Still space",
"Symbolic Space Shot - Representational space",
"Narrative Space Shot - Story-driven space",
"Thematic Space Shot - Theme-aligned space",
"Genre-Specific Space Shot - Genre-aligned space",
"Contextual Space Shot - Scene-matching space",
"Technical Space Shot - Precision space",
"Experimental Space Shot - Innovative space",
"Balanced Composition - Even distribution",
"Dynamic Composition - Active layout",
"Symmetrical Composition - Mirror balance",
"Asymmetrical Composition - Uneven balance",
"Rule of Thirds Composition - Balanced thirds",
"Golden Ratio Composition - Natural proportions",
"Centered Composition - Central focus",
"Off-Center Composition - Side focus",
"Leading Lines Composition - Directing view",
"Framing Composition - Nested frames",
"Negative Space Composition - Emptiness focus",
"Depth Composition - Focus layers",
"Pattern Composition - Repeating motifs",
"Juxtaposition Composition - Contrast elements",
"Geometric Composition - Shape focus",
"Organic Composition - Natural forms",
"Minimalist Composition - Simple elements",
"Maximalist Composition - Complex elements",
"Diagonal Composition - Slanting lines",
"Circular Composition - Round elements",
"Triangular Composition - Three-point focus",
"Spatial Composition - 3D space",
"Linear Composition - Straight lines",
"Zigzag Composition - Angled lines",
"Radiant Composition - Spreading lines",
"Grid Composition - Structured layout",
"Frame within Frame Composition - Nested frames",
"Overlapping Composition - Layered elements",
"Rule Breaking Composition - Unconventional layout",
"Balanced Asymmetry Composition - Controlled imbalance",
"Centralized Composition - Single point focus",
"Distributed Composition - Spread focus",
"Isolated Composition - Single element focus",
"Clustered Composition - Grouped elements",
"Fractal Composition - Repeating patterns",
"Text-Based Composition - Lettering focus",
"Icon-Based Composition - Symbol focus",
"Minimal Detail Composition - Sparse elements",
"Detailed Composition - Rich elements",
"Transparent Composition - See-through elements",
"Opaque Composition - Solid elements",
"Dimensional Composition - 3D elements",
"Flat Composition - 2D elements",
"Light Composition - Bright elements",
"Dark Composition - Shadowed elements",
"Colorful Composition - Vibrant colors",
"Monochrome Composition - Single color tone",
"High Contrast Composition - Sharp differences",
"Low Contrast Composition - Subtle differences",
"Muted Composition - Soft colors",
"Vivid Composition - Bright colors",
"Highlight Composition - Emphasized areas",
"Shadow Composition - Darkened areas",
"Balanced Framing - Equal elements",
"Dynamic Framing - Active elements",
"Static Framing - Still elements",
"Interactive Framing - Engaging elements",
"Narrative Framing - Story elements",
"Symbolic Framing - Representational elements",
"Thematic Framing - Theme-aligned elements",
"Genre-Specific Framing - Genre-aligned elements",
"Contextual Framing - Scene-matching elements",
"Technical Framing - Precision elements",
"Experimental Framing - Innovative elements",
"Balanced Layering Shot - Even layers",
"Asymmetrical Layering Shot - Uneven layers",
"Colorful Layering Shot - Vibrant layers",
"Monochrome Layering Shot - Single tone layers",
"Textured Layering Shot - Surface layers",
"Patterned Layering Shot - Repeating layers",
"Geometric Layering Shot - Shape layers",
"Organic Layering Shot - Natural layers",
"Minimalist Layering Shot - Simple layers",
"Maximalist Layering Shot - Complex layers",
"Symmetrical Layering Shot - Mirror layers",
"Asymmetrical Layering Shot - Uneven layers",
"Dynamic Layering Shot - Active layers",
"Static Layering Shot - Fixed layers",
"Balanced Element Shot - Even elements",
"Asymmetrical Element Shot - Uneven elements",
"Layered Element Shot - Multiple elements",
"Overlapping Element Shot - Layered elements",
"Isolated Element Shot - Single element focus",
"Grouped Element Shot - Clustered elements",
"Distributed Element Shot - Spread elements",
"Central Element Shot - Single central focus",
"Peripheral Element Shot - Side elements",
"Foreground Element Shot - Front focus",
"Background Element Shot - Back focus",
"Balanced Space Shot - Even space",
"Asymmetrical Space Shot - Uneven space",
"Layered Space Shot - Multiple layers",
"Depth Space Shot - 3D space",
"Flat Space Shot - 2D space",
"Transparent Space Shot - See-through space",
"Opaque Space Shot - Solid space",
"Dynamic Space Shot - Active space",
"Static Space Shot - Still space",
"Symbolic Space Shot - Representational space",
"Narrative Space Shot - Story-driven space",
"Thematic Space Shot - Theme-aligned space",
"Genre-Specific Space Shot - Genre-aligned space",
"Contextual Space Shot - Scene-matching space",
"Technical Space Shot - Precision space",
"Experimental Space Shot - Innovative space",
"Balanced Composition - Even distribution",
"Dynamic Composition - Active layout",
"Symmetrical Composition - Mirror balance",
"Asymmetrical Composition - Uneven balance",
"Rule of Thirds Composition - Balanced thirds",
"Golden Ratio Composition - Natural proportions",
"Centered Composition - Central focus",
"Off-Center Composition - Side focus",
"Leading Lines Composition - Directing view",
"Framing Composition - Nested frames",
"Negative Space Composition - Emptiness focus",
"Depth Composition - Focus layers",
"Pattern Composition - Repeating motifs",
"Juxtaposition Composition - Contrast elements",
"Geometric Composition - Shape focus",
"Organic Composition - Natural forms",
"Minimalist Composition - Simple elements",
"Maximalist Composition - Complex elements",
"Diagonal Composition - Slanting lines",
"Circular Composition - Round elements",
"Triangular Composition - Three-point focus",
"Spatial Composition - 3D space",
"Linear Composition - Straight lines",
"Zigzag Composition - Angled lines",
"Radiant Composition - Spreading lines",
"Grid Composition - Structured layout",
"Frame within Frame Composition - Nested frames",
"Overlapping Composition - Layered elements",
"Rule Breaking Composition - Unconventional layout",
"Balanced Asymmetry Composition - Controlled imbalance",
"Centralized Composition - Single point focus",
"Distributed Composition - Spread focus",
"Isolated Composition - Single element focus",
"Clustered Composition - Grouped elements",
"Fractal Composition - Repeating patterns",
"Text-Based Composition - Lettering focus",
"Icon-Based Composition - Symbol focus",
"Minimal Detail Composition - Sparse elements",
"Detailed Composition - Rich elements",
"Transparent Composition - See-through elements",
"Opaque Composition - Solid elements",
"Dimensional Composition - 3D elements",
"Flat Composition - 2D elements",
"Light Composition - Bright elements",
"Dark Composition - Shadowed elements",
"Colorful Composition - Vibrant colors",
"Monochrome Composition - Single color tone",
"High Contrast Composition - Sharp differences",
"Low Contrast Composition - Subtle differences",
"Muted Composition - Soft colors",
"Vivid Composition - Bright colors",
"Highlight Composition - Emphasized areas",
"Shadow Composition - Darkened areas"
]

CAMERA_MOVEMENTS = [
"Pan - Horizontal sweep",
"Tilt - Vertical sweep",
"Dolly - Moving towards/away",
"Truck - Lateral movement",
"Zoom - Changing focal length",
"Crane - Elevated motion",
"Handheld - Unsteady movement",
"Steadicam - Smooth tracking",
"Tracking Shot - Following subject",
"Arc Shot - Circular movement",
"Whip Pan - Rapid horizontal",
"Rack Focus - Shifting focus",
"Pull Back Reveal - Gradual exposure",
"Push In - Approaching subject",
"Roll - Rotational movement",
"Boom - Overhead reach",
"Random Movement - Unpredictable motion",
"Virtual Camera Movement - Digital paths",
"Time-Lapse - Accelerated time",
"Hyperlapse - Time and space fast",
"Slow Motion - Reduced speed",
"Fast Motion - Increased speed",
"Frozen Time - Static frame",
"Bullet Time - 360-degree freeze",
"Pan and Tilt - Combined sweep",
"Orbit Shot - Circular around subject",
"Dolly Zoom - Distorted depth",
"Crab Walk - Side movement",
"Whip Tilt - Rapid vertical",
"Dutch Pan - Tilted horizontal",
"Bird's Eye Pan - High angle sweep",
"Low Angle Pan - Ground level sweep",
"High Angle Pan - Elevated sweep",
"Reverse Pan - Opposite direction",
"Jib Shot - Vertical crane movement",
"Helicopter Shot - Aerial sweep",
"Cable Cam Movement - Controlled path",
"Monorail Movement - Rail-guided path",
"Slide Shot - Side-to-side glide",
"Track Dolly Shot - Guided movement",
"Motorized Steadicam - Automated smooth",
"Free Movement - Unrestricted motion",
"Guided Movement - Controlled path",
"Manual Movement - Hand-operated",
"Electronic Movement - Motorized paths",
"Fluid Movement - Smooth transition",
"Staccato Movement - Quick stops",
"Synchronized Movement - Coordinated paths",
"Independent Movement - Subject-driven",
"Reactive Movement - Responding to action",
"Predictive Movement - Anticipating motion",
"Variable Speed Movement - Adjustable pace",
"Consistent Speed Movement - Uniform pace",
"Easing Movement - Smooth start/end",
"Abrupt Movement - Sudden starts/ends",
"Circular Tracking - Round path",
"Figure-Eight Movement - Intersecting circles",
"Lateral Crane Movement - Side overhead",
"Vertical Crane Movement - Straight overhead",
"360 Crane Movement - Full rotation",
"Boom Dolly Movement - Combined reach and track",
"Cable Dolly Movement - Cable-guided path",
"Wireless Steadicam Movement - Untethered smooth",
"Gyroscopic Movement - Balanced rotation",
"Gimbal Stabilized Movement - Smooth multi-axis",
"Drone Movement - Aerial paths",
"RC Camera Movement - Remote-controlled paths",
"Underwater Camera Movement - Fluid paths",
"Space Camera Movement - Zero gravity paths",
"Macro Camera Movement - Tiny adjustments",
"Wide Camera Movement - Broad sweeps",
"Narrow Camera Movement - Focused paths",
"Panorama Movement - Sweeping panorama",
"Dynamic Pan - Variable speed sweep",
"Static Pan - Fixed speed sweep",
"Compound Movement - Multiple actions",
"Complex Movement - Intricate paths",
"Simple Movement - Basic paths",
"Extended Movement - Long duration",
"Brief Movement - Short duration",
"Timed Movement - Scheduled paths",
"Interactive Movement - User-controlled paths",
"Automated Movement - Pre-programmed paths",
"Choreographed Movement - Planned paths",
"Improvised Movement - Unplanned paths",
"Narrative Movement - Story-driven paths",
"Symbolic Movement - Meaningful paths",
"Expressive Movement - Emotion-driven paths",
"Functional Movement - Purpose-driven paths",
"Artistic Movement - Creativity-driven paths",
"Technical Movement - Precision paths",
"Experimental Movement - Innovative paths",
"Conventional Movement - Traditional paths",
"Unconventional Movement - Non-traditional paths",
"Fluid Tracking - Smooth following",
"Staccato Tracking - Quick stops",
"Reverse Tracking - Opposite direction",
"Side Tracking - Lateral follow",
"Front Tracking - Direct follow",
"Back Tracking - Rear follow",
"Circular Tracking - Round follow",
"Linear Tracking - Straight follow",
"Diagonal Tracking - Slant follow",
"Overhead Tracking - Elevated follow",
"Ground Tracking - Low follow",
"Subject Tracking - Following subject",
"Environment Tracking - Following environment",
"Dynamic Tracking - Variable speed",
"Static Tracking - Fixed speed",
"Smooth Tracking - Even follow",
"Jittered Tracking - Unsteady follow",
"Enhanced Tracking - Improved stability",
"Minimal Tracking - Simple follow",
"Extensive Tracking - Comprehensive follow",
"Selective Tracking - Focused follow",
"Comprehensive Tracking - All-inclusive follow",
"Multi-Directional Tracking - Various directions",
"Omni-Directional Tracking - All around follow",
"Controlled Tracking - Managed follow",
"Uncontrolled Tracking - Free follow",
"Guided Tracking - Directed follow",
"Unguided Tracking - Free-form follow",
"Automated Tracking - Machine-driven follow",
"Manual Tracking - Human-driven follow",
"Adaptive Tracking - Responsive follow",
"Predictive Tracking - Anticipating follow",
"Reactive Tracking - Responding to movements",
"Steady Tracking - Consistent follow",
"Variable Tracking - Changing follow",
"Responsive Tracking - Quick follow",
"Lagging Tracking - Delayed follow",
"Leading Tracking - Ahead follow",
"Balanced Tracking - Even follow",
"Unbalanced Tracking - Uneven follow",
"Synchronized Tracking - Coordinated follow",
"Asynchronized Tracking - Uncoordinated follow",
"Complementary Tracking - Harmonious follow",
"Contrasting Tracking - Opposite follow",
"Parallel Tracking - Side-by-side follow",
"Intersecting Tracking - Crossing follow",
"Overlapping Tracking - Layered follow",
"Intermittent Tracking - Sporadic follow",
"Continuous Tracking - Ongoing follow",
"Smooth Transition Movement - Seamless change",
"Abrupt Transition Movement - Sudden change",
"Gradual Transition Movement - Slow change",
"Sudden Transition Movement - Quick change",
"Seamless Transition Movement - Flawless change",
"Disjointed Transition Movement - Broken change",
"Elevated Transition Movement - Higher change",
"Lower Transition Movement - Lower change",
"Dynamic Transition Movement - Active change",
"Static Transition Movement - Fixed change",
"Fluid Transition Movement - Smooth change",
"Choppy Transition Movement - Rough change",
"Creative Transition Movement - Artistic change",
"Technical Transition Movement - Precision change",
"Symbolic Transition Movement - Meaningful change",
"Narrative Transition Movement - Story-driven change",
"Expressive Transition Movement - Emotion-driven change",
"Functional Transition Movement - Purpose-driven change",
"Artistic Transition Movement - Creativity-driven change",
"Technical Transition Movement - Precision change",
"Experimental Transition Movement - Innovative change",
"Conventional Transition Movement - Traditional change",
"Unconventional Transition Movement - Non-traditional change",
"Balanced Transition Composition - Even change",
"Asymmetrical Transition Composition - Uneven change",
"Color Gradient Transition Movement - Smooth color change",
"Black and White Transition Movement - Monochrome change",
"Colored Transition Movement - Vibrant change",
"Natural Transition Movement - Organic change",
"Artificial Transition Movement - Synthetic change",
"High-Speed Transition Movement - Fast change",
"Low-Speed Transition Movement - Slow change",
"Directional Transition Movement - Specific path change",
"Random Transition Movement - Unpredictable change",
"Patterned Transition Movement - Repeating change",
"Single Direction Transition Movement - One path change",
"Multiple Direction Transition Movement - Multiple paths change",
"Dynamic Speed Transition Movement - Variable speed change",
"Consistent Speed Transition Movement - Uniform speed change",
"Easing Transition Movement - Smooth start/end change",
"Abrupt Start Transition Movement - Sudden start change",
"Abrupt End Transition Movement - Sudden end change",
"Steady Transition Movement - Consistent change",
"Variable Transition Movement - Changing change",
"Symmetrical Transition Movement - Balanced change",
"Asymmetrical Transition Movement - Unbalanced change",
"Layered Transition Movement - Multiple layers change",
"Textured Transition Movement - Surface change",
"Patterned Transition Movement - Repeating patterns change",
"Geometric Transition Movement - Shape change",
"Organic Transition Movement - Natural change",
"Minimalist Transition Movement - Simple change",
"Maximalist Transition Movement - Complex change",
"Symmetrical Transition Movement - Mirror change",
"Asymmetrical Transition Movement - Uneven change",
"Dynamic Composition Movement - Active layout change",
"Static Composition Movement - Fixed layout change",
"Symmetrical Composition Movement - Mirror layout change",
"Asymmetrical Composition Movement - Uneven layout change"
]

SHOT_COMPOSITIONS = [
"Rule of Thirds - Balanced thirds",
"Symmetrical - Mirror balance",
"Asymmetrical - Uneven balance",
"Leading Lines - Directing view",
"Centered Composition - Central focus",
"Framing - Nested frames",
"Depth - Focus layers",
"Negative Space - Emptiness focus",
"Patterns - Repeating motifs",
"Juxtaposition - Contrast elements",
"Golden Ratio - Natural proportions",
"Diagonal Composition - Slanting lines",
"Dynamic Symmetry - Active balance",
"Balanced Composition - Even distribution",
"Unbalanced Composition - Uneven distribution",
"Texture - Surface details",
"Color Contrast - Hue differences",
"Silhouette - Outline focus",
"Reflections - Mirror images",
"Subframing - Partial frames",
"Layering - Multiple layers",
"Foreground Interest - Visible front",
"High Angle - Looking down",
"Low Angle - Looking up",
"Rule of Odds - Odd number focus",
"Golden Spiral - Natural curve",
"Frame within Frame - Nested framing",
"Symmetrical Balance - Mirror elements",
"Radial Balance - Central emanation",
"Visual Weight - Element importance",
"Harmony - Unified elements",
"Contrast - Differing elements",
"Repetition - Iterated elements",
"Variety - Diverse elements",
"Movement - Guided eye flow",
"Emphasis - Highlighted focus",
"Proportion - Size relationships",
"Scale - Relative sizes",
"Hierarchy - Element importance",
"Simplicity - Minimal elements",
"Complexity - Rich elements",
"Focal Point - Primary focus",
"Secondary Focus - Supporting focus",
"Tertiary Focus - Additional focus",
"Balance - Equal weight",
"Asymmetrical Balance - Uneven weight",
"Radial Balance - Central symmetry",
"Spatial Balance - Depth and space",
"Visual Path - Eye movement",
"Dominant Element - Primary object",
"Supporting Elements - Secondary objects",
"Harmony in Colors - Cohesive hues",
"Contrast in Colors - Opposing hues",
"Color Harmony - Unified color scheme",
"Color Contrast - Differing colors",
"Light Contrast - Bright vs dark",
"Texture Contrast - Smooth vs rough",
"Shape Contrast - Geometric vs organic",
"Size Contrast - Large vs small",
"Pattern Contrast - Regular vs irregular",
"Element Contrast - Different objects",
"Focus Contrast - Sharp vs blurred",
"Depth Contrast - Foreground vs background",
"Composition Balance - Even layout",
"Composition Harmony - Unified layout",
"Composition Contrast - Differing layout",
"Composition Rhythm - Repeating layout",
"Composition Flow - Directed layout",
"Composition Unity - Cohesive layout",
"Composition Variety - Diverse layout",
"Visual Rhythm - Repeated elements",
"Visual Harmony - Unified elements",
"Visual Contrast - Differing elements",
"Visual Flow - Directed eye movement",
"Visual Unity - Cohesive elements",
"Visual Variety - Diverse elements",
"Textural Harmony - Unified textures",
"Textural Contrast - Differing textures",
"Color Texture Harmony - Unified color and texture",
"Color Texture Contrast - Differing color and texture",
"Shape Texture Harmony - Unified shape and texture",
"Shape Texture Contrast - Differing shape and texture",
"Light Texture Harmony - Unified light and texture",
"Light Texture Contrast - Differing light and texture",
"Depth Texture Harmony - Unified depth and texture",
"Depth Texture Contrast - Differing depth and texture",
"Color Depth Harmony - Unified color and depth",
"Color Depth Contrast - Differing color and depth",
"Shape Depth Harmony - Unified shape and depth",
"Shape Depth Contrast - Differing shape and depth",
"Light Depth Harmony - Unified light and depth",
"Light Depth Contrast - Differing light and depth",
"Balance Depth - Even depth distribution",
"Symmetrical Depth - Mirror depth",
"Asymmetrical Depth - Uneven depth",
"Radial Depth - Central depth",
"Visual Depth - Layered depth",
"Emotional Depth - Emotional layers",
"Narrative Depth - Story layers",
"Thematic Depth - Theme layers",
"Symbolic Depth - Symbol layers",
"Expressive Depth - Emotion layers",
"Technical Depth - Precision layers",
"Experimental Depth - Innovative layers",
"Balanced Element - Even element distribution",
"Asymmetrical Element - Uneven element distribution",
"Layered Element - Multiple element layers",
"Overlapping Element - Layered element arrangement",
"Isolated Element - Single element focus",
"Grouped Element - Clustered elements",
"Distributed Element - Spread elements",
"Central Element - Single central focus",
"Peripheral Element - Side elements",
"Foreground Element - Front focus",
"Background Element - Back focus",
"Rule of Thirds Element - Thirds-based focus",
"Golden Ratio Element - Natural proportion focus",
"Leading Line Element - Directed focus",
"Framing Element - Nested focus",
"Negative Space Element - Emptiness focus",
"Depth Element - Focus layers",
"Pattern Element - Repeating focus",
"Juxtaposition Element - Contrast focus",
"Geometric Element - Shape focus",
"Organic Element - Natural focus",
"Minimalist Element - Simple focus",
"Maximalist Element - Complex focus",
"Diagonal Element - Slant focus",
"Circular Element - Round focus",
"Triangular Element - Three-point focus",
"Spatial Element - 3D focus",
"Linear Element - Straight focus",
"Zigzag Element - Angled focus",
"Radiant Element - Spreading focus",
"Grid Element - Structured focus",
"Frame within Frame Element - Nested focus",
"Overlapping Element Focus - Layered focus",
"Rule Breaking Element - Unconventional focus",
"Balanced Composition Focus - Even distribution",
"Dynamic Composition Focus - Active layout",
"Symmetrical Composition Focus - Mirror balance",
"Asymmetrical Composition Focus - Uneven balance",
"Rule of Thirds Composition Focus - Balanced thirds",
"Golden Ratio Composition Focus - Natural proportions",
"Centered Composition Focus - Central focus",
"Off-Center Composition Focus - Side focus",
"Leading Lines Composition Focus - Directing view",
"Framing Composition Focus - Nested frames",
"Negative Space Composition Focus - Emptiness focus",
"Depth Composition Focus - Focus layers",
"Pattern Composition Focus - Repeating motifs",
"Juxtaposition Composition Focus - Contrast elements",
"Geometric Composition Focus - Shape focus",
"Organic Composition Focus - Natural forms",
"Minimalist Composition Focus - Simple elements",
"Maximalist Composition Focus - Complex elements",
"Diagonal Composition Focus - Slanting lines",
"Circular Composition Focus - Round elements",
"Triangular Composition Focus - Three-point focus",
"Spatial Composition Focus - 3D space",
"Linear Composition Focus - Straight lines",
"Zigzag Composition Focus - Angled lines",
"Radiant Composition Focus - Spreading lines",
"Grid Composition Focus - Structured layout",
"Frame within Frame Composition Focus - Nested frames",
"Overlapping Composition Focus - Layered elements",
"Rule Breaking Composition Focus - Unconventional layout",
"Balanced Asymmetry Composition Focus - Controlled imbalance",
"Centralized Composition Focus - Single point focus",
"Distributed Composition Focus - Spread focus",
"Isolated Composition Focus - Single element focus",
"Clustered Composition Focus - Grouped elements",
"Fractal Composition Focus - Repeating patterns",
"Text-Based Composition Focus - Lettering focus",
"Icon-Based Composition Focus - Symbol focus",
"Minimal Detail Composition Focus - Sparse elements",
"Detailed Composition Focus - Rich elements",
"Transparent Composition Focus - See-through elements",
"Opaque Composition Focus - Solid elements",
"Dimensional Composition Focus - 3D elements",
"Flat Composition Focus - 2D elements",
"Light Composition Focus - Bright elements",
"Dark Composition Focus - Shadowed elements",
"Colorful Composition Focus - Vibrant colors",
"Monochrome Composition Focus - Single color tone",
"High Contrast Composition Focus - Sharp differences",
"Low Contrast Composition Focus - Subtle differences",
"Muted Composition Focus - Soft colors",
"Vivid Composition Focus - Bright colors",
"Highlight Composition Focus - Emphasized areas",
"Shadow Composition Focus - Darkened areas"
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

FILM_FORMATS = [
# 🎞️ FILM GAUGES & FORMATS
"FILM GAUGES & FORMATS - 8mm Film - Vintage home movies",
"FILM GAUGES & FORMATS - 16mm Film - Indie filmmaking",
"FILM GAUGES & FORMATS - 35mm Film - Professional cinema",
"FILM GAUGES & FORMATS - 70mm Film - Large-scale films",
"FILM GAUGES & FORMATS - IMAX - High-resolution cinema",
"FILM GAUGES & FORMATS - Super 8 - Enhanced 8mm",
"FILM GAUGES & FORMATS - VistaVision - High-resolution format",
"FILM GAUGES & FORMATS - Anamorphic Format - Wide cinematic",
"FILM GAUGES & FORMATS - Super 16mm - Enhanced 16mm",
"FILM GAUGES & FORMATS - Large Format Digital - Big sensor digital",
"FILM GAUGES & FORMATS - Reverse Film - Negative positives",
"FILM GAUGES & FORMATS - Slide Film - Positive transparency",
"FILM GAUGES & FORMATS - Instant Film - Immediate results",
"FILM GAUGES & FORMATS - Film Noir - Dark cinematic style",
"FILM GAUGES & FORMATS - Monochrome - Single color tone",
"FILM GAUGES & FORMATS - Flat 2D Film - Traditional format",
"FILM GAUGES & FORMATS - Flat 3D Film - Single perspective",
"FILM GAUGES & FORMATS - Virtual Reality Film - Immersive cinema",
"FILM GAUGES & FORMATS - 360 Film - All-around view",
"FILM GAUGES & FORMATS - Stereo 3D Film - Dual perspective",

# 🎨 FILM STOCKS & COLOR PROCESSES
"FILM STOCKS & COLOR PROCESSES - Eastman Color Negative Film - Color film stock",
"FILM STOCKS & COLOR PROCESSES - Kodachrome - Classic color film",
"FILM STOCKS & COLOR PROCESSES - Technicolor - Vibrant color process",
"FILM STOCKS & COLOR PROCESSES - Black and White Film - Monochrome",
"FILM STOCKS & COLOR PROCESSES - CineStill - Cinematic film stock",
"FILM STOCKS & COLOR PROCESSES - Color Negative Film - Inverted colors",
"FILM STOCKS & COLOR PROCESSES - Color Grading Film - Enhanced color process",

# 💾 DIGITAL RESOLUTIONS & FORMATS
"DIGITAL RESOLUTIONS & FORMATS - Digital 2K - Standard digital",
"DIGITAL RESOLUTIONS & FORMATS - Digital 4K - Ultra high-definition",
"DIGITAL RESOLUTIONS & FORMATS - Digital 8K - Extreme resolution",
"DIGITAL RESOLUTIONS & FORMATS - HDR Video - High dynamic range",
"DIGITAL RESOLUTIONS & FORMATS - Dolby Vision - Advanced HDR",
"DIGITAL RESOLUTIONS & FORMATS - Log Format - Dynamic range preservation",
"DIGITAL RESOLUTIONS & FORMATS - Raw Format - Unprocessed data",

# 📷 CAMERA SYSTEMS & MODELS
"CAMERA SYSTEMS & MODELS - ARRI Master Prime - Prime lenses",
"CAMERA SYSTEMS & MODELS - Zeiss Compact Prime - High-quality lenses",
"CAMERA SYSTEMS & MODELS - Cooke S4 - Cinematic lenses",
"CAMERA SYSTEMS & MODELS - Leica Summicron - Precision optics",
"CAMERA SYSTEMS & MODELS - Arriflex - Professional cameras",
"CAMERA SYSTEMS & MODELS - Panasonic Film Cameras - Versatile use",
"CAMERA SYSTEMS & MODELS - RED Digital Cinema - High-res digital",
"CAMERA SYSTEMS & MODELS - Arri Alexa - Industry standard",
"CAMERA SYSTEMS & MODELS - Blackmagic Design - Affordable high-res",
"CAMERA SYSTEMS & MODELS - Canon Cinema EOS - DSLR-style cinema",
"CAMERA SYSTEMS & MODELS - Sony CineAlta - Professional digital",
"CAMERA SYSTEMS & MODELS - Light Iron - Post-production services",
"CAMERA SYSTEMS & MODELS - RED Weapon - High-performance digital",
"CAMERA SYSTEMS & MODELS - Canon C700 - Cinema DSLR",
"CAMERA SYSTEMS & MODELS - Blackmagic URSA Mini - Compact digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Mini - Portable digital",
"CAMERA SYSTEMS & MODELS - Sony FX9 - Versatile cinema camera",
"CAMERA SYSTEMS & MODELS - Canon C300 - Professional DSLR",
"CAMERA SYSTEMS & MODELS - Panasonic GH5 - Hybrid digital camera",
"CAMERA SYSTEMS & MODELS - RED Epic-W - Advanced digital",
"CAMERA SYSTEMS & MODELS - ARRI Amira - Mid-range cinema",
"CAMERA SYSTEMS & MODELS - Blackmagic Pocket Cinema - Compact digital",
"CAMERA SYSTEMS & MODELS - Canon EOS R5 - High-res mirrorless",
"CAMERA SYSTEMS & MODELS - Sony A7S III - Low-light digital",
"CAMERA SYSTEMS & MODELS - Nikon Z6 - Versatile mirrorless",
"CAMERA SYSTEMS & MODELS - Fujifilm X-T4 - Hybrid mirrorless",
"CAMERA SYSTEMS & MODELS - Olympus OM-D - Compact mirrorless",
"CAMERA SYSTEMS & MODELS - Leica M10 - Precision digital",
"CAMERA SYSTEMS & MODELS - Pentax K-1 - Full-frame DSLR",
"CAMERA SYSTEMS & MODELS - Sigma fp - Modular digital",
"CAMERA SYSTEMS & MODELS - Hasselblad X1D - Medium format digital",
"CAMERA SYSTEMS & MODELS - Phase One XF - High-end medium format",
"CAMERA SYSTEMS & MODELS - Panasonic Lumix S1H - Cinema-ready digital",
"CAMERA SYSTEMS & MODELS - Sony Alpha 1 - Ultimate high-res",
"CAMERA SYSTEMS & MODELS - Canon EOS R6 - Fast digital",
"CAMERA SYSTEMS & MODELS - Blackmagic URSA Broadcast - Live broadcast",
"CAMERA SYSTEMS & MODELS - ARRI Alexa LF - Large format digital",
"CAMERA SYSTEMS & MODELS - Sony FX6 - Compact cinema",
"CAMERA SYSTEMS & MODELS - Canon C500 - Full-frame digital",
"CAMERA SYSTEMS & MODELS - Panasonic AG-CX350 - Broadcast digital",
"CAMERA SYSTEMS & MODELS - Blackmagic Design URSA Broadcast - Live digital",
"CAMERA SYSTEMS & MODELS - RED Komodo - Compact digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Classic - Traditional digital",
"CAMERA SYSTEMS & MODELS - Sony Venice 2 - Enhanced digital",
"CAMERA SYSTEMS & MODELS - Canon EOS C700 FF - Full-frame cinema",
"CAMERA SYSTEMS & MODELS - Panasonic EVA1 - Lightweight cinema",
"CAMERA SYSTEMS & MODELS - Blackmagic Pocket Cinema Camera 6K - High-res compact",
"CAMERA SYSTEMS & MODELS - ARRI Alexa LF - Large format digital",
"CAMERA SYSTEMS & MODELS - RED DSMC2 - Modular digital",
"CAMERA SYSTEMS & MODELS - Sony PXW-FX9 - Versatile cinema",
"CAMERA SYSTEMS & MODELS - Canon C300 Mark III - Dual gain digital",
"CAMERA SYSTEMS & MODELS - Panasonic GH6 - Advanced hybrid",
"CAMERA SYSTEMS & MODELS - Blackmagic URSA Mini Pro 12K - Extreme resolution",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Mini LF - Large format portable",
"CAMERA SYSTEMS & MODELS - RED Monstro - Massive digital",
"CAMERA SYSTEMS & MODELS - Sony FX9 - Versatile cinema camera",
"CAMERA SYSTEMS & MODELS - Canon C500 Mark II - Modular digital",
"CAMERA SYSTEMS & MODELS - Panasonic AU-EVA1 - Professional digital",
"CAMERA SYSTEMS & MODELS - Blackmagic Pocket Cinema Camera 6K Pro - Enhanced compact",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Studio - Studio digital",
"CAMERA SYSTEMS & MODELS - RED Scarlet-W - Compact high-res",
"CAMERA SYSTEMS & MODELS - Sony Alpha 1 - All-in-one mirrorless",
"CAMERA SYSTEMS & MODELS - Canon EOS R5C - Cinema mirrorless",
"CAMERA SYSTEMS & MODELS - Panasonic Lumix DC-S1H - Full cinema digital",
"CAMERA SYSTEMS & MODELS - Blackmagic Design URSA Broadcast G2 - Enhanced digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Studio LF - Large format studio",
"CAMERA SYSTEMS & MODELS - RED Raven - Lightweight digital",
"CAMERA SYSTEMS & MODELS - Sony FX9 - Versatile cinema camera",
"CAMERA SYSTEMS & MODELS - Canon C300 Mark III - Dual gain digital",
"CAMERA SYSTEMS & MODELS - Panasonic GH6 - Advanced hybrid",
"CAMERA SYSTEMS & MODELS - Blackmagic URSA Mini Pro 12K - Extreme resolution",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Mini LF Plus - Portable large format",
"CAMERA SYSTEMS & MODELS - RED Epic Dragon 6K - High-end digital",
"CAMERA SYSTEMS & MODELS - Sony FX3 - Compact cinema mirrorless",
"CAMERA SYSTEMS & MODELS - Canon C70 - Lightweight cinema",
"CAMERA SYSTEMS & MODELS - Panasonic AG-DVX200 - Compact broadcast",
"CAMERA SYSTEMS & MODELS - Blackmagic URSA Mini Pro G3 - Advanced digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa XT - Traditional digital",
"CAMERA SYSTEMS & MODELS - RED Epic Dragon - High-end digital",
"CAMERA SYSTEMS & MODELS - Sony Venice Pro - Professional cinema",
"CAMERA SYSTEMS & MODELS - Canon EOS C700 FF - Full-frame cinema",
"CAMERA SYSTEMS & MODELS - Panasonic Lumix DC-S1H - Cinema-grade mirrorless",
"CAMERA SYSTEMS & MODELS - Blackmagic Design URSA Mini Pro 12K - Ultra high-res digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Classic Plus - Traditional enhanced digital",
"CAMERA SYSTEMS & MODELS - RED Scarlet-X - High-res compact",
"CAMERA SYSTEMS & MODELS - Sony FX3 - Compact cinema mirrorless camera",
"CAMERA SYSTEMS & MODELS - Canon C70 - Lightweight cinema camera",
"CAMERA SYSTEMS & MODELS - Panasonic Lumix DC-S1H - Cinema-grade mirrorless camera",
"CAMERA SYSTEMS & MODELS - Blackmagic Design URSA Mini Pro 12K - Ultra high-res digital camera",
"CAMERA SYSTEMS & MODELS - ARRI Alexa XT Plus - Traditional enhanced digital",
"CAMERA SYSTEMS & MODELS - RED Ranger Monstro 8K - Massive high-res digital",
"CAMERA SYSTEMS & MODELS - Sony FX6 - Compact cinema camera",
"CAMERA SYSTEMS & MODELS - Canon C500 Mark III - Advanced modular digital",
"CAMERA SYSTEMS & MODELS - Panasonic AU-EVA1 - Professional cinema digital",
"CAMERA SYSTEMS & MODELS - Blackmagic Pocket Cinema Camera 6K Pro - Enhanced compact digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Mini LF Plus - Portable large format digital",
"CAMERA SYSTEMS & MODELS - RED Epic Dragon 6K - High-end digital",
"CAMERA SYSTEMS & MODELS - Sony Alpha 1 - Ultimate all-in-one",
"CAMERA SYSTEMS & MODELS - Canon EOS C500 Mark II - Modular cinema digital",
"CAMERA SYSTEMS & MODELS - Panasonic GH6 - Advanced hybrid digital camera",
"CAMERA SYSTEMS & MODELS - Blackmagic Design URSA Mini Pro 12K - Extreme resolution digital",
"CAMERA SYSTEMS & MODELS - ARRI Alexa Mini LF Plus - Portable large format digital",
"CAMERA SYSTEMS & MODELS - RED Epic Dragon 6K - High-end digital",
"CAMERA SYSTEMS & MODELS - Sony FX3 - Compact cinema mirrorless camera",
"CAMERA SYSTEMS & MODELS - Canon C70 - Lightweight cinema camera",
"CAMERA SYSTEMS & MODELS - Panasonic Lumix DC-S1H - Cinema-grade mirrorless camera",
"CAMERA SYSTEMS & MODELS - Blackmagic Design URSA Mini Pro 12K - Ultra high-res digital camera",

# 🔍 LENSES & OPTICAL SYSTEMS
"LENSES & OPTICAL SYSTEMS - Anamorphic Format - Wide cinematic",
"LENSES & OPTICAL SYSTEMS - Cinemascope - Wide aspect ratio",
"LENSES & OPTICAL SYSTEMS - Ultra Panavision 70 - Massive widescreen",
"LENSES & OPTICAL SYSTEMS - ARRI Master Prime - Prime lenses",
"LENSES & OPTICAL SYSTEMS - Zeiss Compact Prime - High-quality lenses",
"LENSES & OPTICAL SYSTEMS - Cooke S4 - Cinematic lenses",
"LENSES & OPTICAL SYSTEMS - Leica Summicron - Precision optics",
"LENSES & OPTICAL SYSTEMS - ARRIFLEX - Professional cameras",
"LENSES & OPTICAL SYSTEMS - Panavision - Professional lenses",
"LENSES & OPTICAL SYSTEMS - ARRI Master Prime - Prime lenses",

# 🛠️ POST-PRODUCTION & EQUIPMENT
"POST-PRODUCTION & EQUIPMENT - Light Iron - Post-production services",
"POST-PRODUCTION & EQUIPMENT - D-Box - 3D seating integration",
"POST-PRODUCTION & EQUIPMENT - Kino Flo - Lighting systems",
"POST-PRODUCTION & EQUIPMENT - Tiffen Film - Specialty filters",

# 💾 FILE FORMATS & ENCODING
"FILE FORMATS & ENCODING - CINEon - Digital film",
"FILE FORMATS & ENCODING - Log Format - Dynamic range preservation",
"FILE FORMATS & ENCODING - Raw Format - Unprocessed data",

# 🖥️ SPECIALIZED FORMATS
"SPECIALIZED FORMATS - Virtual Reality Film - Immersive cinema",
"SPECIALIZED FORMATS - 360 Film - All-around view",
"SPECIALIZED FORMATS - Stereo 3D Film - Dual perspective",
"SPECIALIZED FORMATS - Flat 2D Film - Traditional format",
"SPECIALIZED FORMATS - Flat 3D Film - Single perspective",

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
"Wide Angle", "Telephoto", "Macro", "Fisheye", "Prime", "Zoom", "Anamorphic",
"Tilt-Shift", "Ultra Wide Angle", "Standard Lens", "Short Telephoto",
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
            text="COMBINE (Under Construction)",
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
        directory_name = f"{sanitized_concept}_{unique_id}_{timestamp}"
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
        that each set contains exactly one positive and one negative statement with content.

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

        for idx, prompt_set in enumerate(prompt_sets, start=1):
            prompt_set = prompt_set.strip()
            if not prompt_set:
                print(f"Prompt set {idx} is empty.")
                return False
            positive_match = re.search(r"positive:\s*(.+)", prompt_set, re.IGNORECASE | re.DOTALL)
            negative_match = re.search(r"negative:\s*(.+)", prompt_set, re.IGNORECASE | re.DOTALL)
            if not positive_match or not negative_match:
                print(f"Prompt set {idx} is missing 'positive' or 'negative' sections.")
                return False
            positive_content = positive_match.group(1).strip()
            negative_content = negative_match.group(1).strip()
            if not positive_content or not negative_content:
                print(f"Prompt set {idx} has empty 'positive' or 'negative' content.")
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
        In 'Story Mode', generate prompts sequentially, passing previous prompts as context.
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

        # Determine the number of prompts to generate
        num_prompts = self.video_prompt_number_var.get()

        # Gather all options set by the user
        video_options = {
            "theme": self.video_theme_var.get(),
            "art_style": self.video_art_style_var.get(),
            "lighting": self.video_lighting_var.get(),
            "framing": self.video_framing_var.get(),
            "camera_movement": self.video_camera_movement_var.get(),
            "shot_composition": self.video_shot_composition_var.get(),
            "time_of_day": self.video_time_of_day_var.get(),
            "decade": self.video_decade_var.get(),  # Keep this as a string
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
            f"Theme: {video_options['theme']}\n"
            f"Art Style: {video_options['art_style']}\n"
            f"Lighting: {video_options['lighting']}\n"
            f"Framing: {video_options['framing']}\n"
            f"Camera Movement: {video_options['camera_movement']}\n"
            f"Shot Composition: {video_options['shot_composition']}\n"
            f"Time of Day: {video_options['time_of_day']}\n"
            f"Decade: {video_options['decade']}\n"
            f"Camera: {video_options['camera']}, Lens: {video_options['lens']}\n"
            f"Resolution: {video_options['resolution']}\n"
        )

        # Add optional elements dynamically
        if video_options["wildlife_animal"]:
            options_context += f"Feature a {video_options['wildlife_animal']}.\n"

        if video_options["domesticated_animal"]:
            options_context += f"Include a {video_options['domesticated_animal']}.\n"

        if video_options["soundscape_mode"]:
            options_context += "Incorporate soundscapes relevant to the scene.\n"

        if video_options["holiday_mode"]:
            options_context += f"Apply holiday themes: {video_options['selected_holidays']}.\n"

        if video_options["no_people_mode"]:
            options_context += "Focus on the environment or animals, without human figures.\n"

        if video_options["chaos_mode"]:
            options_context += "Introduce chaotic elements that create tension or contrast in the visuals.\n"

        if video_options["remix_mode"]:
            options_context += "Add creative variations in visual styles or thematic choices.\n"

        # Retrieve selected camera and decade
        selected_camera = video_options['camera']
        selected_decade = video_options['decade']

        generated_prompts = []

        if video_options["story_mode"]:
            # Initialize previous prompt variable
            previous_prompt = ""

            for prompt_index in range(1, num_prompts + 1):
                while True:
                    try:
                        # Build the prompt with previous context
                        single_prompt = (
                            f"Generate a detailed video prompt that continues the narrative for the concept '{input_concept}' "
                            f"based on the {selected_decade}. Ensure that the prompt includes:\n"
                            f"Shot on a {selected_camera}.\n"
                            f"{options_context}"
                            f"Your prompt should follow from the previous scene and progress the story.\n"
                            f"Previous prompt: '{previous_prompt}'\n"
                            f"Provide the next prompt in the format:\n"
                            f"positive: [Your positive prompt]\n"
                            f"negative: [Your negative prompt]\n"
                        )

                        # Call the model to generate a single video prompt
                        raw_video_prompt = self.generate_prompts_via_ollama(single_prompt, 'video', 1)

                        if not raw_video_prompt:
                            raise Exception(f"No video prompt generated for prompt set {prompt_index}. Retrying...")

                        # Clean and format the prompt
                        cleaned_prompt = self.clean_prompt_text(raw_video_prompt)
                        formatted_prompt = self.remove_unwanted_headers(cleaned_prompt)

                        # Validate the generated prompt
                        if self.validate_prompts(formatted_prompt, 1):
                            generated_prompts.append(formatted_prompt)
                            previous_prompt = formatted_prompt  # Update previous prompt
                            print(f"Prompt {prompt_index} generated successfully.")
                            break  # Move to the next prompt set
                        else:
                            print(f"Validation failed for prompt {prompt_index}. Retrying...")
                            # Optionally, implement a retry limit here

                    except Exception as e:
                        print(f"Error generating video prompt {prompt_index}: {e}. Retrying...")
        else:
            # Generate prompts individually without story context
            for prompt_index in range(1, num_prompts + 1):
                while True:
                    try:
                        # Build the final prompt for this specific prompt set
                        single_prompt = (
                            f"Generate a detailed video prompt for the concept '{input_concept}' "
                            f"based on the {selected_decade}. Ensure that the prompt includes:\n"
                            f"Shot on a {selected_camera}.\n"
                            f"{options_context}"
                        )

                        # Call the model to generate a single video prompt
                        raw_video_prompt = self.generate_prompts_via_ollama(single_prompt, 'video', 1)

                        if not raw_video_prompt:
                            raise Exception(f"No video prompt generated for prompt set {prompt_index}. Retrying...")

                        # Clean and format the prompt
                        cleaned_prompt = self.clean_prompt_text(raw_video_prompt)
                        formatted_prompt = self.remove_unwanted_headers(cleaned_prompt)

                        # Validate the generated prompt
                        if self.validate_prompts(formatted_prompt, 1):
                            generated_prompts.append(formatted_prompt)
                            print(f"Prompt {prompt_index} generated successfully.")
                            break  # Move to the next prompt set
                        else:
                            print(f"Validation failed for prompt {prompt_index}. Retrying...")
                            # Optionally, implement a retry limit here

                    except Exception as e:
                        print(f"Error generating video prompt {prompt_index}: {e}. Retrying...")

        # After generating all prompts, save them
        try:
            directory, video_folder, audio_folder, video_filename, _ = self.create_smart_directory_and_filenames(input_concept)
            video_save_path = os.path.join(video_folder, video_filename)

            # Combine all prompts into a single string separated by the delimiter
            formatted_prompts = f"\n--------------------\n".join(generated_prompts)

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

    def generate_audio_prompts(self):
        """
        Generate detailed audio prompts based on the input concept entered by the user.
        Automatically retries each audio prompt generation step until valid prompts are obtained.
        Ensures the same number of audio prompts as video prompts without blanks.
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
            # Gather all video prompts
            video_prompt_sets = [p.strip() for p in self.video_prompts.split("--------------------") if p.strip()]
            if len(video_prompt_sets) != video_prompt_count:
                raise Exception("Mismatch in video prompt count during processing.")

            sonic_descriptions = []

            for i, video_prompt_set in enumerate(video_prompt_sets, start=1):
                while True:
                    try:
                        # Create a sound-specific prompt from the video prompt
                        sound_prompt = (
                            f"Generate a sonic description based on the following visual scene: '{video_prompt_set}'. "
                            "Focus only on the sounds, atmosphere, background noises, ambient audio, and specific sonic elements. "
                            "Do not describe visual elements. Describe what the listener would hear in this scene, including subtle "
                            "details like environmental sounds, echoes, footsteps, machinery, voices, or any relevant sonic cues."
                        )

                        # Send the sound prompt to Ollama to generate the sonic description
                        translated_sonic_prompt = self.generate_prompts_via_ollama(sound_prompt, 'audio', 1)  # Process one audio prompt at a time

                        if not translated_sonic_prompt:
                            raise Exception("No sonic description generated. Retrying...")

                        # Clean and store the sonic description
                        cleaned_sonic_prompt = clean_prompt_text(translated_sonic_prompt)
                        formatted_sonic_prompt = self.remove_unwanted_headers(cleaned_sonic_prompt)

                        # Validate the generated audio prompt
                        if self.validate_prompts(formatted_sonic_prompt, 1):
                            sonic_descriptions.append(formatted_sonic_prompt)
                            break  # Valid prompt obtained, move to the next
                        else:
                            print(f"Validation failed for audio prompt {i}. Retrying...")

                    except Exception as e:
                        print(f"Error generating audio prompt {i}: {e}. Retrying...")

            # Join the sonic descriptions together with the same format as video prompts
            formatted_sonic_prompts = "\n--------------------\n".join(sonic_descriptions)

            # Save and display the formatted audio prompts
            directory, _, audio_folder, _, audio_filename = self.create_smart_directory_and_filenames(input_concept)
            self.audio_save_folder = audio_folder
            audio_save_path = os.path.join(self.audio_save_folder, audio_filename)
            self.save_to_file(formatted_sonic_prompts, audio_save_path)

            # Store the audio prompts for later use
            self.audio_prompts = formatted_sonic_prompts
            enable_button(self.generate_sound_button)

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

        positive: Describe the positive aspects of the scene or shot in masterful visual detail including specific features.
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

        # Define a style for the ttk Progressbar
        style = ttk.Style()
        style.theme_use('clam')  # You can use any theme that supports progress bars
        style.configure("custom.Horizontal.TProgressbar", troughcolor='#0A2239', background='white')

        self.progress_bar = ttk.Progressbar(
            self.progress_window,
            orient='horizontal',
            length=300,
            mode='determinate',
            style="custom.Horizontal.TProgressbar"
        )
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


    def combine_media(self):
        """
        Combines the selected video file with the generated audio sound effects.
        Saves the combined media in the smart-named directory.
        """
        video_files = sorted(
            [os.path.join(self.video_save_folder, f) for f in os.listdir(self.video_save_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))],
            key=extract_number_from_filename
        )

        if not video_files:
            messagebox.showwarning("No Video Files", "No video files found in the Video folder.")
            return

        audio_files = sorted(
            [os.path.join(self.audio_save_folder, f) for f in os.listdir(self.audio_save_folder) if f.endswith(('.mp3', '.wav'))],
            key=extract_number_from_filename
        )

        if not audio_files:
            messagebox.showwarning("No Audio Files", "No audio files found in the Audio folder.")
            return

        try:
            video_clips = [VideoFileClip(video_file) for video_file in video_files]
            audio_clips = [AudioFileClip(audio_file) for audio_file in audio_files]

            # Concatenate video and audio clips
            final_video = concatenate_videoclips(video_clips)
            combined_audio = concatenate_audioclips(audio_clips)

            final_video = final_video.set_audio(combined_audio)

            # Define the save path within the master directory
            save_path = os.path.join(self.video_save_folder, "combined_media.mp4")

            final_video.write_videofile(save_path, codec="libx264", audio_codec="aac")
            messagebox.showinfo("Combine Successful", f"Media combined and saved to: {save_path}")

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
