from pydub import AudioSegment
import os
import tkinter.filedialog as filedialog
import tkinter as tk
import tkinter.ttk as ttk
import subprocess
import json
import requests
import re

OLLAMA_API_URL = "http://localhost:11434"
REQUIRED_MODEL = "llama3.2"

# Prompt user to select directory containing audio files
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    audio_directory = filedialog.askdirectory(title="Select Folder Containing Audio Files")
    return audio_directory

# Load the audio files and apply volume adjustments for diegetic sound
def load_and_adjust_audio(file_path, volume_db):
    print(f"Loading audio file: {file_path} with volume adjustment: {volume_db} dB")
    audio = AudioSegment.from_wav(file_path)
    audio = audio[:6000]  # Trim or pad each audio to exactly 6 seconds
    if len(audio) < 6000:
        audio = audio + AudioSegment.silent(duration=(6000 - len(audio)))
    return audio + volume_db

def ollama_check(audio_descriptions):
    # Use Ollama to analyze the content and provide volume suggestions for an entire set of descriptions
    system_prompt = f"""
    You are an AI assistant tasked with analyzing the audio content descriptions and suggesting relative volume levels for a cohesive soundscape. 
    Given the following audio descriptions, suggest volume adjustments in dB for each sound, considering its relevance, distance to the camera, and how prominently it should feature in the final mix:
    {audio_descriptions}
    Return a JSON object with each audio description as the key and the suggested volume level in dB as the value.
    """
    api_url = f"{OLLAMA_API_URL}/api/generate"
    payload = {
        "model": REQUIRED_MODEL,
        "prompt": system_prompt,
        "max_tokens": 150,
        "stream": False
    }
    headers = {'Content-Type': 'application/json'}
    try:
        print(f"Requesting volume suggestions from Ollama for multiple audio descriptions.")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            analysis = json.loads(response.text.strip())
            return analysis
        else:
            raise Exception(f"Ollama API returned an error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error generating volume suggestion via Ollama: {e}")
        return {}  # Return empty dictionary if Ollama fails

def create_soundscape(audio_directory, volume_settings):
    # Create an empty AudioSegment of 6 seconds silence to accumulate all layers
    soundscape = AudioSegment.silent(duration=6000)

    # Get all audio files in the selected directory
    audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]
    audio_files.sort()  # Sort files alphabetically to maintain order

    for idx, file_name in enumerate(audio_files):
        file_path = os.path.join(audio_directory, file_name)

        # Get user-adjusted volume if available
        suggested_volume_db = volume_settings.get(file_name, -15)
        audio = load_and_adjust_audio(file_path, suggested_volume_db)

        # Mix each audio layer with the soundscape to build a final composite
        print(f"Mixing {file_name} into the final soundscape.")
        soundscape = soundscape.overlay(audio)

    print("Soundscape creation complete.")
    return soundscape

def extract_sound_name(file_name):
    # Extract the sound effect description from the file name, removing steps or any prefix information
    match = re.search(r'SonicLayer\d+_(.*)\.wav', file_name)
    if match:
        sound_name = match.group(1)
        sound_name = re.sub(r'[_\-]', ' ', sound_name)
        return sound_name.strip()
    else:
        # Default to entire file name without extension if pattern does not match
        return re.sub(r'[_\-]', ' ', os.path.splitext(file_name)[0]).strip()

def get_volume_settings(audio_files, audio_directory):
    volume_settings = {}

    # Extract sound descriptions for each file
    audio_descriptions = {file_name: extract_sound_name(file_name) for file_name in audio_files}

    # Request volume suggestions for all audio descriptions at once
    ollama_suggestions = ollama_check(audio_descriptions)

    # Create popup window to show sliders for each audio file
    root = tk.Tk()
    root.title("Adjust Volume Levels")

    sliders = {}
    for file_name in audio_files:
        sound_name = extract_sound_name(file_name)
        frame = ttk.Frame(root)
        frame.pack(fill=tk.X, padx=5, pady=5)

        label = ttk.Label(frame, text=sound_name)
        label.pack(side=tk.LEFT, padx=5)

        suggested_volume = ollama_suggestions.get(sound_name, -15)
        slider = ttk.Scale(frame, from_=-30, to=10, orient=tk.HORIZONTAL)
        slider.set(suggested_volume)  # Set default volume level using Ollama suggestion
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        sliders[file_name] = slider

    def submit():
        for file_name, slider in sliders.items():
            volume_settings[file_name] = slider.get()
        root.quit()

    submit_button = ttk.Button(root, text="Submit", command=submit)
    submit_button.pack(pady=10)

    root.mainloop()
    root.destroy()

    return volume_settings

def main():
    # Select directory containing audio files
    audio_directory = select_directory()
    if not audio_directory:
        print("No directory selected. Exiting.")
        return

    # Get all audio files in the selected directory
    audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]
    audio_files.sort()  # Sort files alphabetically to maintain order

    if not audio_files:
        print("No valid audio files found in the selected directory. Exiting.")
        return

    # Get volume settings from the user
    print("Opening volume adjustment interface...")
    volume_settings = get_volume_settings(audio_files, audio_directory)
    
    # Create the soundscape
    print("Creating the soundscape...")
    soundscape = create_soundscape(audio_directory, volume_settings)
    
    # Check if soundscape is created
    if soundscape is None:
        print("No valid audio files found in the selected directory. Exiting.")
        return
    
    # Export the final soundscape to a file in the selected directory
    output_path = os.path.join(audio_directory, "combined_soundscape.wav")
    print(f"Exporting soundscape to: {output_path}")
    soundscape.export(output_path, format="wav")
    print(f"Soundscape created and saved to {output_path}")

if __name__ == "__main__":
    main()