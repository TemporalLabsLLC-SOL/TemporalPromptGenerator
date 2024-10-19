import os
import shutil
import re
import subprocess
from tkinter import Tk
from tkinter import filedialog

# Function to overlay text onto the video using ffmpeg
def overlay_text_on_video(video_path, output_path, overlay_text):
    ffmpeg_command = [
        "ffmpeg", "-i", video_path, "-vf",
        f"drawtext=text='{overlay_text}':fontcolor=white:fontsize=24:x=(w-tw)/2:y=h-th-10",
        "-codec:a", "copy", output_path
    ]
    
    # Run ffmpeg command
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Created video with overlay: {output_path}")

# Function to organize videos and subtitles, and generate overlay videos
def organize_videos_in_place(base_dir):
    # Broader regex pattern to match cfg and steps more flexibly
    pattern = r'video_\d+_(\d+b)_(\d+\.\d+gs)_(\d+steps)_.+\.mp4'

    # Loop through all files in the base directory
    for filename in os.listdir(base_dir):
        if filename.endswith('.mp4'):  # Only process video files
            # Match the pattern to extract cfg, steps, etc.
            match = re.match(pattern, filename)
            if match:
                cfg = match.group(2)  # Get the cfg value
                steps = match.group(3)  # Get the steps value
                
                # Create the directory for this cfg if it doesn't exist
                cfg_dir = os.path.join(base_dir, cfg)
                if not os.path.exists(cfg_dir):
                    os.makedirs(cfg_dir)
                
                # Create the directory for this steps count inside the cfg directory
                steps_dir = os.path.join(cfg_dir, steps)
                if not os.path.exists(steps_dir):
                    os.makedirs(steps_dir)
                
                # Move the video file to the appropriate folder
                source_video = os.path.join(base_dir, filename)
                destination_video = os.path.join(steps_dir, filename)
                shutil.move(source_video, destination_video)
                print(f"Moved {filename} to {destination_video}")
                
                # Look for the corresponding SRT file (replace .mp4 with .srt)
                srt_filename = filename.replace(".mp4", ".srt")
                source_srt = os.path.join(base_dir, srt_filename)
                if os.path.exists(source_srt):
                    destination_srt = os.path.join(steps_dir, srt_filename)
                    shutil.move(source_srt, destination_srt)
                    print(f"Moved {srt_filename} to {destination_srt}")
                else:
                    print(f"Warning: No corresponding SRT found for {filename}")

                # Create a second copy of the video with the overlay text
                overlay_text = f"CFG: {cfg}, Steps: {steps}"
                output_video_with_overlay = os.path.join(steps_dir, f"overlay_{filename}")
                overlay_text_on_video(destination_video, output_video_with_overlay, overlay_text)
            else:
                print(f"Filename {filename} does not match expected pattern.")

    # Second pass: Ensure all .srt files are moved, even if the video was already moved
    for filename in os.listdir(base_dir):
        if filename.endswith('.srt'):  # Only process SRT files
            # Find matching .mp4 video in target folders
            video_filename = filename.replace(".srt", ".mp4")
            found = False
            
            # Search for the video in the organized folders
            for root, dirs, files in os.walk(base_dir):
                if video_filename in files:
                    # Found the matching video in the target folder
                    destination_srt = os.path.join(root, filename)
                    source_srt = os.path.join(base_dir, filename)
                    shutil.move(source_srt, destination_srt)
                    print(f"Moved {filename} to {destination_srt}")
                    found = True
                    break
            
            if not found:
                print(f"Warning: No corresponding video found for {filename}. SRT remains in the source folder.")

    print("Organization complete.")

# Function to open a folder selection dialog
def select_folder():
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()
    return folder_selected

# Main script
if __name__ == "__main__":
    # Prompt the user to select the folder containing the videos
    print("Please select the folder containing the videos and/or subtitles to organize.")
    base_dir = select_folder()
    
    if base_dir:
        # Call the function to organize videos and subtitles
        organize_videos_in_place(base_dir)
    else:
        print("No folder selected. Exiting.")
