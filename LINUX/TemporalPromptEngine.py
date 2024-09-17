import os
import sys
import datetime
import json
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
from elevenlabs.client import ElevenLabs
import re
import random


# Load environment variables from .env file
load_dotenv()

# Initialize global variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")
COMFYUI_PROMPTS_FOLDER = os.getenv("COMFYUI_PROMPTS_FOLDER")
MAX_CHAR_LIMIT = 400  # Maximum characters allowed for prompts
MAX_PROMPTS = 250  # Maximum number of prompt sets allowed
DEFAULT_PROMPTS = 10  # Default number of prompt sets
LAST_USED_DIRECTORY = os.getenv("LAST_USED_DIRECTORY") or os.getcwd()

SETTINGS_FILE = "settings.json"

# Save settings to .env file for persistence
def save_settings():
    with open(".env", "w") as f:
        f.write(f"ELEVENLABS_API_KEY={ELEVENLABS_API_KEY}\n")
        f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}\n")
        f.write(f"OUTPUT_DIRECTORY={OUTPUT_DIRECTORY}\n")
        f.write(f"COMFYUI_PROMPTS_FOLDER={COMFYUI_PROMPTS_FOLDER}\n")
        f.write(f"LAST_USED_DIRECTORY={LAST_USED_DIRECTORY}\n")
    messagebox.showinfo("Settings Saved", "Settings have been saved to .env file.")
    
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

# Function to initialize ElevenLabs API client
# Function to initialize ElevenLabs API client
def initialize_elevenlabs_client():
    global elevenlabs_client, ELEVENLABS_API_KEY
    if not ELEVENLABS_API_KEY:
        ELEVENLABS_API_KEY = simpledialog.askstring("ElevenLabs API Key", "Please enter your ElevenLabs API Key:", show='*')
        if ELEVENLABS_API_KEY:
            save_settings()  # Save the key to the .env file
        else:
            messagebox.showerror("Error", "API key is required to proceed. Exiting...")
            sys.exit(1)

    print(f"Using ElevenLabs API key: {ELEVENLABS_API_KEY}")  # Debugging line
    try:
        # Initialize the ElevenLabs client here
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        return elevenlabs_client
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Failed to initialize ElevenLabs client: {e}")
        sys.exit(1)

def initialize_openai_client():
    global OPENAI_API_KEY, OUTPUT_DIRECTORY
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = simpledialog.askstring("OpenAI API Key", "Please enter your OpenAI API Key:", show='*')
        if OPENAI_API_KEY:
            save_settings()
        else:
            messagebox.showerror("Error", "API key is required to proceed. Exiting...")
            sys.exit(1)
    if not OUTPUT_DIRECTORY:
        messagebox.showinfo("Set Output Directory", "Please select an output directory.")
        set_output_directory()
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        messagebox.showerror("OpenAI Initialization Error", f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)

# Function to set the output directory
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

# Function to generate prompts using OpenAI
def generate_prompts_via_openai(input_concept, prompt_type, number_of_prompts, video_prompts=None, options=None):
    initialize_openai_client()

    # Load video or audio options from settings file
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            saved_options = json.load(f)
    else:
        saved_options = {}

    # Update options with saved options
    if options is None:
        options = saved_options.get('video_options', {}) if prompt_type == 'video' else saved_options.get('audio_options', {})

    # Construct details string based on options
    details = ""
    if options:
        theme = options.get('theme')
        art_style = options.get('art_style')
        lighting = options.get('lighting')
        framing = options.get('framing')
        camera_movement = options.get('camera_movement')
        shot_composition = options.get('shot_composition')
        time_of_day = options.get('time_of_day')
        decade = options.get('decade')
        camera = options.get('camera')
        lens = options.get('lens')
        resolution = options.get('resolution')
        wildlife_animal = options.get('wildlife_animal', "")
        domesticated_animal = options.get('domesticated_animal', "")
        chaos_mode = options.get('chaos_mode', False)
        time_traveler_mode = options.get('time_traveler_mode', False)
        uhd = options.get('uhd', False)
        soundscape_mode = options.get('soundscape_mode', False)
        holiday_mode = options.get('holiday_mode', False)
        selected_holidays = options.get('selected_holidays', [])
        specific_modes = options.get('specific_modes', [])
        no_people_mode = options.get('no_people_mode', False)
        test_mode = options.get('test_mode', False)

        details = (
            f"Theme: {theme}; Art Style: {art_style}; Lighting: {lighting}; Framing: {framing}; "
            f"Camera Movement: {camera_movement}; Shot Composition: {shot_composition}; "
            f"Time of Day: {time_of_day}; Decade: {decade}; Camera: {camera}; Lens: {lens}; "
            f"Resolution: {resolution}; Wildlife Animal: {wildlife_animal}; Domesticated Animal: {domesticated_animal}."
        )

    # Define the system prompt based on the prompt_type
    if prompt_type == 'video':
        system_prompt = f"""
You are an AI assistant tasked with generating optimized prompts for video generation models.
Generate {number_of_prompts} sets of positive and negative prompts based on the following concept: '{input_concept}'.
Incorporate the following details to enrich the scene: {details}.
"""

        # Holiday Mode: Add specific atmospheric holiday-related elements
        if options.get('holiday_mode') and options.get('selected_holidays'):
            holidays_formatted = ', '.join(options['selected_holidays'])
            system_prompt += f"Integrate elements typical of the following holidays: {holidays_formatted}. Include seasonal environmental effects like snow, bells, or eerie wind.\n"

        # Specific Modes
        for mode in options.get('specific_modes', []):
            system_prompt += f"Incorporate elements of {mode}.\n"

        # SoundScape Mode: Focus solely on environmental elements
        if options.get('soundscape_mode'):
            system_prompt += "Focus on natural and environmental elements, including sounds like wind, water, and wildlife. Avoid any references to music or vocals.\n"

        # Chaos Mode: Regular generation, followed by inversion of prompts
        if chaos_mode:
            system_prompt += "Generate regular positive and negative prompts. Once complete, invert the positive and negative prompts for a chaotic scene.\n"

        # No People Mode: Focus entirely on non-human elements
        if no_people_mode:
            system_prompt += "Focus on non-human elements such as nature, objects, or animals, without referencing humans.\n"

        # Test Mode: Generate two contrasting prompts
        if test_mode:
            system_prompt += "Generate two contrasting prompts with differing lighting, camera movement, and actions.\n"

        system_prompt += f"""
Each prompt must be no more than {MAX_CHAR_LIMIT} characters.
The prompts should be formatted exactly as the examples below, separated by '--------------------'.

Example format:

positive: a detailed positive prompt1
negative: a detailed negative prompt1
--------------------
positive: a detailed positive prompt2
negative: a detailed negative prompt2
--------------------
positive: a detailed positive prompt{number_of_prompts}
negative: a detailed negative prompt{number_of_prompts}
"""

        user_prompt = f"Generate the video prompts based on the concept: '{input_concept}'."

    elif prompt_type == 'audio':
        if not video_prompts:
            messagebox.showerror("Audio Prompt Error", "Video prompts are required to generate audio prompts.")
            return None

        system_prompt = f"""
You are an AI assistant that generates rich, detailed soundscape prompts for audio generation models.
Based on the following video prompts, generate {number_of_prompts} audio prompts that complement and enhance the scenes.
Incorporate the following details to create layered and immersive audio environments: {details}.
"""

        # Holiday Mode: Add atmospheric holiday-related sound effects
        if options.get('holiday_mode') and options.get('selected_holidays'):
            holidays_formatted = ', '.join(options['selected_holidays'])
            system_prompt += f"Include atmospheric sounds associated with the following holidays: {holidays_formatted}. For example, soft bells for Christmas or eerie wind for Halloween.\n"

        # SoundScape Mode: Focus entirely on environmental and ambient sounds
        if options.get('soundscape_mode'):
            system_prompt += "Focus on ambient natural sounds such as wind, water, and wildlife. Ensure the soundscape is immersive and realistic.\n"

        # Story Mode: Maintain cohesiveness across prompts
        if options.get('story_mode', False):
            system_prompt += "Ensure consistency and cohesiveness across all prompts, maintaining a unified atmosphere.\n"

        system_prompt += f"""
Each prompt must be no more than {MAX_CHAR_LIMIT} characters.
The prompts should be formatted as shown below, separated by '--------------------'.

Example format:

a detailed audio prompt1
--------------------
a detailed audio prompt2
--------------------
a detailed audio prompt{number_of_prompts}
"""

        user_prompt = f"Using the following positive video prompts, generate matching audio prompts:\n\n{video_prompts}\n\nGenerate the audio prompts."

    else:
        messagebox.showerror("Error", "Invalid prompt type.")
        return None

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        generated_prompts = completion.choices[0].message.content.strip()

        # If chaos mode is active and it's a video prompt, invert prompts
        if prompt_type == 'video' and options.get('chaos_mode'):
            generated_prompts = invert_positive_negative(generated_prompts)

        return generated_prompts

    except Exception as e:
        print(f"Error generating prompts via OpenAI: {e}")
        messagebox.showerror("OpenAI Error", f"An error occurred while generating prompts: {e}")
        return None        
def randomize_options_for_remix_mode(options):
    """
    Randomizes the selected parameters for each prompt when Remix Mode is enabled.
    """
    randomized_options = options.copy()

    # Randomize only the selected parameters
    if self.selected_remix_params.get("Theme"):
        randomized_options["theme"] = random.choice(THEMES)
    if self.selected_remix_params.get("Art Style"):
        randomized_options["art_style"] = random.choice(ART_STYLES)
    if self.selected_remix_params.get("Lighting"):
        randomized_options["lighting"] = random.choice(LIGHTING_OPTIONS)
    if self.selected_remix_params.get("Framing"):
        randomized_options["framing"] = random.choice(FRAMING_OPTIONS)
    if self.selected_remix_params.get("Camera Movement"):
        randomized_options["camera_movement"] = random.choice(CAMERA_MOVEMENTS)
    if self.selected_remix_params.get("Shot Composition"):
        randomized_options["shot_composition"] = random.choice(SHOT_COMPOSITIONS)
    if self.selected_remix_params.get("Time of Day"):
        randomized_options["time_of_day"] = random.choice(TIME_OF_DAY_OPTIONS)
    if self.selected_remix_params.get("Decade"):
        randomized_options["decade"] = random.choice(DECADES)
        randomized_options["camera"] = random.choice(CAMERAS[randomized_options["decade"]])
    if self.selected_remix_params.get("Lens"):
        randomized_options["lens"] = random.choice(LENSES)
    if self.selected_remix_params.get("Resolution"):
        randomized_options["resolution"] = random.choice(RESOLUTIONS)
    if self.selected_remix_params.get("Wildlife Animal"):
        randomized_options["wildlife_animal"] = random.choice(WILDLIFE_ANIMALS)
    if self.selected_remix_params.get("Domesticated Animal"):
        randomized_options["domesticated_animal"] = random.choice(DOMESTICATED_ANIMALS)

    # Return the new randomized options for this prompt
    return randomized_options
        
def invert_positive_negative(generated_prompts):
    """
    Invert the positive and negative labels in the generated prompts.
    This is used for Chaos Mode.
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

def generate_sound_effect(text: str, output_path: str, duration: int):
    """
    Generate a sound effect using the ElevenLabs API based on a given prompt and save it to the specified filename.
    """
    print(f"Generating sound effect for: {text}")
    
    try:
        # Use the global ElevenLabs client to convert text to sound effect
        result = elevenlabs_client.text_to_sound_effects.convert(
            text=text,
            duration_seconds=duration,  # Use the user-defined duration
            prompt_influence=0.3  # Optional, adjust influence of the prompt as needed
        )
        
        # Check if the result contains data
        if not result:
            print(f"Error: No data returned for prompt '{text}'")
            return

        # Save the resulting sound file
        with open(output_path, "wb") as f:
            for chunk in result:
                f.write(chunk)
        print(f"Audio saved to {output_path}")

    except Exception as e:
        print(f"Failed to generate sound effect for '{text}'. Error: {e}")
        messagebox.showerror("ElevenLabs API Error", f"Failed to generate sound effect for '{text}'. Error: {e}")
            
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

# Define options for dropdowns outside the class
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

RESOLUTIONS = [
    # Including historical resolutions
    "Standard Definition", "720p HD", "1080p Full HD", "2K", "4K UHD", "8K UHD",
    "IMAX Digital", "Custom Resolution"
]

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
# Example to fetch camera models based on a specific decade
selected_decade = "2020s"
print(CAMERAS[selected_decade])

# Decades list for UI or sorting purposes
DECADES = sorted(CAMERAS.keys())

FRAME_RATES = [
    # Include advanced frame rates
    "24fps", "25fps", "30fps", "48fps", "50fps", "60fps", "120fps", "240fps",
    "Variable Frame Rate", "High-Speed Recording", "Undercranking", "Overcranking"
]

class MultimediaSuiteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Temporal Labs LLC - Multimedia Suite")
        self.root.configure(bg='#0A2239')

        self.video_prompts = ""
        self.audio_prompts = ""
        self.duration = 10  # Default duration for sound effects
        self.video_prompt_file_path = ""
        self.audio_prompt_file_path = ""
        self.output_folder = ""
        self.video_options_set = False
        self.audio_options_set = False
        self.last_used_directory = LAST_USED_DIRECTORY

        self.build_gui()

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
        settings_menu.add_command(label="Set ElevenLabs API Key", command=self.set_elevenlabs_api_key)
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

        # Footer Frame containing Crypto Donations and Logo
        self.footer_frame = tk.Frame(self.root, bg='#0A2239')
        self.footer_frame.grid(row=7, column=0, sticky='ew', padx=20, pady=10)
        self.footer_frame.columnconfigure(0, weight=1)
        self.footer_frame.columnconfigure(1, weight=1)

        # Crypto Donations Frame
        self.crypto_frame = tk.Frame(self.footer_frame, bg='#0A2239')
        self.crypto_frame.grid(row=0, column=0, sticky='w', padx=10, pady=5)

        crypto_label = tk.Label(
            self.crypto_frame,
            text="Crypto Donations:",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        crypto_label.grid(row=0, column=0, columnspan=2, pady=(0,10))

        crypto_addresses = {
            "BTC": "1A1zp1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "ETH": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "LTC": "ltc1qwlyjz8aymy9uagqhht5a4kaq06kmv58dxlzyww",
            "SOL": "FVPGxfGT7QWfQEWvXpFkwdgiiKFM3VdvzNG6mEmX8pgi",
            "DOGE": "DAeWAroHCy8nXCoUsobderPRSNXNu1WY34"
        }

        crypto_logos = {
            "BTC": "https://cryptologos.cc/logos/bitcoin-btc-logo.png",
            "ETH": "https://cryptologos.cc/logos/ethereum-eth-logo.png",
            "LTC": "https://cryptologos.cc/logos/litecoin-ltc-logo.png",
            "SOL": "https://cryptologos.cc/logos/solana-sol-logo.png",
            "DOGE": "https://cryptologos.cc/logos/dogecoin-doge-logo.png"
        }

        def create_crypto_button(crypto, address, logo_url, row):
            try:
                response = requests.get(logo_url)
                response.raise_for_status()
                logo_image = Image.open(BytesIO(response.content)).resize((40, 40), Image.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_image)
                button = tk.Button(
                    self.crypto_frame,
                    image=logo_photo,
                    command=lambda addr=address: copy_to_clipboard(addr),
                    bg='#0A2239',
                    fg='white',
                    borderwidth=0,
                    activebackground='#0A2239',
                    cursor="hand2"
                )
                button.image = logo_photo
                button.grid(row=row, column=0, padx=5, pady=5)
                label = tk.Label(
                    self.crypto_frame,
                    text=crypto,
                    bg='#0A2239',
                    fg='white',
                    font=('Helvetica', 12)
                )
                label.grid(row=row, column=1, padx=5, pady=5)
            except Exception as e:
                print(f"Failed to create crypto button for {crypto}: {e}")
                messagebox.showerror("Crypto Button Error", f"Failed to load logo for {crypto}: {e}")

        for i, (crypto, address) in enumerate(crypto_addresses.items()):
            create_crypto_button(crypto, address, crypto_logos[crypto], i+1)

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

    def set_elevenlabs_api_key(self):
        global ELEVENLABS_API_KEY
        ELEVENLABS_API_KEY = simpledialog.askstring("Set ElevenLabs API Key", "Enter your ElevenLabs API Key:", show="*")
        if ELEVENLABS_API_KEY:
            save_settings()
            

    def create_smart_directory_and_filenames(self, input_concept):
        """
        Creates a smart-named directory based on the input concept and current timestamp.
        Generates smart filenames for video and audio prompts.
        
        Returns:
            directory_path (str): Path to the created directory.
            video_filename (str): Smart filename for video prompts.
            audio_filename (str): Smart filename for audio prompts.
        """
        # Sanitize the input concept to create a valid directory name
        sanitized_concept = re.sub(r'[^\w\s-]', '', input_concept).strip().replace(' ', '_')
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory name
        directory_name = f"{sanitized_concept}_{timestamp}"
        directory_path = os.path.join(OUTPUT_DIRECTORY, directory_name)
        
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)
        
        # Create smart filenames
        video_filename = f"{sanitized_concept}_video_prompts.txt"
        audio_filename = f"{sanitized_concept}_audio_prompts.txt"
        
        return directory_path, video_filename, audio_filename

            
    def show_video_prompt_options(self):
        self.video_options_window = tk.Toplevel(self.root)
        self.video_options_window.title("Video Prompt Options")
        self.video_options_window.configure(bg='#0A2239')

        self.build_video_options(self.video_options_window)

    def show_audio_prompt_options(self):
        self.audio_options_window = tk.Toplevel(self.root)
        self.audio_options_window.title("Audio Prompt Options")
        self.audio_options_window.configure(bg='#0A2239')

        self.build_audio_options(self.audio_options_window)
    def generate_video_prompts(self):
        """
        Generates video prompts based on user input and selected cinematic options.
        Displays the prompts in the output_text widget and saves them to a smart-named file.
        """
        # Check if the video options are set
        if not self.video_options_set:
            messagebox.showerror("Options Not Set", "Please set the video prompt options before generating prompts.")
            return

        # Retrieve the input concept from the text box
        input_concept = self.input_text.get("1.0", tk.END).strip()

        if len(input_concept) == 0 or len(input_concept) > MAX_CHAR_LIMIT:
            messagebox.showerror("Input Error", f"The prompt must be between 1 and {MAX_CHAR_LIMIT} characters.")
            return

        # Retrieve the number of prompts
        number_of_prompts = self.video_prompt_number_var.get()

        # Collect the cinematic options from the UI selections
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
            "story_mode": self.video_story_mode_var.get(),
            "chaos_mode": self.video_chaos_mode_var.get()
        }

        # Use the base directory and save the video prompts in the "Video" folder
        directory_path = self.video_save_folder

        # Create smart filenames for saving video prompts
        sanitized_concept = re.sub(r'[^\w\s-]', '', input_concept).strip().replace(' ', '_')
        video_filename = f"{sanitized_concept}_video_prompts.txt"
        video_save_path = os.path.join(directory_path, video_filename)

        # Call OpenAI API to generate video prompts
        try:
            self.video_prompts = generate_prompts_via_openai(
                input_concept, 'video', number_of_prompts, options=cinematic_options
            )

            if not self.video_prompts:
                raise Exception("No prompts generated. Please try again.")

        except Exception as e:
            messagebox.showerror("Prompt Generation Error", f"Failed to generate video prompts: {e}")
            return

        # Save the generated video prompts to the smart-named file
        try:
            with open(video_save_path, "w", encoding="utf-8") as file:
                file.write(self.video_prompts)

            # Enable the audio prompts generation button after successful video prompt generation
            enable_button(self.generate_audio_prompts_button)

            # Display the generated prompts in the output_text widget
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Generated Video Prompts:\n\n" + self.video_prompts)

        except Exception as e:
            messagebox.showerror("File Save Error", f"Failed to save video prompts: {e}")
            disable_button(self.generate_audio_prompts_button)

    def generate_audio_prompts(self):
        """
        Generates audio prompts based on previously generated video prompts and selected audio options.
        Displays the prompts in the output_text widget and saves them to a smart-named file.
        """
        if not self.audio_options_set:
            messagebox.showerror("Options Not Set", "Please set the audio prompt options before generating prompts.")
            return

        number_of_prompts = self.video_prompt_number_var.get()

        # Retrieve the input concept from the text box
        input_concept = self.input_text.get("1.0", tk.END).strip()

        if len(input_concept) == 0 or len(input_concept) > MAX_CHAR_LIMIT:
            messagebox.showerror("Input Error", f"The prompt must be between 1 and {MAX_CHAR_LIMIT} characters.")
            return

        # Ensure video prompts are generated
        if not self.video_prompts:
            messagebox.showerror("Video Prompts Missing", "Please generate video prompts before generating audio prompts.")
            return

        # Extract positive prompts from video prompts
        video_prompt_pairs = self.video_prompts.split('--------------------')
        positive_prompts = []
        for pair in video_prompt_pairs:
            if "positive:" in pair:
                positive = pair.split("positive:", 1)[1].split("\n", 1)[0].strip()
                positive_prompts.append(positive)

        if not positive_prompts:
            messagebox.showerror("Extraction Error", "No positive prompts found in video prompts.")
            return

        # Join positive prompts into a single string separated by '--------------------'
        positive_prompts_str = '--------------------\n'.join(positive_prompts)

        # Collect the audio options from the UI selections
        audio_options = {
            "exclude_music": self.audio_exclude_music_var.get(),
            "holiday_mode": self.audio_holiday_mode_var.get(),
            "selected_holidays": [self.audio_holidays_var.get()] if self.audio_holiday_mode_var.get() else [],
            "specific_modes": [mode for mode, var in self.audio_specific_modes_vars.items() if var.get()],
            "layer_intensity": self.audio_layer_intensity_var.get(),
            "story_mode": self.audio_story_mode_var.get(),
            "chaos_mode": self.audio_chaos_mode_var.get()
        }

        # Create smart filenames for saving audio prompts
        sanitized_concept = re.sub(r'[^\w\s-]', '', input_concept).strip().replace(' ', '_')
        audio_filename = f"{sanitized_concept}_audio_prompts.txt"
        audio_save_path = os.path.join(self.audio_save_folder, audio_filename)

        # Call OpenAI API to generate audio prompts based on positive video prompts
        try:
            self.audio_prompts = generate_prompts_via_openai(
                input_concept, 'audio', number_of_prompts, video_prompts=positive_prompts_str, options=audio_options
            )

            if not self.audio_prompts:
                raise Exception("No audio prompts generated. Please try again.")

        except Exception as e:
            messagebox.showerror("Prompt Generation Error", f"Failed to generate audio prompts: {e}")
            return

        # Save the generated audio prompts to the smart-named file
        try:
            with open(audio_save_path, "w", encoding="utf-8") as file:
                file.write(self.audio_prompts)

            # Enable the sound effects generation button after successful audio prompt generation
            enable_button(self.generate_sound_button)

            # Display the generated prompts in the output_text widget
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Generated Audio Prompts Based on Positive Video Prompts:\n\n" + self.audio_prompts)

        except Exception as e:
            messagebox.showerror("File Save Error", f"Failed to save audio prompts: {e}")
            disable_button(self.generate_sound_button)

    def generate_audio_prompts_from_video_file(self, video_file_path, audio_file_path, output_directory, options=None):
        """
        Reads the video prompts from a .txt file, generates corresponding audio prompts for each video prompt,
        and saves them to a new .txt file in the same order.
        """
        try:
            # Read video prompts from the file
            with open(video_file_path, 'r', encoding='utf-8') as video_file:
                video_prompts = video_file.read().split('--------------------')

            # Read audio prompts from the file
            with open(audio_file_path, 'r', encoding='utf-8') as audio_file:
                audio_prompts = audio_file.read().split('--------------------')

            if len(video_prompts) != len(audio_prompts):
                raise ValueError("The number of video and audio prompts does not match!")

            matched_prompts = []
            
            # Match each video prompt with its corresponding audio prompt
            for i, (video_prompt, audio_prompt) in enumerate(zip(video_prompts, audio_prompts)):
                matched_prompts.append(f"Video Prompt {i+1}: {video_prompt.strip()}\nAudio Prompt {i+1}: {audio_prompt.strip()}")

            # Save the matched prompts to a new file
            output_path = os.path.join(output_directory, "matched_video_audio_prompts.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(matched_prompts))

            print(f"Matched prompts saved to {output_path}")

        except Exception as e:
            print(f"Error while generating matched prompts: {e}")
            
    def generate_matching_audio_prompt(self, video_prompt, options):
        """
        Generates an audio prompt based on the given video prompt, ensuring alignment with the visual elements.
        This function is designed to correlate audio with the video prompt and optimize it for ElevenLabs API sound effect generation.
        """

        # Base prompt for correlated audio generation with a focus on desired sound elements
        system_prompt = f"""
        You are generating soundscapes that match the visual description in the video. Focus on environmental and atmospheric sounds that complement the scene:
        - Wind, water, and nature-based sounds (leaves rustling, streams, birds)
        - Urban or rural sounds (traffic hum, distant machinery, footsteps)

        Video Prompt: '{video_prompt}'
        """

        # If soundscape mode is enabled, focus entirely on environmental and ambient sounds
        if options.get('soundscape_mode'):
            system_prompt += """
            In soundscape mode, focus only on natural and ambient environmental sounds:
            - Wind moving through forests or fields
            - Flowing water (streams, rivers, rain)
            - Urban sounds like distant traffic or machinery hum
            Ensure the atmosphere feels immersive and realistic, using no tonal or musical elements.
            """

        # Holiday mode inclusion for specific holiday-related atmospheric sounds
        if options.get('holiday_mode') and options.get('selected_holidays'):
            holidays_formatted = ', '.join(options['selected_holidays'])
            system_prompt += f"""
            Include atmospheric sounds typically associated with these holidays: {holidays_formatted}.
            - Christmas: Light bells, snowfall, winter wind
            - Halloween: Dry leaves, eerie wind, distant animal calls
            Ensure these sounds are atmospheric, contributing to the overall environment.
            """

        # Layering sound complexity for depth and atmosphere
        system_prompt += """
        Ensure the audio is multi-layered:
        - Background layer: Constant ambiance (wind, distant urban hum, soft water sounds)
        - Mid layer: Wildlife or mechanical sounds (birds, machinery, insects)
        - Foreground: Active environmental elements relevant to the scene (footsteps, wildlife, flowing water)
        Build these layers to form an immersive and cohesive atmosphere that matches the visual prompt.
        """

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}]
            )
            return completion.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error generating matching audio prompt: {e}")
            return None
    # Function to generate initial audio prompt from video prompt
    def generate_audio_prompt(self, video_prompt, options):
        """
        Generates a detailed audio prompt based on the correlated video prompt.
        The audio prompt aligns with the visual context and adheres to the selected modes,
        such as soundscape, holiday, and layering.
        In soundscape mode, focus strictly on natural, non-musical environmental elements.
        """

        # Base system prompt for audio prompt generation focusing only on desired elements
        system_prompt = f"""
        You are an AI assistant tasked with generating rich, immersive soundscapes that perfectly complement the video context.
        Based on the following video prompt, craft an audio prompt that includes:
        - Environmental sounds (wind, water, nature)
        - Urban or rural ambient sounds (city traffic, machinery, crowd murmur)
        - Layered natural or mechanical sounds (wildlife, footsteps, distant thunder)
        
        Ensure all sounds match the tone and atmosphere described in the video prompt:
        Video Prompt: '{video_prompt}'
        """

        # Soundscape mode: Focus strictly on non-musical environmental sounds
        if options.get('soundscape_mode'):
            system_prompt += """
            In soundscape mode, focus entirely on environmental sounds such as:
            - Wind in natural settings (forests, open fields)
            - Flowing water (streams, rivers, rain)
            - Wildlife (birds, insects, animals)
            - City and mechanical sounds (machinery, urban noise, distant traffic)
            Create an immersive natural or urban atmosphere that feels realistic and engaging.
            """

        # Holiday mode: Add specific, positive sounds associated with selected holidays
        if options.get('holiday_mode') and options.get('selected_holidays'):
            holidays_formatted = ', '.join(options['selected_holidays'])
            system_prompt += f"""
            Since holiday mode is enabled, incorporate festive or atmospheric sounds related to the following holidays: {holidays_formatted}.
            Examples include:
            - Christmas: Gentle bells, soft snowfall, winter wind
            - Halloween: Rustling leaves, eerie wind, distant owls or wolves
            - New Year’s: Distant fireworks, crowd celebrations, cheering
            Focus on positive atmospheric sound elements that align with these holidays.
            """

        # Layering complexity: Define layers for sound design
        system_prompt += """
        Ensure the soundscape is multi-layered for depth:
        - Background layer: Constant ambient sounds (wind, water, distant urban hum)
        - Mid layer: Secondary environmental elements (birds, insects, machinery)
        - Foreground layer: Active sounds relevant to the scene (footsteps, flowing water, wildlife calls)

        The audio should build a cohesive, immersive atmosphere that matches the visual pacing and tone of the video.
        """

        # If specific modes are selected (like chaos mode or other custom modes), append further instructions dynamically:
        if options.get('specific_modes'):
            for mode in options['specific_modes']:
                system_prompt += f"Incorporate {mode}-specific elements where relevant.\n"

        # Chaos mode: Increase unpredictability and intensity of the sounds without introducing unwanted elements
        if options.get('chaos_mode'):
            system_prompt += """
            In chaos mode, create a sense of unpredictability by layering intense and irregular environmental elements such as:
            - Sudden gusts of wind, breaking branches, or crashing waves
            - Distant thunderclaps or sharp animal sounds
            Ensure the scene remains natural but with higher intensity and irregularities in the sound patterns.
            """

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}]
            )
            return completion.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error generating audio prompt: {e}")
            return None

    # Function to refine audio prompts that exceed the 450-character limit
    def refine_audio_prompt(self, audio_prompt):
        """
        Refines audio prompts that exceed the character limit to ensure alignment and conciseness.
        This is especially important for ElevenLabs API optimization.
        """

        refinement_prompt = f"""
        Refine the following audio prompt to keep it under {MAX_CHAR_LIMIT} characters while ensuring it remains aligned with the video context.
        The prompt should focus on environmental, natural, and mechanical sounds, layered to match the atmosphere described.
        
        Audio Prompt: '{audio_prompt}'
        """

        try:
            refinement = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": refinement_prompt}]
            )
            refined_audio_prompt = refinement.choices[0].message['content'].strip()

            if len(refined_audio_prompt) <= MAX_CHAR_LIMIT:
                return refined_audio_prompt
            else:
                print(f"Refinement failed: Prompt still exceeds {MAX_CHAR_LIMIT} characters.")
                return None

        except Exception as e:
            print(f"Error refining audio prompt: {e}")
            return None

    def get_number_of_prompts(self):
        """
        Returns the number of prompts the user has set.
        This assumes there is a variable that tracks the number of prompts.
        """
        return self.video_prompt_number_var.get()  # Or another similar variable for video

    def generate_sound_effects(self):
        """
        Generates sound effects based on the audio prompts.
        Saves the sound effects in the Audio subfolder.
        """
        # Initialize ElevenLabs client before generating sound effects
        initialize_elevenlabs_client()

        if not self.audio_prompts:
            messagebox.showwarning("Prompt Error", "No audio prompts found. Please generate audio prompts first.")
            return

        # Ask for the duration
        duration = simpledialog.askinteger(
            "Sound Duration",
            "Enter the duration of each sound effect in seconds:",
            minvalue=1,
            maxvalue=60
        )
        if not duration:
            messagebox.showwarning("Duration Error", "Please provide a valid duration.")
            return

        # Retrieve the input concept from the text box
        input_concept = self.input_text.get("1.0", tk.END).strip()

        # Use the base directory and save the sound effects in the "Audio" folder
        audio_save_folder = self.audio_save_folder

        # Create the output directory if it doesn't exist
        os.makedirs(audio_save_folder, exist_ok=True)

        # Split the audio prompts into individual prompts
        prompts = [p.strip() for p in self.audio_prompts.strip().split('--------------------') if p.strip()]

        # Retrieve the number of prompts to generate
        number_of_prompts = self.video_prompt_number_var.get()

        for i, prompt_text in enumerate(prompts[:number_of_prompts]):
            if prompt_text:
                # Generate a meaningful filename based on the prompt content
                sanitized_prompt = re.sub(r'[^\w\s-]', '', prompt_text).strip().replace(' ', '_')[:50]  # Limit length
                output_filename = os.path.join(audio_save_folder, f"sound_effect_{i + 1}_{sanitized_prompt}.mp3")

                # Append instructions to avoid music, ethereal tones, and voices
                refined_prompt = f"{prompt_text}. Ensure the soundscape contains only environmental sounds without any music, ethereal tones, or human voices."

                # Generate sound effect using the ElevenLabs API and save it
                print(f"\nProcessing Prompt {i+1}: {refined_prompt}")
                generate_sound_effect(refined_prompt, output_filename, duration)

        messagebox.showinfo("Sound Effects Generated", f"Sound effects have been generated in the '{audio_save_folder}' folder.")
        enable_button(self.combine_button)

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

    def combine_media(self):
        """
        Combines the selected video file with the generated audio sound effects.
        Saves the combined media in the smart-named directory.
        """
        
        # Retrieve the input concept from the text box
        input_concept = self.input_text.get("1.0", tk.END).strip()

        # Create smart directory and filenames
        directory_path, _, _ = self.create_smart_directory_and_filenames(input_concept)

        video_file = filedialog.askopenfilename(
            initialdir=self.video_save_folder,
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")],
            title="Select Video File to Combine"
        )

        if not video_file:
            messagebox.showwarning("No Video File", "No video file selected.")
            return

        audio_files = filedialog.askopenfilenames(
            initialdir=self.audio_save_folder,
            filetypes=[("Audio Files", "*.mp3;*.wav"), ("All Files", "*.*")],
            title="Select Audio Files to Combine"
        )

        if not audio_files:
            messagebox.showwarning("No Audio Files", "No audio files selected.")
            return

        try:
            video = VideoFileClip(video_file)
            audios = [AudioFileClip(audio_file) for audio_file in sorted(audio_files, key=extract_number_from_filename)]

            # Concatenate all audio clips into one
            combined_audio = concatenate_audioclips(audios)
            video_with_audio = video.set_audio(combined_audio)

            # Define the save path within the smart directory
            save_path = os.path.join(directory_path, "combined_media.mp4")

            video_with_audio.write_videofile(save_path, codec="libx264", audio_codec="aac")
            messagebox.showinfo("Combine Successful", f"Media combined and saved to: {save_path}")

        except Exception as e:
            print(f"Error during media combination: {e}")
            messagebox.showerror("Combine Error", f"An error occurred while combining media: {e}")

    # Function to open video prompt options
    def show_video_prompt_options(self):
        self.video_options_window = tk.Toplevel(self.root)
        self.video_options_window.title("Video Prompt Options")
        self.video_options_window.configure(bg='#0A2239')

        self.build_video_options(self.video_options_window)

    # Function to open audio prompt options
    def show_audio_prompt_options(self):
        self.audio_options_window = tk.Toplevel(self.root)
        self.audio_options_window.title("Audio Prompt Options")
        self.audio_options_window.configure(bg='#0A2239')

        self.build_audio_options(self.audio_options_window)

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

        # Video Test Mode (Initialized as BooleanVar)
        self.video_test_mode_var = tk.BooleanVar(value=False)  

        # Theme Dropdown
        self.video_theme_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Theme:", THEMES, 0, 0, self.video_theme_var)

        # Art Style Dropdown
        self.video_art_style_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Art Style:", ART_STYLES, 1, 0, self.video_art_style_var)

        # Lighting Dropdown
        self.video_lighting_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Lighting:", LIGHTING_OPTIONS, 2, 0, self.video_lighting_var)

        # Framing Dropdown
        self.video_framing_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Framing:", FRAMING_OPTIONS, 3, 0, self.video_framing_var)

        # Camera Movement Dropdown
        self.video_camera_movement_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Camera Movement:", CAMERA_MOVEMENTS, 4, 0, self.video_camera_movement_var)

        # Shot Composition Dropdown
        self.video_shot_composition_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Shot Composition:", SHOT_COMPOSITIONS, 5, 0, self.video_shot_composition_var)

        # Time of Day Dropdown
        self.video_time_of_day_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Time of Day:", TIME_OF_DAY_OPTIONS, 6, 0, self.video_time_of_day_var)

        # Decade Dropdown and Camera Options (Dependent on Decade Selection)
        self.video_decade_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Decade:", DECADES, 7, 0, self.video_decade_var)
        self.video_decade_var.trace('w', self.update_video_camera_options)

        self.video_camera_var = tk.StringVar()
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
        self.video_lens_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Lens:", LENSES, 9, 0, self.video_lens_var)

        # Resolution Dropdown
        self.video_resolution_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Resolution:", RESOLUTIONS, 10, 0, self.video_resolution_var)

        # Wildlife and Domesticated Animal Dropdowns
        self.wildlife_animal_var = tk.StringVar()
        self.create_dropdown(options_label_frame, "Wildlife Animal:", WILDLIFE_ANIMALS, 11, 0, self.wildlife_animal_var)

        self.domesticated_animal_var = tk.StringVar()
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
            text="No¹ Mode - Removes people from the generated content",  # Superscript 1 and description in the title
            variable=self.video_no_people_mode_var,
            style='TCheckbutton'
        )
        self.video_no_people_checkbox.grid(row=14, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # SoundScape Mode Checkbox
        self.video_soundscape_mode_var = tk.BooleanVar()
        self.video_soundscape_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="SoundScape Mode - Generates dynamic soundscapes based on visuals",  # Mode with description
            variable=self.video_soundscape_mode_var,
            style='TCheckbutton'
        )
        self.video_soundscape_checkbox.grid(row=14, column=1, columnspan=2, sticky='w', padx=20, pady=2)

        # Story Mode Checkbox
        self.video_story_mode_var = tk.BooleanVar()
        self.video_story_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Story Mode - Ensures prompts flow cohesively as a narrative",  # Mode with description
            variable=self.video_story_mode_var,
            style='TCheckbutton'
        )
        self.video_story_mode_checkbox.grid(row=16, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # Chaos Mode Checkbox
        self.video_chaos_mode_var = tk.BooleanVar()
        self.video_chaos_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Chaos Mode - Inverts positive and negative labels for chaotic results",  # Mode with description
            variable=self.video_chaos_mode_var,
            style='TCheckbutton'
        )
        self.video_chaos_mode_checkbox.grid(row=16, column=1, columnspan=2, sticky='w', padx=20, pady=2)

        # Remix Mode Checkbox
        self.video_remix_mode_var = tk.BooleanVar()
        self.video_remix_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Remix Mode - Applies creative variations to the generated prompts",  # Mode with description
            variable=self.video_remix_mode_var,
            style='TCheckbutton'
        )
        self.video_remix_mode_checkbox.grid(row=18, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # Holiday Mode Checkbox
        self.video_holiday_mode_var = tk.BooleanVar()
        self.video_holiday_mode_checkbox = ttk.Checkbutton(
            options_label_frame,
            text="Holiday Mode - Applies seasonal or holiday-specific themes to prompts",  # Mode with description
            variable=self.video_holiday_mode_var,
            style='TCheckbutton'
        )
        self.video_holiday_mode_checkbox.grid(row=18, column=1, columnspan=2, sticky='w', padx=20, pady=2)
        # Specific Modes Section
        modes_frame = tk.LabelFrame(
            options_label_frame,
            text="Specific Modes",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        modes_frame.grid(row=17, column=0, columnspan=2, padx=10, pady=20, sticky='ew')
        modes_frame.columnconfigure(0, weight=1)

        SPECIFIC_MODES_WITH_DESC = {
            "Chaos Mode": "Generates prompts where positive and negative labels are inverted."
        }
        self.video_specific_modes_vars = {}

        for idx, (mode, description) in enumerate(SPECIFIC_MODES_WITH_DESC.items()):
            var = tk.BooleanVar()
            ttk.Checkbutton(
                modes_frame,
                text=mode,
                variable=var,
                style='TCheckbutton'
            ).grid(row=idx * 2, column=0, sticky='w', padx=20, pady=2)

            self.video_specific_modes_vars[mode] = var

            # Description in smaller font under the checkbox
            tk.Label(
                modes_frame,
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
        
    def show_remix_mode_options(self):
        """
        Opens a popup window to allow the user to select which parameters to randomize in Remix Mode.
        """
        if not self.video_remix_mode_var.get():
            return  # Only show popup when Remix Mode is enabled

        self.remix_options_window = tk.Toplevel(self.root)
        self.remix_options_window.title("Remix Mode Options")
        self.remix_options_window.configure(bg='#0A2239')

        # Label for instructions
        remix_instructions = tk.Label(
            self.remix_options_window,
            text="Select parameters to randomize for each prompt:",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 14, 'bold')
        )
        remix_instructions.pack(pady=10)

        # Frame for checkboxes
        remix_options_frame = tk.Frame(self.remix_options_window, bg='#0A2239')
        remix_options_frame.pack(fill='both', expand=True, padx=10, pady=10)
        remix_options_frame.columnconfigure(0, weight=1)

        # Create checkboxes for each parameter
        self.remix_vars = {}
        parameters_to_randomize = [
            "Theme", "Art Style", "Lighting", "Framing", "Camera Movement",
            "Shot Composition", "Time of Day", "Decade", "Camera", "Lens", "Resolution",
            "Wildlife Animal", "Domesticated Animal", "SoundScape Mode"
        ]

        for idx, param in enumerate(parameters_to_randomize):
            var = tk.BooleanVar()
            self.remix_vars[param] = var
            tk.Checkbutton(
                remix_options_frame,
                text=param,
                variable=var,
                bg='#0A2239',
                fg='white',
                selectcolor="#28a745",
                font=('Helvetica', 12)
            ).grid(row=idx, column=0, sticky='w', padx=10, pady=2)

        # Save button
        save_remix_button = tk.Button(
            self.remix_options_window,
            text="Save Remix Options",
            command=self.save_remix_options,
            bg="#28a745",
            fg='white',
            font=('Helvetica', 12, 'bold'),
            activebackground="#1e7e34",
            activeforeground='white',
            cursor="hand2",
            width=20,
            height=2
        )
        save_remix_button.pack(pady=10)

    def save_remix_options(self):
        """
        Saves the selected parameters to be randomized in Remix Mode.
        """
        self.selected_remix_params = {param: var.get() for param, var in self.remix_vars.items() if var.get()}
        self.remix_options_window.destroy()


    def build_audio_options(self, parent):
        options_label_frame = tk.LabelFrame(
            parent,
            text="Audio Prompt Options",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 14, 'bold')
        )
        options_label_frame.pack(fill='both', expand=True, padx=10, pady=10)
        options_label_frame.columnconfigure((0,1), weight=1)

        # Modes Checkboxes
        modes_frame = tk.LabelFrame(
            options_label_frame,
            text="Modes",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        modes_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=20, sticky='ew')
        modes_frame.columnconfigure(0, weight=1)

        # Exclude Music Checkbox
        self.audio_exclude_music_var = tk.BooleanVar()
        self.audio_exclude_music_checkbox = ttk.Checkbutton(
            modes_frame,
            text="Exclude Music & Musical Instruments",
            variable=self.audio_exclude_music_var,
            style='TCheckbutton'
        )
        self.audio_exclude_music_checkbox.grid(row=0, column=0, sticky='w', padx=10, pady=5)

        # Description for Exclude Music Mode
        exclude_music_description = tk.Label(
            modes_frame,
            text="Prevents background music and instrumental sounds from being added to the soundscape.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        exclude_music_description.grid(row=1, column=0, sticky='w', padx=40, pady=(0, 10))

        # Holiday Mode Checkbox
        self.audio_holiday_mode_var = tk.BooleanVar()
        self.audio_holiday_mode_checkbox = ttk.Checkbutton(
            modes_frame,
            text="Enable Holiday Mode",
            variable=self.audio_holiday_mode_var,
            command=self.update_audio_modes,
            style='TCheckbutton'
        )
        self.audio_holiday_mode_checkbox.grid(row=2, column=0, sticky='w', padx=10, pady=5)

        # Description for Holiday Mode
        holiday_mode_description = tk.Label(
            modes_frame,
            text="Adds holiday-themed sounds (e.g., bells, seasonal wind) to the soundscape.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        holiday_mode_description.grid(row=3, column=0, sticky='w', padx=40, pady=(0, 10))

        # Holiday Selection Dropdown (initially disabled)
        self.audio_holidays_var = tk.StringVar()
        self.audio_holidays_combobox = ttk.Combobox(
            options_label_frame,
            textvariable=self.audio_holidays_var,
            state="disabled",  # Initially disabled
            values=HOLIDAYS,
            font=('Helvetica', 12)
        )
        self.audio_holidays_combobox.set(HOLIDAYS[0])
        self.audio_holidays_combobox.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        self.create_label(options_label_frame, "Holiday:", 1, 0)

        # Holiday Influence Slider
        self.audio_holiday_influence_var = tk.IntVar(value=50)  # Default influence set to 50%
        self.audio_holiday_influence_slider = ttk.Scale(
            modes_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.audio_holiday_influence_var
        )
        self.audio_holiday_influence_slider.grid(row=4, column=0, sticky='ew', padx=20, pady=5)

        holiday_influence_description = tk.Label(
            modes_frame,
            text="Adjust the influence of holiday-related sound effects (0-100%).",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        holiday_influence_description.grid(row=5, column=0, sticky='w', padx=40, pady=(0, 10))

        # Story Mode Checkbox
        self.audio_story_mode_var = tk.BooleanVar()
        self.audio_story_mode_checkbox = ttk.Checkbutton(
            modes_frame,
            text="Story Mode",
            variable=self.audio_story_mode_var,
            style='TCheckbutton'
        )
        self.audio_story_mode_checkbox.grid(row=6, column=0, sticky='w', padx=10, pady=5)

        story_mode_description = tk.Label(
            modes_frame,
            text="Ensures cohesiveness and continuity across generated prompts.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        story_mode_description.grid(row=7, column=0, sticky='w', padx=40, pady=(0, 10))

        # Chaos Mode Checkbox
        self.audio_chaos_mode_var = tk.BooleanVar()
        self.audio_chaos_mode_checkbox = ttk.Checkbutton(
            modes_frame,
            text="Chaos Mode",
            variable=self.audio_chaos_mode_var,
            style='TCheckbutton'
        )
        self.audio_chaos_mode_checkbox.grid(row=8, column=0, sticky='w', padx=10, pady=5)

        chaos_mode_description = tk.Label(
            modes_frame,
            text="Generates prompts where positive and negative labels are inverted.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        chaos_mode_description.grid(row=9, column=0, sticky='w', padx=40, pady=(0, 10))

        # Specific Modes Checkboxes with Influence Sliders
        specific_modes_label = tk.Label(
            modes_frame,
            text="Specific Modes:",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        specific_modes_label.grid(row=10, column=0, sticky='w', padx=10, pady=(15, 5))

        self.audio_specific_modes_vars = {}
        self.audio_specific_modes_influence_vars = {}

        for idx, mode in enumerate(SPECIFIC_MODES):
            # Mode Checkboxes
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(
                modes_frame,
                text=mode,
                variable=var,
                style='TCheckbutton'
            )
            cb.grid(row=11+idx*3, column=0, sticky='w', padx=20, pady=2)
            self.audio_specific_modes_vars[mode] = var

            # Mode Influence Sliders
            influence_var = tk.IntVar(value=50)  # Default influence set to 50%
            self.audio_specific_modes_influence_vars[mode] = influence_var
            influence_slider = ttk.Scale(
                modes_frame,
                from_=0,
                to=100,
                orient="horizontal",
                variable=influence_var
            )
            influence_slider.grid(row=12+idx*3, column=0, sticky='ew', padx=20, pady=5)

            # Description for each Mode Influence
            mode_influence_description = tk.Label(
                modes_frame,
                text=f"Adjust the influence of {mode.lower()} mode (0-100%).",
                bg='#0A2239',
                fg='light gray',
                font=('Helvetica', 10, 'italic')
            )
            mode_influence_description.grid(row=13+idx*3, column=0, sticky='w', padx=40, pady=(0, 10))

        # Layer Intensity for Sound Effects
        layer_intensity_label = tk.Label(
            modes_frame,
            text="Layer Intensity (0-100%):",
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12, 'bold')
        )
        layer_intensity_label.grid(row=14+len(SPECIFIC_MODES)*3, column=0, sticky='w', padx=10, pady=10)

        self.audio_layer_intensity_var = tk.IntVar(value=50)  # Default intensity set to 50%
        self.audio_layer_intensity_slider = ttk.Scale(
            modes_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.audio_layer_intensity_var
        )
        self.audio_layer_intensity_slider.grid(row=15+len(SPECIFIC_MODES)*3, column=0, sticky='ew', padx=20, pady=5)

        # Description for Layer Intensity Slider
        layer_intensity_description = tk.Label(
            modes_frame,
            text="Control how many sound layers are generated. Lower values create simpler soundscapes, higher values add complexity.",
            bg='#0A2239',
            fg='light gray',
            font=('Helvetica', 10, 'italic')
        )
        layer_intensity_description.grid(row=16+len(SPECIFIC_MODES)*3, column=0, sticky='w', padx=40, pady=(0, 10))

        # Save Button
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

    def create_dropdown(self, parent, label_text, values_list, row, column, var):
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
        combobox.set(values_list[0])
        combobox.grid(row=row, column=column+1, padx=10, pady=10, sticky='ew')

    def create_label(self, parent, text, row, column):
        label = tk.Label(
            parent,
            text=text,
            bg='#0A2239',
            fg='white',
            font=('Helvetica', 12)
        )
        label.grid(row=row, column=column, padx=10, pady=10, sticky='e')

    def update_video_camera_options(self, *args):
        decade = self.video_decade_var.get()
        self.video_camera_var.set('')
        self.video_camera_combobox.config(values=CAMERAS[decade])
        self.video_camera_combobox.set(CAMERAS[decade][0])

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

# Function to save video options and prompt the user for a base directory
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
            "test_mode": self.video_test_mode_var.get()
        }
        self.save_options_to_file('video_options', cinematic_options)

        # Prompt the user to select a base directory for saving the files
        base_directory = filedialog.askdirectory(title="Select Base Directory for Saving Files")
        if not base_directory:
            messagebox.showerror("Error", "No directory selected. Please select a directory to save files.")
            return

        # Create subfolders for Video and Audio inside the base directory
        self.video_save_folder = os.path.join(base_directory, "Video")
        self.audio_save_folder = os.path.join(base_directory, "Audio")

        os.makedirs(self.video_save_folder, exist_ok=True)
        os.makedirs(self.audio_save_folder, exist_ok=True)

        self.base_directory = base_directory  # Store the base directory for later use
        self.video_options_window.destroy()

    def save_audio_options(self):
        self.audio_options_set = True
        # Save options to settings file
        audio_options = {
            "exclude_music": self.audio_exclude_music_var.get(),
            "holiday_mode": self.audio_holiday_mode_var.get(),
            "selected_holidays": [self.audio_holidays_var.get()] if self.audio_holiday_mode_var.get() else [],
            "specific_modes": [mode for mode, var in self.audio_specific_modes_vars.items() if var.get()]
        }
        self.save_options_to_file('audio_options', audio_options)
        self.audio_options_window.destroy()

    def save_options_to_file(self, key, options):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        settings[key] = options
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)

    # Functions to update API keys
    def set_openai_api_key(self):
        global OPENAI_API_KEY
        OPENAI_API_KEY = simpledialog.askstring("Set OpenAI API Key", "Enter your OpenAI API Key:", show="*")
        save_settings()

    def set_elevenlabs_api_key(self):
        global ELEVENLABS_API_KEY
        ELEVENLABS_API_KEY = simpledialog.askstring("Set ElevenLabs API Key", "Enter your ElevenLabs API Key:", show="*")
        save_settings()

# Main Execution
if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(800, 600)
    app = MultimediaSuiteApp(root)
    root.mainloop()