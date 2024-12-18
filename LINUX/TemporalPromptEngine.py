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
from pydub import AudioSegment, effects
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
from scipy.signal import welch
import torch
from diffusers import AudioLDM2Pipeline
from pathlib import Path
import uuid
import numpy as np  # Ensure this line is present




# Load environment variables from .env file
load_dotenv()

# Initialize global variables
# Initialize global variables
OUTPUT_DIRECTORY = os.path.join(os.getcwd(), "prompts_output")
COMFYUI_PROMPTS_FOLDER = os.getenv("COMFYUI_PROMPTS_FOLDER")
MAX_CHAR_LIMIT = 10000  # Maximum characters allowed for prompts
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

# --------------------- Theming Configuration ---------------------
# Define the color palette
COLOR_PALETTE = {
    "white": "#ffffff",
    "dark_blue_black": "#1a1a2e",
    "gold": "#ffab00",
    "dark_navy": "#040e24",
    "deep_gold": "#ffa200",
    "light_gold": "#ffc14e",
    "light_blue": "#00aced"  # Added for 'light blue' text if needed
}

def apply_theme(style, root):
    style.theme_use('default')

    # Configure styles for ttk widgets
    style.configure('TFrame', background=COLOR_PALETTE["dark_blue_black"])
    style.configure('TLabel', background=COLOR_PALETTE["dark_blue_black"], foreground=COLOR_PALETTE["white"])
    style.configure('TButton', background=COLOR_PALETTE["gold"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.configure('TEntry', fieldbackground=COLOR_PALETTE["white"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.configure('TCombobox', fieldbackground=COLOR_PALETTE["white"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.configure('TNotebook', background=COLOR_PALETTE["dark_blue_black"])
    style.configure('TNotebook.Tab', background=COLOR_PALETTE["gold"], foreground=COLOR_PALETTE["dark_blue_black"])
    style.map('TButton',
              background=[('active', COLOR_PALETTE["deep_gold"])],
              foreground=[('active', COLOR_PALETTE["dark_blue_black"])])
    style.configure('TCheckbutton', background=COLOR_PALETTE["dark_blue_black"], foreground=COLOR_PALETTE["white"])

    # Set the root window background
    root.configure(background=COLOR_PALETTE["dark_blue_black"])

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
    # Historical & Period - Rooted in authenticity and era-specific detail
    "Historical & Period, meticulous production design, era-accurate costumes, authentic locales",
    "Historical Epic, large-scale sets, sweeping cinematography, grandeur and historical scope",
    "Period Drama, nuanced performances, intimate framing, refined attention to historical nuance",
    "Sword and Sandal, ancient worlds, grand arenas, heroic silhouettes against classical backdrops",
    "Western, sunbaked landscapes, dusty main streets, iconic long shots and frontier isolation",
    "Spaghetti Western, heightened style, intense close-ups, sparse dialogue punctuated by silence",

    # Noir & Neo-Noir - Shadows, moral complexity, urban dread
    "Film Noir, stark contrast lighting, cigarette smoke, cynical antiheroes framed by urban nights",
    "Neo-Noir, updated grit, neon reflections, morally ambiguous protagonists in modern cityscapes",
    "German Expressionism, angular sets, skewed perspectives, evocative shadows shaping emotion",
    "Italian Neorealism, on-location shooting, non-actors, raw authenticity reflecting social realities",

    # Documentary & Realism - Truthful lenses, minimal interference
    "Documentary, real-world footage, natural sound, handheld immediacy",
    "Nature Documentary, macro details, sweeping aerials, reverence for wildlife",
    "Educational Documentary, clear visuals, explanatory graphics, a didactic yet engaging tone",
    "Biographical, respectful portrayal of true lives, mix of archival and recreated moments",

    # Drama & Character Studies - Emotionally resonant, human complexity
    "Drama, intimate close-ups, subtle lighting, emotional authenticity guiding visual choices",
    "Coming-of-Age, warm natural light, relatable suburban or rural settings, evolving visual tone as characters mature",
    "Social Realism, subdued palettes, handheld camerawork, everyday struggles portrayed without varnish",
    "Family Drama, soft hues, familiar domestic spaces, close framing that captures nuanced dynamics",

    # Action & Adventure - Movement, scale, and energy
    "Action, kinetic camerawork, dynamic editing, vivid choreography showcasing stunts",
    "Adventure, wide vistas, lush environments, a sense of exploration in composition",
    "Martial Arts, graceful camera tracking, crisp focus on technique, rhythmic pacing of combat",
    "Heist, slick framing, meticulous focus on details, tension built through spatial relationships",

    # Comedy & Lighthearted Tones - Brightness, pace, and levity
    "Comedy, brighter lighting, punchy color, playful camera angles and comedic timing",
    "Romantic Comedy, urban warmth, soft focus on faces, charmingly lit interiors",
    "Satire/Parody, exaggerated tropes, heightened sets or costumes to underscore the joke",
    "Black Comedy, ironic lighting contrasts, cheerful palettes framing dark humor",

    # Crime & Thriller - Shadows, suspense, and moral tension
    "Crime, gritty cityscapes, moody corners, naturalistic low-key lighting for tension",
    "Mystery Thriller, controlled reveals, shadowy compositions, camera linger on crucial details",
    "Political Thriller, formal interiors, restrained camera moves, a tension beneath polished surfaces",
    "Heist Thriller, immaculate framing of plans and blueprints, high-contrast focus on faces under pressure",

    # War, Conflict & Larger Issues - Intensity, moral complexity, historical weight
    "War Drama, handheld immediacy, muted color grades, visually conveying the cost of conflict",
    "Social Commentary, realistic settings, symbolic framing, contrasts in wealth or power",
    "Slice-of-Life, mundane beauty, gentle handheld observation, unembellished reality",
    "Independent, inventive compositions, resourceful lighting, character-driven spaces",

    # Fantasy & Science Fiction - World-building, imagination, and spectacle
    "Fantasy, ornate sets, magical lighting cues, lush color saturations hinting at the unreal",
    "Epic Fantasy, sweeping crane shots, intricate world-building, costumes and sets that transport viewers",
    "Science Fiction, sleek production design, futuristic architecture, balanced color schemes of metals and neons",
    "Cyberpunk, neon reflections in puddles, overcrowded city vistas, technology-infused mise-en-scène",
    "Steampunk, intricate clockwork props, warm sepia tones, Victorian-industrial fusion",
    "Dystopian, desaturated palettes, decayed urban environments, oppressive symmetry in composition",
    "Utopian, pristine whites, harmonious geometry, soft diffused lighting conveying an idealized order",
    "Magical Realism, otherwise real settings gently tilted by subtle, wondrous details",
    "Superhero, bold primary colors, dynamic angles, kinetic sequences foregrounding extraordinary abilities",

    # Horror & Suspense - Atmosphere, tension, and the unseen
    "Horror, dim lighting, claustrophobic framing, suggestive shadows and unsettling angles",
    "Psychological Horror, skewed perspectives, uncanny compositions, prolonged close-ups on haunted expressions",
    "Found Footage Horror, raw camerawork, grainy low-light conditions, frantic real-time feel",

    # Arthouse & Experimental - Formal innovation, thematic depth, visual poetry
    "Arthouse, painterly frames, symbolic images, contemplative pacing over narrative clarity",
    "Surrealism, dream-logic visuals, bizarre juxtapositions, layered metaphors in every shot",
    "Avant-Garde, non-linear editing, abstract shapes and colors, sound and image as pure expression",
    "Existential, minimalist sets, long takes, existential emptiness captured through negative space",

    # Musical & Dance - Movement, color, and rhythm
    "Musical, vivid palettes, dynamic crane shots, choreography synchronized with camerawork",
    "Dance Film, fluid tracking, emphasis on bodies in motion, rhythmic editing aligned to music",

    # Epic & Grand Scale - Majestic aesthetics, heroic narratives
    "Mythological Adventure, grand vistas, legendary creatures, artful framing of larger-than-life tales",
    "Space Opera, cosmic panoramas, shimmering starfields, operatic scale in design and motion",

    # Youth & Family - Warmth, accessibility, and relatability
    "Children's, bright colors, gentle lighting, camera angles at child’s eye-level",
    "Family, cozy interiors, warm color grading, inclusive compositions where characters share the frame",
    "Teen Drama, natural daylight, authentic cluttered bedrooms, handheld honesty reflecting youth",

    # Science & Tech - Analytical and visionary aesthetics
    "Hard Science Fiction, rigorous attention to realism in spacecraft and labs, crisp lighting, neutral palettes",
    "Space Exploration, awe-inspiring wide shots of cosmic bodies, elegantly slow camera movements conveying wonder",
    "Alien Invasion, dramatic skies, silhouettes against unknown forms, a hint of the otherworldly"
]



ART_STYLES = [
    "Documentary Style, nonfiction approach, observational camera work, authentic scenarios",
    "Realism, true-to-life depiction, natural lighting, believable performances",
    "Surrealist Cinema, dreamlike sequences, symbolic imagery, unconventional narratives",
    "Expressionist Cinema, stylized sets, dramatic shadows, heightened emotional tone",
    "Impressionist Cinema, soft focus, atmospheric visuals, evocative moods",
    "Minimalist Cinema, sparse sets, limited props, focus on subtle narrative elements",
    "Abstract Film, non-literal visuals, experimental edits, shapes and colors as storytelling tools",
    "Experimental Film, avant-garde techniques, non-traditional narratives, boundary-pushing visuals",
    "Silent Era Style, intertitles for dialogue, expressive acting, emphasis on visual storytelling",
    "Classic Hollywood Style, studio era production, polished lighting, genre conventions",
    "Film Noir Style, high-contrast lighting, urban crime settings, morally ambiguous characters",
    "Italian Neorealism, on-location shooting, non-professional actors, gritty social realism",
    "French New Wave, handheld cameras, jump cuts, self-referential narratives",
    "Soviet Montage, rapid editing, juxtaposed images, thematic intellectual impact",
    "Dogme 95 Style, handheld camera, natural sound, available lighting, minimal post-production",
    "Postmodern Cinema, genre blending, intertextual references, playful narrative structures",
    "Cyberpunk Cinema, neon-lit cityscapes, high-tech societies, urban dystopia",
    "Steampunk Cinema, mechanical retro-futurism, industrial aesthetics, Victorian-era tech",
    "Retro-Futuristic Cinema, future imagined through past lens, vintage technology designs",
    "Dystopian Cinema, oppressive societies, decayed settings, bleak color palettes",
    "Utopian Cinema, idealized worlds, harmonious visuals, balanced compositions",
    "Fantasy Cinema, magical worlds, mythical creatures, elaborate production design",
    "Futuristic Cinema, advanced technology, sleek minimalism, forward-looking environments",
    "Magical Realist Cinema, subtle fantastical elements in real settings, seamless blending",
    "Anime-Style Animation, stylized character design, vibrant color palettes, expressive motion",
    "Traditional 2D Animation, hand-drawn frames, fluid movement, classic cel techniques",
    "Stop-Motion Animation, frame-by-frame manipulation of models or puppets, tangible textures",
    "Claymation, sculpted clay figures animated frame-by-frame, pliable characters and sets",
    "Puppet Animation, articulated puppets, controlled movements, tangible tactile feel",
    "Cut-Out Animation, flat paper or card figures, layered scenes, collage-like movement",
    "Silhouette Animation, backlit cut-outs, stark contrast, elegant shadow play",
    "Rotoscope Animation, traced over live-action footage, fluid hybrid style, realistic motion",
    "3D CGI Animation, computer-generated models, virtual lighting, photoreal or stylized worlds",
    "Motion Capture Animation, recorded human performance data, naturalistic character movement",
    "Pixilation, stop-motion with live actors, choppy surreal movement, playful reality bending",
    "Live-Action/Animation Hybrid, real footage merged with drawn or CGI elements, mixed realities",
    "Found Footage Style, existing video repurposed, documentary-like authenticity, collage narrative",
    "Mockumentary Style, fictional events portrayed as real doc, comedic or satiric tone",
    "Music Video Style, rhythm-driven editing, stylized visuals, performance emphasis",
    "Neon-Noir Style, blend of noir’s darkness with neon aesthetic, moody urban visuals",
    "High Fantasy Cinema, epic scope, grand mythical landscapes, richly detailed costumes and sets",
    "VHS Aesthetic Cinema, analog video texture, tape artifacts, retro home-video feel",
    "IMAX Large Format, high-resolution film, immersive image scale, grand visual impact",
    "3D Stereoscopic Cinema, added depth perception, layered planes of action, enhanced immersion",
    "VR/360-Degree Film, immersive spherical viewing, interactive vantage points, spatial storytelling",
    "Motion Graphics & Title Sequences, animated typography, stylized intros, graphic-driven narration"
]


LIGHTING_OPTIONS = [
    "Contextual Lighting, scene-aligned source, adaptive ambiance, purposeful illumination",

    "Natural Lighting, organic ambient source, environmental radiance, non-artificial glow",
    "Sunlight, bright natural rays, crisp outdoor illumination, strong directional highlights",
    "Golden Hour, warm low-angle glow, soft transitions, gentle contrast",
    "Blue Hour, cool twilight tones, subtle softness, muted gradients",
    "Moonlight, faint bluish hue, nocturnal calm, gentle luminance",
    "Candlelight, warm flicker, intimate radiance, soft, localized glow",
    "Firelight, dynamic flames, dancing highlights, warm shifting hues",

    "Studio Lighting, controlled conditions, stable outputs, customizable setups",
    "Softbox Lighting, diffused softness, even coverage, flattering highlights",
    "Ring Light, circular catchlights, facial emphasis, balanced frontal glow",
    "Spotlight, focused beam, high-contrast spot, intensified focal emphasis",
    "Continuous Lighting, steady beam, consistent output, reliable baseline",
    "Strobe Lighting, flash bursts, crisp definition, frozen action details",
    "LED Lighting, energy-efficient, versatile hues, adjustable intensity",
    "RGB Lighting, customizable color tones, mood-based hues, creative variation",

    "Directional Lighting, angle-defined source, contour shaping, shadow interplay",
    "Side Lighting, lateral illumination, pronounced depth, accentuated texture",
    "Top Lighting, overhead beam, stark shadows, downward emphasis",
    "Underlighting, upward glow, unsettling ambiance, unusual highlight angles",
    "Front Lighting, direct frontal beam, reduced shadows, clear subject definition",
    "Backlighting, silhouetted form, outlined edges, separated subject",
    "Rim Lighting, edge highlights, subject-background separation, outlined contours",
    "Cross Lighting, intersecting beams, sculpted texture, dynamic dimension",
    "Split Lighting, half-face illumination, dramatic contrast, focused tension",
    "Loop Lighting, subtle cheek shadow, popular portrait style, soft facial contour",

    "High & Low Key Lighting, tonal control, mood shaping, contrast management",
    "High Key, bright, low-contrast glow, airy mood, minimal shadow",
    "Low Key, dark, high-contrast mood, dramatic depth, strong shadow shapes",
    "Chiaroscuro, intense light-dark interplay, rich dimensionality, pronounced drama",
    "Rembrandt Lighting, classic painterly style, triangular face light, refined portrait",
    "Ambient Lighting, overall gentle glow, background softness, diffused presence",
    "Motivated Lighting, narrative-justified glow, scene-integrated, logical source placement",

    "Colored & Thematic Lighting, hue-based mood, stylistic choices, scene-specific color",
    "Neon Lighting, vibrant chroma, urban flair, punchy highlights",
    "Colored Gel Lighting, tinted filters, thematic hues, mood enhancement",
    "Candle Glow, warm intimate tone, soft transitions, cozy ambiance",
    "Blacklight, UV hue, otherworldly glow, glowing elements",
    "Vintage Lighting, nostalgic warmth, old-era feel, softened palette",
    "Modern Lighting, sleek brightness, clean lines, contemporary vibe",
    "Futuristic Lighting, high-tech tone, cool accents, advanced aesthetic",
    "Genre-Specific Lighting, style-fitting hues, aligned with thematic genre, narrative tone",

    "Dynamic & Adaptive Lighting, interactive shifts, evolving intensity, responsive changes",
    "Reactive Lighting, motion-triggered variance, scene-responsive, fluid illumination",
    "Timed Lighting, scheduled intensity shifts, rhythmic changes, pre-set transitions",
    "Transitional Lighting, gradual brightness shifts, smooth luminosity flow, evolving mood",
    "Shadow Play, artistic silhouette use, pattern casting, dramatic shapes",
    "Light Painting, controlled light streaks, creative tracings, artistic overlays",

    "Specialized Lighting Techniques, distinct setups, unique patterns, stylized approach",
    "Silhouette, outlined form, backlit shape, clear profile",
    "Gobo Lighting, patterned beams, textural overlays, shaped projections",
    "Grid Lighting, focused spots, isolated illumination, selective emphasis",
    "Projected Lighting, cast shapes, thematic patterns, narrative visuals",
    "Butterfly Lighting, nose shadow below, flattering facial highlight, portrait classic",
    "Split Lighting, bisected facial light, symmetrical tension, dramatic mood",
    "Color Gradient Lighting, smooth hue transitions, blended chroma, layered palette",

    "Practical Lighting, on-set motivated sources, integrated fixtures, natural placement",
    "Functional Practical Lighting, utility-based glow, situational realism, justified illumination",
    "Symbolic Practical Lighting, meaning-laden sources, narrative reinforcement, metaphorical hue",
    "Decorative Practical Lighting, aesthetic accents, visual enhancements, pleasing highlights",
    "Thematic Practical Lighting, style-aligned sources, set-matched hues, story synergy",
    "Mood Practical Lighting, emotive color, subtle ambiance, tone setting",
    "Accent Lighting, focused highlights, detail emphasis, subtle framing",

    "Control & Adjustment Lighting, fine-tuned outputs, precise handling, customizable quality",
    "Dimmed Lighting, reduced intensity, subdued glow, softened presence",
    "Bright Lighting, high intensity, crisp clarity, strong definition",
    "Diffused Light, softened spread, gentle shadows, minimized harshness",
    "Bounce Lighting, reflected beam, indirect illumination, softened contrast",
    "Fill Lighting, shadow reduction, balanced exposure, even tone",
    "Directional Light, angled beam, sculpted texture, controlled highlight",
    "Temperature Controlled Lighting, warm-cool balance, hue tuning, refined atmosphere",
    "Intensity Controlled Lighting, adjustable brightness, custom luminosity, scene matching",
    "Focus Controlled Lighting, pinpointed clarity, targeted emphasis, selective highlight",
    "Shadowless Lighting, even wash, minimal contrast, flatter dimension",
    "Matte Lighting, non-reflective glow, reduced glare, even surface tone",
    "Glossy Lighting, reflective highlight, shiny surfaces, crisp specularity",

    "Atmospheric & Narrative Lighting, story-aligned mood, environment shaping, thematic tone",
    "Atmospheric Lighting, enveloping glow, environmental enhancement, immersive aura",
    "Narrative Lighting, story-supporting hues, plot-driven tones, contextual intensity",
    "Symbolic Lighting, metaphorical hue, representational brightness, interpretive depth",
    "Mood Lighting, emotional tone setting, color-driven sentiment, scene feel",
    "Thematic Lighting, concept-aligned style, cohesive atmosphere, narrative consistency",
    "Environmental Lighting, realistic match, location-fitting brightness, scene authenticity",

    "Edge & Rim Lighting, contour emphasis, subject separation, defined silhouettes",
    "Edge Lighting, highlighting borders, clear outlines, crisp separation",
    "Accent Rim Lighting, subtle edge highlight, gentle glow, refined border",
    "Diffuse Rim Lighting, softened halo, less defined edges, hazy outline",
    "Bounced Rim Lighting, reflected edge illumination, indirect highlight, gentle contour",

    "Experimental & Creative Lighting, bold techniques, innovative choices, artistic flair",
    "Layered Lighting, multiple levels, complex interplay, depth-rich luminance",
    "Patterned Lighting, textural motifs, structured beams, decorative overlay",
    "Animated Lighting, moving brightness, fluctuating intensity, kinetic effect",
    "Integrated Lighting, seamless source blend, cohesive multi-light mix, holistic glow",
    "Artistic Lighting, expressive configurations, stylized brilliance, aesthetic emphasis",
    "Technical Lighting, precise control, meticulous setups, exactingly crafted illumination",
    "Balanced Lighting, even distribution, harmonious spread, visually stable",
    "Asymmetrical Lighting, uneven distribution, dynamic emphasis, unique contrasts",
    "Complementary Lighting, harmonized hues, color synergy, pleasing combinations",
    "Light Blending, smooth transitions, merged sources, unified radiance"
]


FRAMING_OPTIONS = [
    "Contextual Shot, adaptive framing, scene-integrated viewpoint, flexible perspective",

    "Basic Shots, foundational setups, standard distances, commonly used perspectives",
    "Wide Shot, broad coverage, environmental context, full spatial awareness",
    "Medium Shot, waist-level framing, moderate proximity, balanced subject detail",
    "Close-Up, focused detail, intimate scale, enhanced facial or object features",
    "Extreme Close-Up, minute detail emphasis, magnified focus, heightened intimacy",
    "Long Shot, full-body view, contextual background, spatial orientation",
    "Full Shot, entire figure in frame, comprehensive subject capture, stable layout",
    "Headshot, close framing of face, expressive detail, direct subject focus",

    "Specialty Shots, unique angles, unconventional framing, distinctive vantage",
    "Over-the-Shoulder, subject foreground silhouette, contextual background, relational perspective",
    "Dutch Angle, tilted axis, tension or unease, dynamic visual cue",
    "Bird's Eye View, top-down angle, overarching scope, diminutive subject appearance",
    "Point of View, first-person perspective, subjective immersion, personal vantage",
    "Two-Shot, dual-subject framing, relational balance, shared focus",
    "Cowboy Shot, mid-thigh framing, character stance emphasis, stylized distance",
    "Establishing Shot, sets scene context, locational clarity, spatial orientation",
    "Aerial Shot, high-elevation viewpoint, expansive overview, grand scale",
    "Insert Shot, close detail of object/action, focal emphasis, narrative clarity",
    "Reverse Angle, opposite viewpoint, alternate perspective, situational contrast",

    "Movement Shots, camera motion integrated, dynamic framing, shifting perspectives",
    "Tracking Shot, follows subject’s path, consistent alignment, fluid movement",
    "Static Shot, fixed position, stable composition, stationary viewpoint",
    "Pan Shot, horizontal sweep, scene-wide coverage, lateral progression",
    "Tilt Shot, vertical sweep, layered vertical reveal, changing vertical axis",
    "Zoom Shot, focal length shift, magnification change, altered subject scale",
    "Crane Shot, elevated camera movement, overhead dimension, vertical expansion",
    "Dolly Shot, forward/backward glide, depth variation, closer/farther engagement",
    "Arc Shot, circular motion, evolving perspective, subject-centric orbit",
    "Rack Focus, focus plane shift, selective clarity transitions, layered emphasis",
    "Push In Shot, camera advances, intensified subject detail, closer interaction",
    "Pull Back Shot, camera retreats, widened context, diminishing subject emphasis",

    "Composition Shots, structured visual arrangement, artful balance, deliberate layout",
    "Symmetrical Composition, mirrored balance, harmonious arrangement, stable equilibrium",
    "Asymmetrical Composition, uneven element placement, dynamic tension, visual interest",
    "Rule of Thirds Composition, grid-based alignment, natural balance, guided focal points",
    "Golden Ratio Composition, proportioned framing, organic harmony, aesthetic appeal",
    "Centered Composition, main focus centered, direct emphasis, straightforward clarity",
    "Negative Space Composition, emptiness around subject, isolated emphasis, minimalist impact",
    "Layered Composition, multiple depth planes, spatial complexity, dimensional richness",
    "Foreground Interest, prominent frontal elements, immediate draw, layered scene",
    "Background Interest, emphasized backdrop, contextual depth, secondary narrative",

    "Angle Shots, perspective shifts, altered vantage points, varied visual impact",
    "High Angle Shot, camera looking down, subject diminished, increased field overview",
    "Low Angle Shot, camera looking up, subject amplified, imposing presence",
    "Worm's Eye View, ground-up vantage, towering subjects, dramatic vertical scale",
    "Extreme High Angle, very elevated perspective, comprehensive scene layout, small subject scale",
    "Extreme Low Angle, near-ground view, exaggerated height, striking upward view",
    "Overhead Shot, directly above subject, top-down framing, symmetrical floor pattern",

    "Lens Effects, optical variations, altered perspective, stylized imagery",
    "Wide Angle Shot, broad field, exaggerated depth, expanded peripheral detail",
    "Telephoto Shot, narrow view, compressed distance, selective subject isolation",
    "Macro Shot, extreme close focus, tiny detail magnification, intricate clarity",
    "Fisheye Shot, curved distortion, wide spherical view, unique spatial warp",
    "Tilt-Shift Shot, selective focus, miniaturized appearance, painterly depth",
    "Anamorphic Shot, wide cinematic ratio, horizontal stretch, filmic aesthetic",
    "Soft Focus, gentle blur, atmospheric haze, diffused details",
    "Infrared Shot, heat-sensitive capture, surreal color range, alternate spectrum",
    "UV Shot, ultraviolet capture, revealed hidden details, unconventional spectrum",

    "Special Effects Shots, temporal or dimensional alterations, stylized pacing, enhanced impact",
    "Time-Lapse Shot, accelerated passage, compressed chronology, evolving scene",
    "Hyperlapse Shot, moving time-lapse, spatial-temporal progression, dynamic journey",
    "Slow Motion Shot, reduced speed, accentuated action details, emotional emphasis",
    "Fast Motion Shot, increased speed, urgent energy, heightened tempo",
    "Frozen Time Shot, suspended instant, static subject, paused motion",
    "Bullet Time Shot, 360-degree freeze, rotational vantage, multi-angle pause",

    "Composition Techniques, creative frameworks, aesthetic structuring, artistic emphasis",
    "Leading Lines, guiding lines, directed eye movement, focal route",
    "Framing Within Framing, nested boundaries, multi-layered enclosure, dimensional depth",
    "Silhouette Composition, backlit outline, shape emphasis, high contrast",
    "Reflection Composition, mirrored surface, duplicated imagery, symmetrical intrigue",
    "Pattern Composition, repeated motifs, rhythmic structure, cohesive repetition",
    "Geometric Composition, shape-based design, orderly structure, graphic clarity",
    "Organic Composition, natural forms, fluid lines, gentle curves",
    "Minimalist Composition, sparse elements, focused simplicity, uncluttered emphasis",
    "Dynamic Composition, active arrangement, energetic layout, visual excitement",
    "Static Composition, stable structure, balanced equilibrium, calm presence",
    "Juxtaposition Composition, contrasting elements, side-by-side tension, comparative interest",
    "Color Contrast Composition, differing hues, vibrant emphasis, chromatic distinction",

    "Layering Techniques, depth management, spatial interplay, dimensional arrangement",
    "Foreground Element Shot, subject forefront, immediate attention, layered focus",
    "Background Element Shot, distant emphasis, contextual layer, scene depth",
    "Balanced Layering, even depth distribution, harmonized spatial placement",
    "Asymmetrical Layering, uneven depth distribution, dynamic spatial tension",
    "Central Layering, mid-depth focus, layered emphasis in center plane",
    "Peripheral Layering, edge-based arrangement, extended field interest, lateral depth",

    "Space and Depth Shots, manipulating perception, 3D illusion, spatial nuance",
    "Depth of Field Shot, selective focus layers, foreground clarity, blurred backdrop",
    "Shallow Depth, tight focus plane, isolated subject detail, soft background",
    "Deep Depth, extensive focus range, overall clarity, comprehensive detail",
    "Spatial Composition, 3D effect, layered distances, immersive perspective",
    "Flat Composition, minimal depth cues, 2D emphasis, graphic simplicity",

    "Framing Techniques, visual structure, thematic alignment, narrative reinforcement",
    "Balanced Framing, equal distribution, harmonious arrangement, stable layout",
    "Dynamic Framing, active placement, engaging arrangement, lively composition",
    "Narrative Framing, story-enhancing alignment, context-driven angle, plot emphasis",
    "Symbolic Framing, thematic representation, meaningful placement, interpretive depth",
    "Thematic Framing, scene-aligned structure, unified thematic elements, cohesive narrative"
]


CAMERA_MOVEMENTS = [
    "Contextual Movement, adaptive trajectory, scene-driven adjustments, variable pacing",

    "Basic Movements, foundational motion, core directional shifts, stable viewpoint",
    "Pan, horizontal pivot, sweeping view, extended lateral coverage",
    "Tilt, vertical pivot, up-down angle shift, layered vertical reveal",
    "Dolly, forward-backward track, depth variation, perspective shift",
    "Truck, side-to-side track, lateral reposition, expanded spatial awareness",
    "Zoom, focal length shift, magnification change, altered subject scale",
    "Crane, vertical lift/drop, elevated vantage, aerial dimension",
    "Handheld, unsteady shake, naturalistic vibe, immediate intimacy",
    "Steadicam, stabilized glide, smooth tracking, fluid viewpoint",
    "Tracking Shot, subject-following motion, continuous alignment, dynamic framing",
    "Arc Shot, circular path, encircling perspective, evolving angles",

    "Dynamic Movements, quick adjustments, sudden shifts, heightened impact",
    "Whip Pan, rapid horizontal pivot, blurred transition, energetic reorientation",
    "Whip Tilt, rapid vertical pivot, swift angle shift, abrupt viewpoint change",
    "Rack Focus, focus plane shift, selective clarity, layered subject emphasis",
    "Pull Back Reveal, backward movement, gradual scene exposure, unveiling context",
    "Push In, forward approach, intensified subject detail, closer inspection",
    "Roll, camera rotation, off-kilter axis, disorienting angle",
    "Boom, vertical travel, overhead descent/ascent, top-down dimension",

    "Time-Based Movements, temporal distortion, altered pace, time perception shift",
    "Time-Lapse, accelerated frames, compressed change, evolving scene",
    "Hyperlapse, moving time-lapse, spatial-temporal shift, fluid progression",
    "Slow Motion, reduced speed, elongated action, emphasized detail",
    "Fast Motion, increased speed, urgent pacing, kinetic intensity",
    "Frozen Time, static moment hold, paused action, suspended instant",
    "Bullet Time, multi-angle freeze, full 360 capture, time-suspended rotation",

    "Complex Movements, combined techniques, intricate motion, layered perspective",
    "Pan and Tilt, dual-axis pivot, compound angle shift, multi-directional scan",
    "Dolly Zoom, simultaneous dolly and zoom, warped depth, tension effect",
    "Orbit Shot, full 360 circle, evolving background, subject-centric rotation",
    "Jib Shot, crane-like arc, gentle vertical/horizontal glide, elevated framing",
    "Helicopter Shot, high aerial sweep, expansive overview, grand spatial scale",
    "Cable Cam Movement, suspended linear glide, stable overhead travel, controlled path",
    "Monorail Movement, fixed track slide, consistent direction, steady alignment",
    "Slide Shot, horizontal glide, smooth lateral shift, stable reframe",

    "Adaptive & Controlled Movements, guided direction, mechanized stability, precise adjustments",
    "Motorized Steadicam, powered stabilization, even movement, mechanical consistency",
    "Drone Movement, aerial flight, free spatial navigation, bird’s-eye perspective",
    "Gyroscopic Movement, rotational stabilization, level horizons, steady rotations",
    "Gimbal Stabilized Movement, multi-axis smoothing, controlled angles, fluid transitions",
    "RC Camera Movement, remote-guided path, consistent travel, directional precision",
    "Underwater Camera Movement, subaquatic traversal, fluid drifting, aquatic environment",
    "Macro Camera Movement, minute positional shifts, close-up framing, fine detail",
    "Wide Camera Movement, broad sweeps, extensive coverage, panoramic scope",
    "Narrow Camera Movement, restricted motion, tight framing, focused detail",

    "Creative & Expressive Movements, stylistic shifts, emotive motion, thematic emphasis",
    "Fluid Movement, smooth transitions, natural flow, graceful pace",
    "Staccato Movement, abrupt starts/stops, jittery shifts, energetic pattern",
    "Synchronized Movement, coordinated motion, subject parallel, cohesive visual timing",
    "Random Movement, erratic trajectory, unpredictable framing, spontaneous energy",
    "Narrative Movement, storytelling flow, scene-enhancing angle shifts, thematic alignment",
    "Symbolic Movement, representational motion, thematic arcs, layered meaning",
    "Expressive Movement, emotional resonance, subjective shifts, mood-driven pacing",
    "Functional Movement, purposeful adjustment, problem-solving angle, direct reposition",

    "Transition Movements, shot-to-shot linkage, bridging frames, evolving visual context",
    "Smooth Transition Movement, seamless camera shift, fluid scene merge, graceful linkage",
    "Abrupt Transition Movement, sudden change, jolting reframe, sharp scene jump",
    "Gradual Transition Movement, slow progression, gently changing perspective, extended dissolve",
    "Creative Transition Movement, inventive swaps, stylized merging, artistic shift",
    "Narrative Transition Movement, story-driven angle change, plot-bound reframe, thematic progression",

    "Advanced Tracking Movements, subject/location follow, stable pursuit, evolving focus",
    "Reverse Tracking, backward movement, retracting viewpoint, reversed follow",
    "Side Tracking, lateral follow, parallel alignment, consistent relational pacing",
    "Overhead Tracking, top-down follow, elevated alignment, detached perspective",
    "Ground Tracking, low-level follow, near-ground vantage, intimate proximity",
    "Subject Tracking, locked on main subject, consistent centering, maintained focus",
    "Environment Tracking, following setting changes, shifting backdrop, atmospheric shift",
    "Dynamic Tracking, variable speed, fluid pacing, responsive alignment",
    "Static Tracking, constant speed follow, uniform progression, steady relation",
    "Smooth Tracking, even velocity, stable pursuit, controlled follow",

    "Technical & Specialized Movements, precision execution, focused application, tailored paths",
    "Predictive Movement, anticipatory path, pre-emptive framing, foresight in angle",
    "Reactive Movement, responsive shifts, immediate adjustment, situational adaptation",
    "Controlled Movement, managed trajectory, stable execution, intentional motion",
    "Uncontrolled Movement, free-form wander, organic fluctuation, unplanned flow",
    "Automated Movement, pre-programmed route, mechanical repetition, consistent pattern",
    "Manual Movement, human-guided shifts, tactile feedback, intuitive reposition",
    "Adaptive Movement, flexible angle changes, subject-aligned adjustments, real-time correction",

    "Multi-Directional Movements, complex paths, intersecting arcs, spatial interplay",
    "Figure-Eight Movement, looping curves, layered crossing, rhythmic pattern",
    "Circular Tracking, round path travel, continuous wrap-around, evolving backdrop",
    "Linear Tracking, straight line follow, consistent direction, direct motion",
    "Diagonal Tracking, angled route, skewed perspective, non-orthogonal travel",
    "Omni-Directional Tracking, unrestricted range, full spatial freedom, all-around coverage",
    "Intersecting Tracking, crossing paths, multi-layered dimension, visual complexity",
    "Continuous Tracking, uninterrupted path, seamless follow, unbroken scene",
    "Intermittent Tracking, stop-start flow, punctuated emphasis, rhythmic interruption",

    "Composition & Framing Movements, perspective shifts, framing refinement, compositional alignment",
    "Symmetrical Movement, balanced path, mirror-like axis, harmonious flow",
    "Asymmetrical Movement, uneven route, dynamic imbalance, visually stimulating offset",
    "Easing Movement, gentle start/stop, gradual speed changes, fluid transitions",
    "Abrupt Movement, sudden shift, sharp change, immediate reframe",
    "Smooth Transition Movement, blended angle change, seamless move, integrated shift",
    "Creative Framing, artistic angle selection, unique vantage, inventive borders",
    "Rule of Thirds Framing, subject aligned on thirds, balanced layout, natural focal points"
]


SHOT_COMPOSITIONS = [
    "Contextual Framing, flexible arrangement, adaptive layout, integrates scene elements",
    "Basic Framing Techniques, foundational structure, stable layout, moderate visual complexity",
    "Rule of Thirds, aligned on grid intersections, balanced spacing, natural focal points",
    "Symmetrical, mirror-like distribution, harmonious balance, uniform visual weight",
    "Asymmetrical, uneven element placement, dynamic tension, visually stimulating",
    "Centered Composition, main subject centered, stable focus, straightforward arrangement",
    "Golden Ratio, naturally pleasing proportions, subtle balance, refined visual flow",
    "Framing, internal borders, layered depth, contained subject focus",
    "Leading Lines, directional guides, eye movement channels, focal trajectory",
    "Diagonal Composition, angled lines, energetic flow, heightened visual movement",
    "Negative Space, empty areas around subject, isolated focus, minimalist emphasis",

    "Depth and Layering, spatial dimension, overlapping elements, enhanced perspective",
    "Depth, layered planes, 3D impression, relative focus distribution",
    "Layering, multiple visual strata, complex dimensionality, textural stacking",
    "Foreground Interest, prominent frontal elements, immediate attention, layered scene",
    "Background Interest, detailed backdrop, subtle secondary focus, contextual complexity",
    "Rule of Odds, odd-numbered elements, visually pleasing grouping, balanced tension",
    "Golden Spiral, curved compositional path, fluid directional flow, organic focal lead",
    "Frame within Frame, nested boundaries, multi-level depth, concentrated viewing area",

    "Balance and Symmetry, structured harmony, proportional placement, stable design",
    "Dynamic Symmetry, active yet balanced layout, flowing order, nuanced harmony",
    "Balanced Composition, even distribution, steady visual weight, stable equilibrium",
    "Unbalanced Composition, uneven element weighting, focal emphasis, visual tension",
    "Symmetrical Balance, mirror placement, perfect equilibrium, serene order",
    "Radial Balance, elements radiating outwards, centered energy, circular harmony",
    "Visual Weight, element importance, hierarchical emphasis, selective focus",

    "Contrast and Juxtaposition, opposing elements, visual tension, heightened interest",
    "Juxtaposition, contrasting subjects, side-by-side intrigue, visual comparison",
    "Color Contrast, differing hues, intensified emphasis, vibrant focal points",
    "Light Contrast, bright vs dark, depth and drama, tonal separation",
    "Texture Contrast, smooth vs rough, tactile differentiation, enhanced complexity",
    "Shape Contrast, geometric vs organic, distinct forms, visual variety",
    "Size Contrast, large vs small elements, scale interplay, hierarchical interest",
    "Focus Contrast, sharp vs blurred areas, selective clarity, guided attention",

    "Patterns and Repetition, rhythmic elements, structured motifs, visual continuity",
    "Patterns, repeated forms, consistent rhythm, cohesive visual beat",
    "Repetition, iterated motifs, guided eye movement, stable uniformity",
    "Geometric Patterns, ordered shapes, structured regularity, neat alignment",
    "Organic Patterns, natural forms, fluid repetition, soft visual flow",
    "Texture, surface details, tactile impression, subtle complexity",

    "Visual Path and Flow, guided eye movement, directional emphasis, viewing sequence",
    "Visual Path, defined route for the eye, orchestrated viewing order, guided travel",
    "Movement, implied action lines, dynamic flow, energetic composition",
    "Rhythm, recurring elements, visual pacing, steady viewing tempo",
    "Hierarchy, layered importance, prioritized focal points, organized emphasis",
    "Harmony, unified elements, cohesive arrangement, pleasing integration",
    "Contrast, differing components, heightened attention, striking variation",
    "Emphasis, highlighted detail, dominant focus, main visual priority",

    "Shape and Form, defining structure, clear outlines, organized spatial relations",
    "Circular Composition, round elements, unifying curvature, enclosed focus",
    "Triangular Composition, three-point framework, stable configuration, balanced tension",
    "Linear Composition, straight guiding lines, direct directional flow, structured path",
    "Zigzag Composition, angled turns, energetic movement, lively direction shifts",
    "Radiant Composition, emanating lines, focal convergence, spreading energy",
    "Grid Composition, orderly division, structured layout, modular clarity",

    "Textural and Color Harmony, cohesive tones, unified surfaces, blended aesthetics",
    "Color Harmony, complementary hues, unified palette, pleasing chromatic flow",
    "Texture Harmony, similar surfaces, consistent tactile feel, integrated detail",
    "Shape Harmony, uniform forms, consistent geometry, visually aligned",
    "Color Texture Harmony, fused hue and surface, integrated richness, balanced feel",
    "Depth Texture Harmony, layered tactile cues, spatial variation, subtle complexity",

    "Advanced Focus and Detail, intricate elements, layered attention, fine distinctions",
    "Focal Point, primary area of attention, visually dominant, guiding viewer",
    "Secondary Focus, supporting interest points, nuanced layers, subtle emphasis",
    "Tertiary Focus, additional minor details, deeper inspection, extended engagement",
    "Highlight, emphasized brightness, luminous detail, accentuated area",
    "Shadow, darkened regions, contrast in tone, dimensional depth",
    "Silhouette, outlined form, backlit subject, distinct contour",
    "Reflections, mirrored imagery, doubled elements, layered perception",
    "Subframing, partial internal frames, focused enclaves, isolated interest",

    "Special Techniques, artistic variations, expressive structures, unconventional layouts",
    "Rule Breaking Composition, defies standards, unexpected placements, novelty appeal",
    "Minimalist Composition, pared-down elements, ample negative space, clarity of focus",
    "Maximalist Composition, dense, intricate arrangement, abundant detail",
    "Isolated Element, single prominent subject, uncluttered view, direct emphasis",
    "Grouped Elements, clustered subjects, collective interest, unified cluster",
    "Distributed Elements, spread-out forms, scattered attention, broad engagement",
    "Central Element, focused in center, stable anchor, immediate recognition",
    "Peripheral Elements, placed at edges, boundary interests, subtle framing"
]


TIME_OF_DAY_OPTIONS = [
    # Important Cinematic Hours (added at the top)
    "Golden Hour (After Sunrise), approx 06:00 - Radiant warm hues, gentle shadows, soft glow that enriches colors and details",
    "Golden Hour (Before Sunset), approx 18:00 - Luminous amber light, elongated shadows, a cinematic warmth that enhances texture",
    "Blue Hour (Before Dawn), approx 05:00 - Subtle blue tones, calm and quiet, silhouettes emerge softly as sky transitions",
    "Blue Hour (After Sunset), approx 19:00 - Deepening blue hues, a tranquil twilight that turns city lights into gentle halos",
    "Magic Hour - The fleeting moments around sunrise and sunset where light is soft, diffused, and visually poetic",

    # Original time slots with atmospheric descriptions
    "00:00 - Midnight, quiet world under starlit darkness, occasional distant hums, reflective city puddles",
    "01:00 - Late night, silent streets and softly glowing windows, subtle wind whispering through empty alleys",
    "02:00 - Late night, deep hush over sleeping neighborhoods, moonlit rooftops and gentle ticking clocks",
    "03:00 - Early morning, faint glow on horizon, nocturnal creatures stirring, soft hum of distant highway",
    "04:00 - Early morning, pale blue hints in eastern sky, first birdsong tentative, crisp air through open windows",
    "05:00 - Dawn begins, pastel light emerging, dew on leaves shimmering, a promising hush before the day’s rush",
    "06:00 - Sunrise, golden rays spread warmth, silhouettes sharpen, the world wakens as warmth seeps into old pavements",
    "07:00 - Morning starts, soft bustle of life, coffee aromas drifting, gentle chatter and footsteps on damp sidewalks",
    "08:00 - Morning, bright and fresh, routine hum of commutes, sunlight on faces and gentle shadows under trees",
    "09:00 - Morning, energetic air, shops opening, mild traffic under a clean azure sky, laughter in nearby parks",
    "10:00 - Late morning, full daylight vibrant, subtle warmth and cheerful colors, quiet productivity settling in",
    "11:00 - Late morning, soft buzz of midday approaching, sunlight highlighting leaves, a calm yet alert ambiance",
    "12:00 - Noon, sunlight at zenith, vivid colors at peak brightness, a pause with lunches unwrapped and shared",
    "13:00 - Early afternoon, gentle warmth, slow pace after midday meals, conversations carrying gently in breeze",
    "14:00 - Afternoon, steady and clear, subtle shadows angled, distant hum of busy streets, leaves rustling softly",
    "15:00 - Afternoon, warm air wrapping scenes in comfort, children playing in parks, a balanced, serene time",
    "16:00 - Late afternoon, golden hue intensifying, long shadows stretching, a soft calm before evening’s approach",
    "17:00 - Evening begins, mellow light turning amber, casual walks home, a hint of dinner scents floating outside",
    "18:00 - Sunset, sky painted in oranges and pinks, silhouettes softened, gentle hush as day’s energy recedes",
    "19:00 - Early evening, soft twilight glow, streetlights warming up, quiet laughter and distant music drifting",
    "20:00 - Evening, cooler breezes, lights twinkling in windows, comforting hum of homes settling down",
    "21:00 - Night starts, deepening blue velvet sky, neon signs reflecting on pavements, relaxed conversations indoors",
    "22:00 - Night, calm streets with fewer cars, warm lamplight through curtains, muffled voices and soft footsteps",
    "23:00 - Late night, hush returning, stars bright above quiet roofs, distant bark or faint train horn fading",
    "Dawn - First light, subtle glow on horizon, silhouettes emerging from darkness, nature stirring awake",
    "Morning - Start of day, crisp air and optimism, sounds of new beginnings, beams of fresh sunlight",
    "Noon - Midday, brightest and clearest point, vibrant colors fully realized, a pause in daily rhythm",
    "Afternoon - Daytime, gentle continuation, steady warmth, casual interactions under generous daylight",
    "Dusk - Twilight begins, sky awash in purples and oranges, soft hush as birds settle, a gentle sigh of the day",
    "Evening - End of day, soft lights indoors, friendly murmurs in neighborhoods, warm glows from windows",
    "Night - Darkness, quiet subtlety, mysteries and stillness, distant city lights twinkling like scattered stars",
    "Midnight - Deep night, world at its quietest, silver moonlight and secret whispers, dreams unfolding in silence"
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
        "Very Low Definition, extremely low detail, very grainy, foundational images with minimal clarity",
        "Glass Plate Quality, improved detail over earliest methods, still soft focus, monochrome plates"
    ],
    "1900s": [
        "Glass Plate Standard, sharper than experimental era, monochrome images, pre-film clarity limits",
        "Low Definition Film, early moving images, grainy texture, flickering frames, limited contrast",
        "Low-Res Panoramic, wide view capture, distorted edges, soft detail, uneven exposure"
    ],
    "1910s": [
        "Standard Definition (SD), black-and-white frames, moderate clarity for era, stable but grainy",
        "Silent Film Standard, flickering monochrome images, limited contrast, basic sharpness",
        "Plate Film Quality, higher sharpness than film, slow exposure plates, fine monochrome detail"
    ],
    "1920s": [
        "Standard Definition (SD), improved film clarity, stable monochrome frames, modest contrast",
        "Silent Film Standard, refined black-and-white images, better exposure control, moderate detail",
        "Panoramic Film Quality, wide-angle captures, still soft detail, uneven lighting, low contrast"
    ],
    "1930s": [
        "Standard Definition (SD), consistent black-and-white imagery, clearer than silent era, stable grain",
        "Film Resolution, reliable monochrome frames, moderate sharpness, limited tonal range",
        "Enhanced Silent Film Quality, steadier exposure, slightly improved detail, subdued contrast"
    ],
    "1940s": [
        "Standard Definition (SD), black-and-white and Technicolor, stable detail, iconic classic imagery",
        "Film Quality, improved lens clarity, moderate tonal depth, uniform grain structure",
        "Early Widescreen, wider aspect ratios, still limited resolution, stretched grain patterns"
    ],
    "1950s": [
        "Standard Definition (SD), early color films, stable but soft detail, moderate saturation",
        "CinemaScope Widescreen, immersive wide frames, moderate clarity, early widescreen grain",
        "Technicolor Quality, rich early color hues, moderate sharpness, vibrant saturation",
        "TV Standard Definition (480i), interlaced monochrome or color, coarse detail, household norm"
    ],
    "1960s": [
        "Standard Definition (SD), improved lenses, moderate clarity, evolving color accuracy",
        "Television 480i, interlaced color broadcasts, limited sharpness, household viewing standard",
        "CinemaScope and Panavision, wider screens, slightly improved detail, balanced framing",
        "Early 720p Equivalent (Film), experimental higher clarity, rare usage, subtle detail gains"
    ],
    "1970s": [
        "Standard Definition (SD), color broadcasts standard, moderate grain, limited fine detail",
        "Television 480i, color widely adopted, modest clarity, interlaced flicker",
        "Panavision Quality, richer color contrast, noticeable film grain, cinematic texture",
        "IMAX Film, large-format frames, much higher detail, immersive scale, specialized format"
    ],
    "1980s": [
        "Standard Definition (SD), dominated by analog TV and VHS, soft detail, signal noise",
        "VHS Resolution, 240p-equivalent, very grainy, tape artifacts, muted colors",
        "LaserDisc 480p, slightly sharper than VHS, still sub-HD, stable but limited clarity",
        "IMAX Film, exceptional clarity in large theaters, very fine detail, huge frame area",
        "Betacam SP, broadcast-quality analog video, stable colors, sub-HD resolution"
    ],
    "1990s": [
        "480p DVD Quality, progressive scan, sharper than VHS, decent color, still sub-HD",
        "720p HD (Early), early high-definition, improved clarity, brighter colors, limited adoption",
        "1080i HD (Broadcast), interlaced high-definition, finer detail, yet interlacing artifacts",
        "IMAX Film, more mainstream in specialty theaters, extremely crisp imagery, large format",
        "MiniDV 480p, compact digital tape, better than VHS, stable colors, moderate clarity"
    ],
    "2000s": [
        "720p HD, early HDTV standard, noticeable detail improvement, vibrant colors",
        "1080p Full HD, widespread adoption, sharp detail, clearer colors, larger file sizes",
        "2K Digital Cinema, cinema-level resolution, smoother detail, good color depth",
        "4K (Cinema), ultra-high resolution, very fine detail, ideal for large screens",
        "IMAX Digital, digital large format, superb clarity, expansive immersive image"
    ],
    "2010s": [
        "1080p Full HD, still common for streaming, crisp detail, vibrant colors, widely accessible",
        "2K Digital, cinema quality above HD, fine detail, suitable for theatrical projection",
        "4K UHD, ultra-sharp consumer standard, rich detail, large file sizes",
        "5K, beyond 4K, extra detail for editing and effects, limited consumer usage",
        "8K UHD, experimental top-tier clarity, extremely sharp images, early adopter phase"
    ],
    "2020s": [
        "4K UHD, standard for new content, sharp detail, wide color range, high bandwidth",
        "8K UHD, cutting-edge clarity, extremely fine detail, heavy data demands",
        "12K Cinema, ultra-fine cinema detail, massive resolution, immense processing needs",
        "1440p QHD, mid-range resolution, sharper than HD, smoother than 4K",
        "5.3K, niche format for specific devices, slightly sharper than standard 4K, moderate usage"
    ],
    "Future": [
        "8K UHD, expected mainstream high-end, super-fine detail, very large storage",
        "12K+ Cinema, future extreme clarity, highest detail for giant screens, massive data",
        "16K, hypothetical ultra-resolution, incredible detail, currently impractical",
        "32K, theoretical next-gen format, beyond known standards, extremely data-heavy"
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
    "Standard Lens, natural field of view, balanced perspective, ideal for everyday coverage",
    "Wide Angle, broader scene capture, dramatic sense of space, emphasized foreground elements",
    "Telephoto, compressed perspective, isolated subjects, perfect for distant close-ups and subtle backgrounds",
    "Macro, extreme close focus, crisp detail of tiny subjects, magnified textures and subtleties",
    "Fisheye, ultra-wide spherical view, distorted edges, dynamic and surreal compositional flair",
    "Prime, fixed focal length, sharper optics, encourages deliberate framing and movement",
    "Zoom, variable focal range, versatile framing, rapid shifts in composition without lens changes",
    "Anamorphic, widescreen aspect, distinctive horizontal flares, cinematic and expansive field of vision",
    "Tilt-Shift, selective focus plane, miniature effects, architectural corrections and creative DOF control",
    "Ultra Wide Angle, even broader scope than standard wide, grand landscapes, immersive spatial context",
    "Short Telephoto, slightly compressed perspective, flattering for portraits, balanced isolation of subject",
    "Long Telephoto, extended reach, prominent subject magnification, ideal for nature, sports, distant action",
    "Catadioptric, mirror-based design, donut-shaped bokeh highlights, unique compressed imagery",
    "Soft Focus, gentle diffusion, romantic glow, perfect for dreamy sequences or classic Hollywood style",
    "Infrared Lens, captures IR spectrum, ethereal foliage glow, surreal and otherworldly imagery",
    "UV Lens, ultraviolet sensitivity, reveals hidden patterns, scientific or forensic cinematic applications",
    "Cine Lens, built for cinema standards, smooth focus pulls, consistent color and minimal breathing",
    "Portrait Lens, medium tele focal length, flattering perspective, creamy background blur for character focus",
    "Super Telephoto, extreme reach, distant wildlife or sports, very shallow depth on far subjects",
    "Pancake Lens, ultra-compact profile, discreet and lightweight, great for run-and-gun or stealthy shots",
    "Refractive Lens, glass-based design, standard imaging, foundational element of most traditional optics",
    "Mirror Lens, reflective element inside, compact form for long focal lengths, quirky bokeh and unique aesthetics",
    "Perspective Control, shifts optical axis, straightens converging lines, crucial for architectural sequences",
    "Fish-Eye Circular, captures a full circular image, extreme distortion, experimental and playful framing"
]


HOLIDAYS = [
    # Seasonal Holidays - Cinematic moods, visual cues, thematic iconography
    "SEASONAL HOLIDAYS - New Year, festive confetti bursts, fireworks illuminating night skies, sparkling celebratory interiors",
    "SEASONAL HOLIDAYS - Valentine's Day, soft, warm lighting, rose petals, pastel palettes, romantic set dressing",
    "SEASONAL HOLIDAYS - Easter, bright spring hues, pastel egg props, gentle natural lighting in garden settings",
    "SEASONAL HOLIDAYS - Halloween, moody low-key lighting, fog machines, carved pumpkins and eerie silhouettes",
    "SEASONAL HOLIDAYS - Thanksgiving, golden autumn hues, rustic wood textures, abundant harvest tablescapes",
    "SEASONAL HOLIDAYS - Christmas, twinkling lights, lush greenery, warm interiors with soft glows and festive decor",
    "SEASONAL HOLIDAYS - Hanukkah, gentle candlelit warmth, blue and white accents, menorah focal point",
    "SEASONAL HOLIDAYS - Diwali, vibrant color grading, flickering diyas, ornate patterns and rangoli designs",
    "SEASONAL HOLIDAYS - Lunar New Year, bright reds and golds, hanging lanterns, lively street processions",
    "SEASONAL HOLIDAYS - Mardi Gras, vivid masks, bold costumes, rhythmic camera movements capturing parades",
    "SEASONAL HOLIDAYS - Palm Sunday, natural daylight, palm fronds as props, reverent processional framing",
    "SEASONAL HOLIDAYS - Good Friday, subdued tones, contemplative compositions, solemn processional imagery",
    "SEASONAL HOLIDAYS - Back to School, cheerful classroom sets, pastel chalkboards, lively youthful energy",

    # National Holidays - Reflecting cultural identity, patriotic color schemes
    "NATIONAL HOLIDAYS - Memorial Day, respectful framing of flags, muted skies, quiet ceremonial imagery",
    "NATIONAL HOLIDAYS - Labor Day, factory or cityscapes, working-class authenticity, balanced neutral lighting",
    "NATIONAL HOLIDAYS - Veterans Day, crisp uniforms, flags in soft focus, formal compositions honoring service",
    "NATIONAL HOLIDAYS - Independence Day, bold primary colors, fireworks overhead, wide community shots",
    "NATIONAL HOLIDAYS - Presidents' Day, classic formal interiors, portraits in background, stately lighting",
    "NATIONAL HOLIDAYS - Columbus Day, maritime references, sails and maps, warm sunlight on historical props",
    "NATIONAL HOLIDAYS - Flag Day, focused framing on flags billowing, symbolic mid-shots, patriotic hues",
    "NATIONAL HOLIDAYS - National Day, landmark backdrops, unified crowd shots, festive public gatherings",

    # Religious Holidays - Spiritual aura, symbolic imagery, sacred lighting
    "RELIGIOUS HOLIDAYS - Easter, soft pastel light, floral arrangements, gentle focus on religious iconography",
    "RELIGIOUS HOLIDAYS - Hanukkah, candlelit scenes, warm intimacy, menorahs as luminous focal points",
    "RELIGIOUS HOLIDAYS - Diwali, radiant diyas, rich jewel-tone color schemes, celebratory dance sequences",
    "RELIGIOUS HOLIDAYS - Ramadan, lantern-lit nights, crescent moon silhouettes, communal Iftar gatherings",
    "RELIGIOUS HOLIDAYS - Passover, table settings with symbolic foods, candlelight, intimate family focus",
    "RELIGIOUS HOLIDAYS - Yom Kippur, subdued lighting, minimalist sets, reflective close-ups on faces",
    "RELIGIOUS HOLIDAYS - Sukkot, natural outdoor structures, greenery overhead, dappled sunlight filtering in",
    "RELIGIOUS HOLIDAYS - Rosh Hashanah, honey and apples, warm early autumn light, hopeful framing",
    "RELIGIOUS HOLIDAYS - Purim, colorful costumes, festive dancing, playful camera movements",
    "RELIGIOUS HOLIDAYS - Navaratri, vibrant saris, rhythmic drum beats, energetic crowd choreography",
    "RELIGIOUS HOLIDAYS - Lent, restrained palettes, contemplative camera pacing, symbolic minimalism",

    # Cultural Celebrations - Distinctive aesthetics, culturally iconic imagery
    "CULTURAL CELEBRATIONS - St. Patrick's Day, lush green color grading, shamrocks, lively pub interiors",
    "CULTURAL CELEBRATIONS - Cinco de Mayo, bright folkloric costumes, mariachi music, colorful papel picado",
    "CULTURAL CELEBRATIONS - Oktoberfest, beer halls, rustic wooden benches, cheerful communal singing",
    "CULTURAL CELEBRATIONS - Bastille Day, French flags, bustling street scenes, Eiffel Tower silhouettes",
    "CULTURAL CELEBRATIONS - Guy Fawkes Night, bonfires and fireworks, historical costumes, shadowy evening streets",
    "CULTURAL CELEBRATIONS - Sinterklaas, Dutch canals, festive arrival by boat, traditional costumes",

    # Sports and Events - Energetic pacing, crowd atmosphere, bold iconography
    "SPORTS AND EVENTS - Super Bowl, huge stadium vistas, dynamic tracking shots of players, halftime spectacle",
    "SPORTS AND EVENTS - FIFA World Cup, global flags, passionate crowd reactions, dramatic slow-motion goals",
    "SPORTS AND EVENTS - Olympic Games, iconic stadium architecture, multi-national color palettes, graceful athleticism",
    "SPORTS AND EVENTS - NBA Finals, hardwood court reflections, intense player close-ups, team color contrasts",
    "SPORTS AND EVENTS - Wimbledon, verdant lawns, white attire, refined angles capturing precision of tennis",
    "SPORTS AND EVENTS - Stanley Cup Finals, cool arena lighting, ice reflections, emotional player close-ups",
    "SPORTS AND EVENTS - Tour de France, rolling countryside vistas, aerial tracking of cyclists, vibrant team jerseys",
    "SPORTS AND EVENTS - The Masters (Golf), manicured greens, pastel spring hues, calm steady camerawork",
    "SPORTS AND EVENTS - World Series, classic ballparks, red-stitched baseballs in close-up, cinematic flyover shots",
    "SPORTS AND EVENTS - UEFA Champions League Final, floodlit stadium, intense player focus, dramatic anthem intros",
    # ... (Add similar cinematic descriptors for remaining sports/events as needed)

    # Awareness Days - Symbolic imagery, subtle thematic cues
    "AWARENESS DAYS - International Women's Day, empowering portraits, warm inclusive lighting, communal strength",
    "AWARENESS DAYS - World Health Day, clinical clean visuals, balanced neutral palettes, caring human interactions",
    "AWARENESS DAYS - World Environment Day, lush natural landscapes, gentle sunrise lighting, hopeful green tones",
    "AWARENESS DAYS - World Mental Health Day, soft interior lighting, calm pastel hues, supportive close-ups on faces",
    # ... (Similarly refine other awareness days, focusing on subtle thematic visuals and emotional resonance)

    # Food & Drink Days - Mouthwatering close-ups, lively colors, gastronomic delight
    "FOOD & DRINK DAYS - National Pancake Day, warm kitchen sets, golden pancakes in close-up, drizzling syrup slow-motion",
    "FOOD & DRINK DAYS - National Margarita Day, vibrant glassware, bright citrus garnishes, tropical lighting cues",
    "FOOD & DRINK DAYS - National Taco Day, colorful street-food stands, diverse textures, handheld camera capturing food prep",
    "FOOD & DRINK DAYS - National Coffee Day, cozy café interiors, aromatic steam rising in shafts of morning light",
    "FOOD & DRINK DAYS - National Pizza Day, bubbling cheese in macro focus, rustic pizzeria ambience, red-checkered tablecloths",
    # ... (Continue with appealing visual and atmospheric cues for other food/drink days)

    # Family and Social Holidays - Warmth, interpersonal closeness, communal settings
    "FAMILY AND SOCIAL HOLIDAYS - Mother's Day, soft backlit vignettes, pastel florals, tender familial close-ups",
    "FAMILY AND SOCIAL HOLIDAYS - Father's Day, subtle masculine hues, sentimental objects in focus, backyard BBQ scenes",
    "FAMILY AND SOCIAL HOLIDAYS - Grandparents' Day, nostalgic sepia tints, family heirlooms, multi-generational group shots",
    "FAMILY AND SOCIAL HOLIDAYS - National Siblings Day, playful candid moments, natural daylight, shared childhood memories",
    "FAMILY AND SOCIAL HOLIDAYS - International Day of Families, balanced framing of all ages, living room gatherings, warm suburban tones",

    # Security and Legal Holidays - Symbolic integrity, formal settings
    "SECURITY AND LEGAL HOLIDAYS - World Day Against Trafficking in Persons, documentary realism, close-up on hands symbolizing hope, neutral color grades",
    "SECURITY AND LEGAL HOLIDAYS - World Press Freedom Day, newsroom sets, typewriters or digital screens in shallow focus, earnest overhead lighting",
    "SECURITY AND LEGAL HOLIDAYS - World Landmine Day, sobering landscapes, careful framing of cleared fields, subdued palettes",

    # Environmental Holidays - Nature’s vibrancy, ecological awareness
    "ENVIRONMENTAL HOLIDAYS - World Environment Day, lush greens, sweeping aerial forest shots, morning dew highlights",
    "ENVIRONMENTAL HOLIDAYS - World Earth Day, global vistas, rotating globe imagery, gentle sunrise/sunset tones",
    "ENVIRONMENTAL HOLIDAYS - World Water Day, shimmering aquatic reflections, underwater shots, crystal clarity",
    "ENVIRONMENTAL HOLIDAYS - World Oceans Day, expansive blue horizons, playful marine life close-ups, shifting coastal lighting",
    "ENVIRONMENTAL HOLIDAYS - World Wildlife Day, intimate animal close-ups, careful sound design, natural daylight",
    "ENVIRONMENTAL HOLIDAYS - World Soil Day, macro shots of earth textures, root systems, warm earthy browns",
    "ENVIRONMENTAL HOLIDAYS - World Energy Conservation Day, solar panels in sunlight, wind turbines at dawn, hopeful sustainable imagery",
    "ENVIRONMENTAL HOLIDAYS - World Migratory Bird Day, birds in flight, gentle breezes, pastel skies at dawn",
    "ENVIRONMENTAL HOLIDAYS - World Turtle Day, tranquil beach scenes, slow graceful movements, turquoise waters",
    "ENVIRONMENTAL HOLIDAYS - World Bee Day, close-ups of pollination, floral pops of color, soft buzzing ambiance",
    "ENVIRONMENTAL HOLIDAYS - World Polar Day, icy blue-white landscapes, crisp cool lighting, reflective ice surfaces",
    "ENVIRONMENTAL HOLIDAYS - World Meteorological Day, dramatic cloud formations, shifting weather patterns, time-lapse skies",

    # Miscellaneous - Various special days
    "MISCELLANEOUS - Boss's Day, office interiors, balanced neutral light, subtle hierarchical framing",
    "MISCELLANEOUS - National Pet Day, playful animal close-ups, bright natural light, cheerful home or garden backdrops",
    "MISCELLANEOUS - National Teacher Day, warm classroom tones, chalk dust motes in sunlight, thoughtful close-ups of teaching moments",
    "MISCELLANEOUS - National Nurses Day, hospital corridors with gentle overhead lighting, compassionate close-ups, soft reassuring palette",
    "MISCELLANEOUS - National Heroes Day, respectful framing of uniforms, clear symbolic imagery, flags and memorials in dignified light",
    "MISCELLANEOUS - National Volunteer Day, communal gatherings, group hugs, natural daylight outdoors",
    "MISCELLANEOUS - World Health Care Day, clinical cleanliness, caring interactions, balanced and neutral color grading",
    "MISCELLANEOUS - World Nursing Day, nurturing bedside scenes, soft bedside lamps, close-ups of gentle hands",
    "MISCELLANEOUS - World Veterinary Day, animal clinics, warm supportive interactions, pastel calming tones",
    "MISCELLANEOUS - World Blood Donor Day, medical reds and whites, careful framing of donation process, hopeful human connection"
]


SPECIFIC_MODES = [
"Documentary Mode","Action Mode"
]

CAMERAS = {
    "Experimental and Proto (Pre-1900s)": [
        "Nicéphore Niépce Camera Obscura, made in 1826, extremely long exposures, single-frame capture, low detail, monochrome image",
        "Louis Daguerre Daguerreotype Camera, made in 1839, metal plate photos, fine detail for era, lengthy exposure, monochrome image",
        "William Henry Fox Talbot Calotype Camera, made in 1840, paper negatives, soft detail, monochrome, low contrast",
        "Frederick Scott Archer Wet Collodion Camera, made in 1851, glass plates, improved clarity, reduced exposure vs previous, monochrome image",
        "E. & H.T. Anthony & Co. Folding Camera, made in 1854, large format plates, foldable design, limited detail, monochrome",
        "Thomas Sutton Panoramic Camera, made in 1859, wide-angle capture, curved plate, uneven exposure, monochrome image",
        "Jules Duboscq Stereoscopic Camera, made in 1860, dual-lens for stereoscopic pairs, low detail, monochrome images",
        "Kodak Kodak Box Camera, made in 1888, roll film, simple fixed focus, grainy, low resolution, monochrome",
        "Charles Bennett Gelatin Dry Plate Camera, made in 1878, dry plates, shorter exposure than wet collodion, moderate detail, monochrome",
        "Eadweard Muybridge Zoopraxiscope, made in 1879, rotating glass discs, low-frame-rate imagery, silhouetted motion, monochrome",
        "George Eastman Roll Film Camera, made in 1885, flexible film roll, soft detail, low resolution, monochrome",
        "Joseph Nicéphore Niépce Physautotype, made in 1832, lavender oil-based process, extremely fragile images, faint detail, monochrome",
        "Thomas Sutton First Panoramic Camera, made in 1859, curved lens, limited sharpness at edges, monochrome",
        "William Friese-Greene Chronophotographic Camera, made in 1889, early motion frames, low resolution, flickering imagery, monochrome",
        "Le Prince Single-Lens Camera, made in 1888, early motion picture film strips, low detail, unstable frame rate, monochrome",
        "Anschütz Tachyscope, made in 1887, sequential glass discs, limited frames, flickering motion, monochrome",
        "Etienne-Jules Marey Chronophotographic Gun, made in 1882, multiple exposures per plate, low detail, monochrome motion studies",
        "Simon Wing Multiplying Camera, made in 1870s, multiple exposures on one plate, very low detail, monochrome",
        "Ives Kinetoscope Prototype, made in 1889, peephole viewer, low-resolution frames, monochrome",
        "Kodak Pocket Kodak Camera, made in 1895, small roll film, grainy, soft detail, monochrome"
    ],
    "1900s": [
        "Graflex Graflex Reflex Camera, made in 1902, large format plates, single-lens reflex view, moderate detail, monochrome",
        "Kodak Brownie, made in 1900, box roll film, very grainy, low detail, monochrome",
        "Contessa-Nettel Nettel Camera, made in 1903, folding plate, moderate detail, monochrome, limited shutter speeds",
        "Kodak Vest Pocket Kodak, made in 1907, small roll film, grainy snapshots, monochrome, low detail",
        "Thornton-Pickard Ruby Reflex, made in 1905, mirror reflex system, moderate detail, monochrome, slower shutter",
        "Kodak Folding Pocket Camera, made in 1906, compact bellows, low detail, monochrome roll film",
        "Kodak No. 2 Folding Autographic Brownie, made in 1909, roll film with note area, grainy, low detail, monochrome",
        "Goerz Tenax, made in 1905, sharp lens for plates, moderate detail, monochrome, heavy setup",
        "Ica Reflex Camera, made in 1909, plate camera with mirror, moderate detail, monochrome, limited speeds",
        "Graflex Auto Graflex, made in 1906, quick plate changes, moderate detail, monochrome, bulky",
        "Kodak No. 3A Folding Pocket, made in 1903, roll film, basic resolution, monochrome snapshots",
        "Conley Universal Camera, made in 1907, plate camera, sharp portrait detail, monochrome, heavy",
        "Kodak Panoram No. 4, made in 1904, panoramic film, uneven exposure, low detail, monochrome",
        "Ernemann Stereo Piccolette, made in 1908, stereo pairs, moderate detail, monochrome, delicate alignment",
        "Folmer & Schwing Century Studio Camera, made in 1902, large format plates, very sharp stills, monochrome, immobile",
        "Graflex Revolving Back Camera, made in 1907, rotating back, moderate detail, monochrome, large plates",
        "Eastman Kodak Panoram Kodak, made in 1900, panoramic roll film, low detail, monochrome",
        "Kodak Autographic Kodak Jr., made in 1909, note-on-film feature, low detail, monochrome roll film",
        "Ica Ideal Camera, made in 1908, folding plate, moderate detail, monochrome, fragile",
        "Carl Zeiss Jena, made in 1901, high-quality lens, sharp monochrome detail, limited shutter speeds"
    ],
    "1910s": [
        "Ernemann Ermanox, made in 1910, glass plates, fast lens for low light, grainy monochrome",
        "Kodak Vest Pocket Autographic, made in 1912, small roll film, low detail, monochrome snapshots",
        "Contessa Nettel Deckrullo, made in 1910, focal plane shutter, moderate detail, monochrome, fragile",
        "Kodak Brownie No. 2, made in 1913, box camera, grainy low detail, monochrome snapshots",
        "Goerz Anschütz Camera, made in 1911, high-speed shutter, moderate detail, monochrome",
        "Ica Volta Reflex, made in 1915, early reflex viewing, moderate detail, monochrome, heavy",
        "Kodak Autographic Special, made in 1914, note-on-film feature, grainy low detail, monochrome",
        "Ernemann Stereo Camera, made in 1916, stereo images, moderate detail, monochrome, delicate",
        "ICA Kinamo, made in 1919, early motion camera, short takes, grainy monochrome frames",
        "Kodak Kodak Panoram, made in 1910, wide format, low detail, monochrome, uneven exposure",
        "Kodak Brownie Autographic, made in 1915, note-on-film, grainy monochrome snapshots",
        "Zeiss Ikon Miroflex, made in 1917, studio plate camera, sharp monochrome detail, fragile build",
        "Goerz Tenax, made in 1918, plate camera, sharp monochrome detail, heavy",
        "Kodak No. 3A Autographic, made in 1919, compact roll film, low detail, monochrome",
        "Voigtländer Avus, made in 1915, plate format, moderate detail, monochrome, limited lenses",
        "Kodak Pocket Kodak, made in 1910, small roll film, low detail, monochrome snapshots",
        "Ernemann Ermanox Folding Camera, made in 1917, low-light lens, moderate detail, monochrome, delicate",
        "Thornton-Pickard Imperial, made in 1911, large format, sharp monochrome portraits, static",
        "Kodak Vest Pocket Kodak, made in 1915, small roll film, grainy monochrome",
        "Folmer & Schwing Century Graphic, made in 1910, large format plates, detailed landscapes, monochrome, requires tripod"
    ],
    "1920s": [
        "Leica Leica I, made in 1925, 35mm film, compact size, fine grain monochrome, sharper detail",
        "Ernemann Ermanox, made in 1924, fast lens, low-light monochrome, moderate detail",
        "Akeley Akeley Gyro, made in 1923, stabilized 35mm footage, silent era, moderate detail, monochrome",
        "Kodak Brownie 2A, made in 1920, simple roll film, grainy low-detail monochrome",
        "Mitchell Standard 35mm, made in 1929, studio 35mm footage, stable frames, fine grain monochrome",
        "Kodak No. 2 Folding Autographic, made in 1923, roll film, grainy monochrome, basic detail",
        "Pathé Pathé-Baby, made in 1922, 9.5mm amateur format, grainy, low detail monochrome",
        "Graflex Super D, made in 1923, reflex viewing, plate film, sharp monochrome still frames",
        "Kodak Autographic Vest Pocket, made in 1925, small roll film, grainy low-detail monochrome",
        "Contessa-Nettel Cocarette, made in 1926, folding plate camera, moderate detail, monochrome",
        "Leica Leica II, made in 1928, 35mm film, interchangeable lenses, fine grain monochrome",
        "Debrie Parvo L, made in 1921, compact studio camera, stable silent-era 35mm frames, monochrome",
        "Bell & Howell Filmo 70, made in 1923, 16mm film, crank-driven, moderate detail monochrome",
        "Kodak Folding Pocket Camera, made in 1925, roll film, grainy monochrome, basic detail",
        "Ica Kinamo, made in 1926, portable 35mm motion camera, grainy silent frames, monochrome",
        "Kodak Rainbow Hawkeye Vest Pocket, made in 1929, small roll film, grainy monochrome, decorative body",
        "Pathé Pathé 28mm Camera, made in 1922, 28mm format, moderate grain, monochrome, limited frames",
        "Bell & Howell Eyemo 35mm, made in 1925, handheld 35mm filming, moderate detail, monochrome",
        "Newman-Sinclair Autokine, made in 1927, early handheld 35mm, grainy monochrome frames",
        "Kodak Pocket Kodak, made in 1928, roll film, low detail monochrome snapshots"
    ],
    "1930s": [
        "Bell & Howell 2709 Standard, made in 1930, studio 35mm camera, fine grain monochrome, stable image",
        "Arriflex Kinarri 35, made in 1937, handheld 35mm, moderate detail, monochrome, mechanical shutter",
        "Kodak Cine-Kodak Eight Model 20, made in 1932, 8mm film, grainy, low-detail monochrome frames",
        "Eyemo 35mm Camera, made in 1936, handheld 35mm, moderate detail, monochrome, limited steadying",
        "Kodak Brownie Movie Camera, made in 1935, 8mm film, coarse grain, low detail monochrome",
        "Debrie Parvo L, made in 1934, silent-era 35mm, stable framing, moderate detail monochrome",
        "Mitchell Standard 35mm, made in 1930, studio camera, fine monochrome detail, no sound capture",
        "Newman-Sinclair Autokine 35mm, made in 1931, handheld 35mm, grainy monochrome, mechanical drive",
        "Kodak Cine-Kodak Model K, made in 1934, 16mm film, grainy low-detail monochrome",
        "Vinten Model H 35mm, made in 1939, stable 35mm frames, fine monochrome detail, heavy",
        "Agfa Movector 16mm, made in 1933, 16mm film, moderate detail, monochrome, basic optics",
        "De Vry Standard 35mm, made in 1931, durable 35mm, moderate monochrome detail, crank-driven",
        "Leica 16mm Film Camera, made in 1936, 16mm format, moderate detail, monochrome, limited optics",
        "Bell & Howell Filmo 70-D, made in 1935, 16mm film, moderate detail, monochrome, portable",
        "Newman-Sinclair Autokine, made in 1934, handheld 35mm, moderate detail, monochrome, basic lens set"
    ],
    "1940s": [
        "Mitchell BNC Camera, made in 1941, 35mm studio camera, fine monochrome detail, stable frames",
        "Arriflex 35 II, made in 1946, compact 35mm film, moderate detail monochrome, improved handling",
        "Bell & Howell Filmo 70, made in 1940, 16mm film, crystal sync motor, low detail monochrome, old film",
        "Kodak Brownie 8mm, made in 1946, 8mm film, grainy low-detail monochrome, simple mechanism",
        "Bolex H16, made in 1941, 16mm film, moderate detail monochrome, manual controls",
        "Kodak Cine-Kodak Special II, made in 1948, 16mm film, moderate detail monochrome, basic footage",
        "Bell & Howell 16mm Magazine Camera, made in 1940, magazine-loaded 16mm, grainy monochrome, limited detail",
        "Mitchell BNC Reflex, made in 1947, 35mm studio camera, sharp monochrome detail, reflex viewing",
        "De Vry Sound Camera, made in 1945, 35mm film with audio recording capability, moderate detail monochrome",
        "Revere Model 99 8mm, made in 1949, 8mm film, grainy low-detail monochrome frames",
        "Eclair Cameflex, made in 1947, 35mm camera, moderate detail monochrome, compact for era",
        "Kodak Brownie Movie Camera, made in 1946, 8mm film, grainy low-detail monochrome",
        "Pathé WEBO M 16mm, made in 1942, 16mm film, moderate detail monochrome, durable construction",
        "Wilart 16mm Camera, made in 1948, 16mm film, basic monochrome, low detail",
        "Mitchell BNC Sound Camera, made in 1949, 35mm sync sound, fine monochrome detail",
        "Kodak Cine-Kodak 8, made in 1946, consumer 8mm, grainy monochrome frames, simple",
        "Bell & Howell 70DR, made in 1944, 16mm film, moderate grain monochrome, compact",
        "Revere 16mm Cine Camera, made in 1943, 16mm film, basic monochrome detail, hand-cranked",
        "De Vry 35mm Studio Camera, made in 1947, 35mm film, moderate monochrome detail, rugged build"
    ],
    "1950s": [
        "Mitchell BNCR, made in 1952, 35mm studio camera, sharp monochrome detail, stable imaging",
        "Arriflex 16ST, made in 1953, 16mm film, moderate detail, monochrome, steady movement",
        "Kodak Brownie 8mm Movie Camera II, made in 1956, 8mm film, grainy low-detail monochrome",
        "Bell & Howell 70DR, made in 1951, 16mm film, moderate detail monochrome, robust build",
        "Eclair NPR, made in 1958, 16mm camera, moderate detail monochrome, portable for era",
        "Bolex H16 Reflex, made in 1956, 16mm film, moderate detail monochrome, reflex viewfinder",
        "Beaulieu R16, made in 1959, 16mm film, improved clarity, moderate grain monochrome, electric drive",
        "Canon Cine 8T, made in 1957, 8mm film, grainy low-detail monochrome, simple optics",
        "Kodak Brownie Movie Camera, made in 1952, 8mm film, grainy low-detail monochrome",
        "Mitchell BNCR, made in 1955, 35mm studio camera, sharp monochrome detail, stable frames",
        "Pathé WEBO M 16mm, made in 1954, 16mm film, moderate detail monochrome, reliable",
        "Revere Eye-Matic EE127, made in 1958, small-format film, grainy monochrome, basic detail",
        "Bolex H16 M, made in 1959, 16mm film, moderate detail monochrome, manual operation",
        "Minolta Autopak-8 D6, made in 1959, 8mm film, grainy low-detail monochrome, simple mechanism",
        "Kodak 16mm Magazine Camera, made in 1951, magazine-fed 16mm, basic monochrome detail",
        "Beaulieu R16, made in 1959, 16mm film, moderate clarity, monochrome, electric motor",
        "Canon Cine 8T, made in 1957, 8mm film, low detail monochrome, basic optics",
        "Mitchell 70DR, made in 1950, rugged 35mm, moderate detail monochrome, location-ready",
        "Eclair NPR, made in 1958, 16mm film, moderate monochrome detail, portable design",
        "Kodak Brownie Movie Camera II, made in 1953, 8mm film, grainy low-detail monochrome"
    ],
    "1960s": [
        "Panavision Silent Reflex, made in 1962, 35mm film, quieter mechanics, stable monochrome or early color",
        "Arriflex 35 IIC, made in 1964, 35mm film, compact body, moderate detail, possible early color",
        "Bolex H16 EBM, made in 1965, 16mm film, electric motor drive, moderate detail, early color possible",
        "Canon Scoopic 16mm, made in 1965, 16mm film, built-in metering, moderate detail, possible color",
        "Beaulieu R16 Electric, made in 1965, 16mm film, electric drive, moderate detail, early color stocks",
        "Eclair ACL, made in 1967, 16mm film, lightweight body, moderate detail, possibly color film",
        "Mitchell Mark II, made in 1967, 35mm studio camera, fine detail, possible color negative",
        "Kodak Instamatic M2, made in 1963, small cartridge film, grainy low-detail color or monochrome",
        "Various Super 8 Cameras, starting 1965, Super 8 film, grainy low-detail color, simple exposure",
        "Nizo S8, made in 1968, Super 8 film, moderate grain, basic color, fixed lens",
        "Bell & Howell Filmosonic XL, made in 1969, Super 8 film, grainy color, basic sound strip",
        "Minolta Autopak-8 D12, made in 1969, Super 8 film, moderate grain color, simple focus",
        "Yashica Super-8 Electro, made in 1969, Super 8 film, grainy color, automatic exposure",
        "Chinon Super 8, made in 1969, Super 8 film, low detail color, simple optics",
        "Agfa Movexoom 6, made in 1968, Super 8 film, moderate grain color, basic zoom"
    ],
    "1970s": [
        "Panavision Panaflex, made in 1972, 35mm film, quieter operation, high-quality color negative",
        "Arriflex 35BL, made in 1972, 35mm film, relatively lightweight, steady color frames",
        "Aaton 7 LTR, made in 1978, 16mm film, quieter mechanism, moderate detail color, sync sound",
        "Beaulieu 4008 ZM II Super 8, made in 1971, Super 8 film, moderate grain color, better lenses",
        "Canon Auto Zoom 814 Electronic, made in 1973, Super 8 film, moderate detail color, electric zoom",
        "Eclair NPR, made in 1971, 16mm film, moderate detail color, handheld use",
        "Mitchell VistaVision, made in 1976, wide-gauge 35mm, high detail color, larger frame",
        "Bolex Pro 16mm, made in 1970, 16mm film, moderate detail color, sturdy build",
        "Elmo Super 110R, made in 1974, Super 8 film, grainy color, basic zoom lens",
        "Nikon R10 Super 8, made in 1973, Super 8 film, improved color detail, stable frames",
        "Minolta XL-601 Super 8, made in 1976, Super 8 film, moderate grain color, automatic exposure",
        "Chinon Pacific 200/12 SMR, made in 1975, Super 8 film, grainy color, simple controls",
        "Pathé DS8 Reflex, made in 1973, Double Super 8 film, moderate grain color, manual focus",
        "Leicina Super RT1, made in 1974, Super 8 film, moderate detail color, fixed lens",
        "Sankyo XL 620 Super 8, made in 1978, Super 8 film, grainy color, simple zoom"
    ],
    "1980s": [
        "Arriflex 765, made in 1989, 65mm film, high-resolution color, very sharp detail",
        "Panavision Panaflex Gold, made in 1981, 35mm film, fine detail color, quiet mechanism",
        "Aaton XTR Prod, made in 1985, 16mm film, stable color footage, sync sound ready",
        "Sony Betacam SP, made in 1986, analog video tape, improved color detail, broadcast quality",
        "Canon L2 Hi8, made in 1988, Hi8 tape, moderate color detail, interlaced video",
        "JVC GR-C1, made in 1984, VHS-C tape, low resolution color, consumer video",
        "Arriflex SRII, made in 1982, 16mm film, moderate color detail, sync sound capable",
        "Panasonic WV-F250, made in 1988, ENG camera, analog video, moderate color detail",
        "Bolex H16 SBM, made in 1980, 16mm film, moderate detail color, manual operation",
        "Sony Video8 Handycam CCD-M8U, made in 1985, Video8 tape, low resolution color, small sensor",
        "Hitachi VK-C820, made in 1989, VHS camcorder, low resolution color, consumer grade",
        "JVC KY-1900, made in 1983, tube-based video camera, moderate color detail, broadcast use",
        "Ikegami HL-79, made in 1985, tube camera, good color detail for era, heavy setup",
        "Bell & Howell 2146 XL Super 8, made in 1981, Super 8 film, grainy color, basic exposure"
    ],
    "1990s": [
        "Panavision Millennium, made in 1997, 35mm film, industry-standard high detail color, stable images",
        "Arriflex 435, made in 1995, 35mm film, high-speed capable, fine color detail",
        "Aaton 35-III, made in 1991, 35mm film, handheld capable, rich color detail",
        "Sony Digital Betacam DVW-700WS, made in 1993, digital tape, improved color resolution, broadcast standard",
        "Canon XL1 MiniDV, made in 1997, MiniDV tape, low-res digital color, progressive frames",
        "Sony DCR-VX1000 MiniDV, made in 1995, MiniDV tape, consumer-level digital color, limited resolution",
        "Arriflex 235, made in 1999, 35mm film, lightweight, crisp color detail",
        "Panasonic AG-DVX100, made in 1999, DV tape, progressive frames, moderate digital color detail",
        "Ikegami HL-V73, made in 1994, broadcast camera, good color detail, standard definition",
        "JVC GY-DV500, made in 1999, DV format, moderate color detail, SD resolution",
        "Sony Betacam SX DNW-9WS, made in 1996, digital tape format, stable color detail, SD resolution"
    ],
    "2000s": [
        "RED One, made in 2007, digital cinema, up to 4K resolution, RAW files, wide dynamic range",
        "Panavision Genesis, made in 2005, digital cinema, Super 35 sensor, HD resolution, good color depth",
        "Arriflex D-20, made in 2005, digital cinema prototype, single sensor HD, moderate dynamic range",
        "Sony CineAlta F900, made in 2000, HD digital tape, 1080p resolution, early digital color",
        "Canon EOS 5D Mark II, made in 2008, DSLR video, 1080p HD, full-frame sensor, moderate dynamic range",
        "Panasonic AG-HVX200, made in 2005, P2 HD digital, 720p/1080i, moderate color detail",
        "Sony PMW-EX1, made in 2007, XDCAM EX format, 1080p HD, 1/2\" sensors, decent low-light",
        "RED Epic, made in 2009, digital cinema, up to 5K resolution, RAW workflow, wide latitude",
        "ARRI Alexa Classic, made in 2009, digital cinema, HD/2K, high dynamic range, natural color",
        "Blackmagic Cinema Camera, made in 2008, digital capture, 2.5K RAW, limited low-light",
        "Canon EOS 7D, made in 2009, DSLR video, 1080p HD, APS-C sensor, moderate detail",
        "GoPro HD Hero, made in 2004, compact HD camera, 1080p, wide-angle, limited dynamic range",
        "Sony α7S, made in 2008, mirrorless digital, 1080p video, excellent low-light sensitivity",
        "Panasonic Lumix GH1, made in 2009, Micro Four Thirds sensor, 1080p HD, moderate dynamic range"
    ],
    "2010s": [
        "ARRI Alexa Mini, made in 2015, Super 35 digital cinema, up to 4K UHD, wide dynamic range, natural color",
        "RED Weapon, made in 2016, digital cinema, up to 8K RAW, high dynamic range, flexible frame rates",
        "Sony Venice, made in 2017, full-frame digital cinema, up to 6K, high dynamic range, rich color",
        "Blackmagic URSA Mini Pro, made in 2017, Super 35 digital, up to 4.6K RAW, decent dynamic range",
        "Canon EOS C300 Mark II, made in 2015, Super 35 digital, 4K DCI, wide dynamic range, rich color",
        "Panasonic VariCam LT, made in 2016, Super 35 digital, 4K, dual ISO, wide dynamic range",
        "DJI Inspire 2 with X7 Camera, made in 2017, Super 35 digital aerial, up to 6K RAW, good dynamic range",
        "GoPro HERO7 Black, made in 2018, action cam, 4K60, small sensor, limited dynamic range",
        "Sony α7 III, made in 2018, full-frame digital, up to 4K30, good low-light, moderate dynamic range",
        "Panasonic Lumix GH5, made in 2017, Micro Four Thirds, 4K60, 10-bit color, moderate dynamic range",
        "Blackmagic Pocket Cinema Camera 4K, made in 2018, MFT sensor, 4K RAW, limited low-light performance",
        "Sony FS7 II, made in 2016, Super 35 digital, 4K60, wide dynamic range, robust codec",
        "RED Raven, made in 2016, digital cinema, 4.5K RAW, moderate dynamic range, lightweight"
    ],
    "2020s": [
        "ARRI Alexa LF, made in 2020, large format digital, up to 4.5K, very wide dynamic range, rich color",
        "RED Komodo 6K, made in 2020, compact digital cinema, 6K RAW, moderate dynamic range",
        "Sony FX9, made in 2020, full-frame digital cinema, 4K60, 10-bit color, high sensitivity",
        "Canon EOS C500 Mark II, made in 2020, full-frame digital, 5.9K RAW, wide dynamic range, natural color",
        "Blackmagic URSA Mini Pro 12K, made in 2020, Super 35 digital, 12K RAW, extreme resolution, heavy data",
        "Panasonic Lumix S1H, made in 2020, full-frame digital, 6K, 10-bit color, good low-light",
        "Sony α7S III, made in 2020, full-frame digital, 4K120, excellent low-light, wide dynamic range",
        "DJI Ronin 4D, made in 2021, integrated gimbal camera, up to 6K ProRes RAW, stable image",
        "GoPro HERO10 Black, made in 2021, action cam, 5.3K60, small sensor, limited dynamic range",
        "Sony FX6, made in 2020, full-frame digital, 4K120, 10-bit 4:2:2, high sensitivity, wide latitude",
        "RED V-Raptor ST, made in 2021, digital cinema, 8K RAW, high dynamic range, high frame rates",
        "Canon EOS C70, made in 2020, Super 35 digital, 4K DCI 10-bit, wide dynamic range, compact body",
        "Panasonic Lumix GH6, made in 2022, Micro Four Thirds, up to 5.7K, 10-bit color, moderate low-light"
    ],
    "Future": [
        "Sony A1Z, hypothetical future, up to 16K, AI-driven autofocus, wide dynamic range, advanced codecs",
        "ARRI Nova, hypothetical future, AI-powered processing, ultra-high resolution, modular sensor blocks",
        "RED Orion 12K, hypothetical future, beyond 12K resolution, deep color, extreme data rates",
        "Canon CinemaX 16K, hypothetical future, 16K resolution, HDR capture, very high data demands",
        "Blackmagic Infinity Pro, hypothetical future, 8K+ RAW, AI-assisted grading, affordable high resolution",
        "DJI Phantom Cinema, hypothetical future, 8K+ HDR aerial video, ultra-stable gimbal, massive data",
        "Nikon X12 Vision, hypothetical future, AI-driven sensor, real-time image enhancements, ultra-high resolution",
        "Panasonic Genesis 32K, hypothetical future, 32K resolution, next-gen AI processing, enormous file sizes",
        "GoPro FutureGo 12K, hypothetical future, 12K action video, waterproof, extremely high resolution",
        "RED Helium Infinity, hypothetical future, 10K at high frame rates, RAW files, huge storage needs",
        "Canon EOS A6, hypothetical future, 10K video, computational imaging, intensive processing",
        "Panasonic Lumix VR, hypothetical future, 8K VR video, advanced stabilization, specialized capture",
        "DJI Inspire FutureX, hypothetical future, 16K aerial footage, AI horizon leveling, colossal files",
        "ARRI AI Max, hypothetical future, 12K+ resolution, AI-enhanced color, complexity in workflow"
    ]
}

DECADES = sorted(CAMERAS.keys())

selected_decade = "2020s"
print(CAMERAS[selected_decade])


# Decades list for UI or sorting purposes
DECADES = sorted(CAMERAS.keys())

WILDLIFE_ANIMALS = [

    # MAMMALS (12)
    "MAMMALS - Lion, African savanna pride, golden mane in warm sunset light, dust motes sparkling as they rest",
    "MAMMALS - Elephant, vast Serengeti plains, tusked silhouettes at watering holes, soft amber glow on wrinkled skin",
    "MAMMALS - Tiger, Asian jungles, orange-black stripes stalking through lush green, dappled sun filtering through leaves",
    "MAMMALS - Wolf, northern forests, howling under moonlight, pack unity against snowy backdrops, frosty breath visible",
    "MAMMALS - Leopard, African woodlands, spotted rosettes blending with tree shadows, poised gracefully on a branch",
    "MAMMALS - Gorilla, misty mountain forests, gentle family groups, expressive eyes in dim emerald canopies",
    "MAMMALS - Panda, Chinese bamboo groves, black-white fur amid green shoots, calm feeding in diffuse forest light",
    "MAMMALS - Giraffe, savanna horizons, elongated necks browsing treetops, pastel sunset hues stretching behind them",
    "MAMMALS - Kangaroo, Australian bush, bounding through dust and eucalyptus scents, silhouettes in late afternoon sun",
    "MAMMALS - Polar Bear, Arctic ice floes, white fur contrasting blue-white seascapes, low polar sun highlights breath",
    "MAMMALS - Bison, North American plains, shaggy shoulders in crisp morning air, grazing under expansive sky",
    "MAMMALS - Orangutan, Bornean rainforest canopy, reddish fur swinging amid lush leaves, sunbeams drifting in humidity",

    # BIRDS (12)
    "BIRDS - Bald Eagle, North American skies, regal wingspan soaring over river valleys, crisp morning clarity",
    "BIRDS - Owl (Great Horned), woodland dusk, silent flight beneath purpling skies, amber eyes in twilight hush",
    "BIRDS - Peacock, South Asian gardens, iridescent tail fanned out, vibrant jewel tones in bright midday light",
    "BIRDS - Macaw (Scarlet), Amazon canopy, vivid reds and blues perched amid emerald leaves, distant waterfall hum",
    "BIRDS - Penguin (Emperor), Antarctic ice fields, huddled colonies under gentle snow, monochrome serenity",
    "BIRDS - Hummingbird, tropical blossoms, wings a shimmering blur, sipping nectar in sunlit floral sprays",
    "BIRDS - Kingfisher, riverbanks, cobalt and orange plumage diving for fish, water droplets freeze in sunlight",
    "BIRDS - Toucan, Central American rainforest, oversized colorful bill, misty morning greens and distant calls",
    "BIRDS - Kiwi, New Zealand forest floor, shy nocturnal forager in leaf litter, moonlit silhouettes",
    "BIRDS - Flamingo, coastal lagoons, pink reflections wading in still waters, pastel dawn skies overhead",
    "BIRDS - Raven, northern woodlands, glossy black feathers stark against snow-laced branches, quiet winter air",
    "BIRDS - Swan (Trumpeter), tranquil lakes, graceful white curves mirrored, early mist and soft diffused light",

    # REPTILES (12)
    "REPTILES - Nile Crocodile, African river’s edge, still and watchful, reflections shimmering in midday heat",
    "REPTILES - Green Iguana, tropical forests, vivid scales basking in dappled sun, leafy textures all around",
    "REPTILES - Komodo Dragon, Indonesian isles, massive lizard on dusty terrain, heat haze and sparse shrubs",
    "REPTILES - Sea Turtle (Green), coral reef, gliding through turquoise beams, fish schooling in gentle currents",
    "REPTILES - Chameleon, forest understory, shifting hues on rough bark, slow deliberate steps in filtered light",
    "REPTILES - Python (Reticulated), Asian rainforest floor, coiled grace amid green gloom, soft humidity",
    "REPTILES - King Cobra, jungle undergrowth, hood flared, cautious sway as sunlight pierces dense foliage",
    "REPTILES - Gecko (Leopard), arid scrub, spotted body blending with sandy rock under warm dusk glow",
    "REPTILES - Bearded Dragon, Australian desert, spiny throat display backlit by amber horizon",
    "REPTILES - Gharial, Indian rivers, slender snout basking on a sandy bank, shimmering midday reflections",
    "REPTILES - Frilled Lizard, outback clearing, dramatic frill spread against red dust, low-angle sun",
    "REPTILES - Anaconda, Amazon wetlands, immense form partially submerged, green watery light filtering down",

    # FISH (12)
    "FISH - Clownfish, Indo-Pacific reefs, orange-white stripes dance amid sea anemones, sunlight rippling overhead",
    "FISH - Blue Tang, coral gardens, electric sapphire body weaving through vibrant sponges and soft corals",
    "FISH - Salmon, northern rivers, silver leaps against waterfalls, forest reflections shimmer on surface",
    "FISH - Great White Shark, open ocean, powerful silhouette below sunbeams, deep cobalt gradients fading into blue",
    "FISH - Angelfish (Emperor), tropical reef, yellow-blue patterns flutter near coral heads, gentle currents",
    "FISH - Swordfish, pelagic waters, sleek body cutting swift arcs, faint sun far above, plankton drifting",
    "FISH - Manta Ray, warm seas, winged form gliding gracefully, dappled surface light dancing on its back",
    "FISH - Pufferfish, reef nook, inflating into spiky orb, playful textures in subtle emerald underwater glow",
    "FISH - Lionfish, Indo-Pacific coral, ornate fins fanned out, drifting elegantly in shifting pastel lighting",
    "FISH - Moray Eel, rocky crevice, greenish elongated form peering from dim cave, subtle drifting plankton",
    "FISH - Whale Shark, tropical open waters, spotted giant feeding on plankton, snorkelers dwarfed by scale",
    "FISH - Barracuda, clear lagoon shallows, sleek silver predator reflecting sand patterns beneath sunlight",

    # INSECTS (12)
    "INSECTS - Monarch Butterfly, flowered meadow, vibrant orange-black wings flutter in gentle summer breeze",
    "INSECTS - Dragonfly, pond margins, iridescent wings catching sunlight, hovering over reeds and lily pads",
    "INSECTS - Praying Mantis, garden foliage, angular form poised patiently, morning dew sparkles on leaves",
    "INSECTS - Honeybee, blooming orchard, fuzzy body dusted with pollen, humming softly among pastel petals",
    "INSECTS - Firefly, twilight woodland, bioluminescent flickers dance in humid dusk, magical pinpoints of light",
    "INSECTS - Ladybug, green leaf surface, red shell and black dots pop against lush background, midday warmth",
    "INSECTS - Leafcutter Ant, tropical forest floor, carrying leaf fragments overhead in filtered green glow",
    "INSECTS - Stick Insect, dense shrubbery, twig-like disguise swaying gently, subtle wind rustling leaves",
    "INSECTS - Atlas Moth, Southeast Asian rainforest, enormous patterned wings resting on mossy trunk at dawn",
    "INSECTS - Rhinoceros Beetle, tropical nights, horned silhouette under moonlight, distant chorus of cicadas",
    "INSECTS - Luna Moth, nighttime garden, pale green wings glowing faintly under starlight and soft shadows",
    "INSECTS - Bumblebee, cottage garden blooms, round striped body drifting among lavender and rose scents",

    # ARACHNIDS (12)
    "ARACHNIDS - Orb Weaver Spider, early dawn meadow, dew-kissed web geometry glinting against rising sun",
    "ARACHNIDS - Jumping Spider, garden shrub, curious eyes and tiny form isolated by shallow depth of field",
    "ARACHNIDS - Tarantula, rainforest floor, furry legs lit by a sun shaft penetrating dense green canopy",
    "ARACHNIDS - Scorpion, desert night, UV fluorescence under blacklight, starlit dunes and quiet wind",
    "ARACHNIDS - Wolf Spider, grassy field, camouflaged pelt and subtle textures revealed in side-angle sun",
    "ARACHNIDS - Crab Spider, flower petals, ambush predator blending perfectly, warm afternoon backlighting",
    "ARACHNIDS - Black Widow, quiet barn corner, red hourglass marking under a dusty shaft of golden light",
    "ARACHNIDS - Harvestman (Daddy Longlegs), forest floor litter, spindly legs crossing fallen logs in dim shade",
    "ARACHNIDS - Funnel-web Spider, woodland understory, funnel-shaped retreat glistening in soft scattered light",
    "ARACHNIDS - Lynx Spider, tropical leaf, spiky legs poised alertly, subtle leaf vein patterns in background",
    "ARACHNIDS - Brown Recluse, dim attic corner, delicate steps on old wood, low tungsten glow accentuating shadows",
    "ARACHNIDS - Golden Orb Weaver, rainforest edge, giant golden-hued web refracting rainbow colors in humidity",

    # CRUSTACEANS (12)
    "CRUSTACEANS - American Lobster, cold Atlantic seabed, dark shell in deep bluish light, drifting marine snow",
    "CRUSTACEANS - Blue Crab, coastal estuary shallows, azure claws reflecting pastel dawn sky, gentle ripples",
    "CRUSTACEANS - Hermit Crab, tide pools, borrowed shell home, tiny legs scuttling over damp sand under warm sun",
    "CRUSTACEANS - Fiddler Crab, mudflat at low tide, one oversized claw raised like a signal, amber late-day glow",
    "CRUSTACEANS - Mantis Shrimp, tropical reef crevice, rainbow carapace and curious eyes in dappled sunbeams",
    "CRUSTACEANS - Coconut Crab, island shore, colossal claws cracking shells beneath palm shadows, saline breeze",
    "CRUSTACEANS - Japanese Spider Crab, deep ocean floor, sprawling legs under faint bioluminescent flickers",
    "CRUSTACEANS - Red King Crab, cold northern seas, vivid red spikes in gentle currents, plankton drifting",
    "CRUSTACEANS - Cleaner Shrimp, coral heads, nimble antennae at a cleaning station, fish awaiting gentle service",
    "CRUSTACEANS - Dungeness Crab, Pacific coast shallows, sandy patterns beneath slanting sunrays, drifting kelp",
    "CRUSTACEANS - Amphipod, microscopic ocean world, translucent body under macro lens light, minuscule universe",
    "CRUSTACEANS - Copepod, planktonic realm, minuscule drifting speck in laser-lit micro-cinematography",

    # AMPHIBIANS (12)
    "AMPHIBIANS - Red-eyed Tree Frog, rainforest leaf at night, brilliant eyes and lime-green skin, droplets glisten",
    "AMPHIBIANS - Poison Dart Frog (Blue), forest floor litter, cobalt hue in green gloom, humid hush and faint streams",
    "AMPHIBIANS - Axolotl, freshwater canals, pale pink body drifting lazily, feathery gills in gentle currents",
    "AMPHIBIANS - Bullfrog, North American pond, resonant calls at dusk, purple-orange sky reflecting on still water",
    "AMPHIBIANS - Fire Salamander, European woodland, black body with yellow patches, moist bark and morning mist",
    "AMPHIBIANS - Green Tree Frog, marshlands, lime form clinging to reeds, soft summer night chorus",
    "AMPHIBIANS - Spring Peeper, eastern wetlands, tiny tan frog calling in moonlight, ripples from gentle breeze",
    "AMPHIBIANS - Wood Frog, boreal pools, brownish mask near melting ice, early spring sunbeams thawing silence",
    "AMPHIBIANS - Leopard Frog, grassy pond edge, spotted green pattern, midday brightness and quiet ripple sounds",
    "AMPHIBIANS - Hellbender, Appalachian streams, large brown salamander under rocky crevice, mottled sunlight",
    "AMPHIBIANS - Blue Poison Dart Frog, tropical understory, intense cobalt body on moss, filtered green dim light",
    "AMPHIBIANS - White’s Tree Frog, Australian gardens, plump body perched on leaf, soft porchlight glow at dusk",

    # PLANTS (12)
    "PLANTS - Venus Flytrap, boggy meadow, hinged leaves poised in morning sun, tiny insect silhouettes nearby",
    "PLANTS - Orchid (Phalaenopsis), tropical understory, delicate petals and pastel hues, humid green backdrop",
    "PLANTS - Sunflower, golden farmland fields, tall heads tracking the sun, late summer warmth radiating in petals",
    "PLANTS - Cactus (Saguaro), desert twilight, towering spines etched against coral-pink sky, silent arid expanses",
    "PLANTS - Bamboo, Asian groves, vertical green stalks, shafts of filtered light and gentle rustling leaves",
    "PLANTS - Lavender, Mediterranean fields, purple blooms and buzzing bees, dry summer air and distant hills",
    "PLANTS - Fern, damp forest floor, intricate fronds unfurling in dim emerald glow, soft raindrops on leaves",
    "PLANTS - Water Lily, still pond surface, pastel petals floating, dragonflies hovering in sunlit reflection",
    "PLANTS - Bonsai, Japanese garden, miniature sculpted trunk, tranquil scene with moss and raked gravel patterns",
    "PLANTS - Baobab Tree, African savanna, thick trunk under golden sky, silhouettes of giraffes in distant haze",
    "PLANTS - Mangrove, coastal wetlands, tangled roots in shallow brackish water, tide gently shifting reflections",
    "PLANTS - Pitcher Plant, rainforest clearing, nectar-lured insects, filtered sunlight glinting off carnivorous rim",

    # CRYPTOANIMALS (12)
    "CRYPTOANIMALS - Bigfoot (Sasquatch), Pacific Northwest forest, towering silhouette amid misty pines, distant wood-knocks",
    "CRYPTOANIMALS - Loch Ness Monster (Nessie), Scottish loch at dusk, subtle ripples on still surface, hills in fading light",
    "CRYPTOANIMALS - Chupacabra, Latin American scrubland, furtive shadow slipping between cacti, eerie hush at twilight",
    "CRYPTOANIMALS - Yeti, Himalayan slopes, large humanoid shape glimpsed through swirling snow, distant ice-crack echoes",
    "CRYPTOANIMALS - Mothman, rural US night, winged figure over moonlit bridge, faint red eyes glowing in silence",
    "CRYPTOANIMALS - Jersey Devil, Pine Barrens darkness, strange winged silhouette against starlit canopy, distant rustling",
    "CRYPTOANIMALS - Thunderbird, storm-laden plains, massive bird-form riding thunderheads, lightning flashes reveal outline",
    "CRYPTOANIMALS - Ogopogo, Canadian lake, subtle wakes in morning calm, conifer-clad hills mirrored in water",
    "CRYPTOANIMALS - Mokele-Mbembe, Congo basin river, submerged hump in murky waters, dense rainforest enveloping",
    "CRYPTOANIMALS - Ahool, Indonesian rainforest canopy, giant bat-like silhouette swooping under moonlit leaves",
    "CRYPTOANIMALS - Beast of Gévaudan, 18th-century French countryside, lurking shape in wheat fields at sunset, historic dread",
    "CRYPTOANIMALS - Ningen, Southern ocean ice floes, pale humanoid marine form beneath cracked ice, eerie underwater silence",

    # DINOSAURS (12)
    "DINOSAURS - Tyrannosaurus Rex, Late Cretaceous floodplain, colossal predator against orange sunset, distant fern groves",
    "DINOSAURS - Triceratops, lush prehistoric plains, three-horned brow lowered, morning mist swirling around ferns",
    "DINOSAURS - Brachiosaurus, Jurassic woodland, towering neck browsing treetops, grand crane shot capturing scale",
    "DINOSAURS - Velociraptor, arid badlands, swift pack hunters kicking dust, warm rocky earth and sparse cycads",
    "DINOSAURS - Stegosaurus, forest clearing, plated back catching dappled sun, gentle tail swishes in humid air",
    "DINOSAURS - Allosaurus, semi-arid floodplain, fearsome stance near stream, distant volcano silhouettes, dusty horizon",
    "DINOSAURS - Ankylosaurus, Cretaceous woodland, armored shell and club tail, low-angle sun reveals rugged texture",
    "DINOSAURS - Pteranodon, coastal cliffs, winged reptile gliding over primordial shore, pastel dawn skies overhead",
    "DINOSAURS - Spinosaurus, North African river delta, sail-backed form fishing in muddy waters, calls echoing",
    "DINOSAURS - Parasaurolophus, open floodplains, crest-topped herbivore trumpeting at sunrise, herds grazing peacefully",
    "DINOSAURS - Diplodocus, Late Jurassic plains, long neck sweeping for leaves, communal giants under gentle sunlight",
    "DINOSAURS - Maiasaura, nesting colonies on rolling uplands, caring for hatchlings in dawn light, ferns rustling in breeze",

    # MYTHICAL CREATURES (12)
    "MYTHICAL CREATURES - Dragon, mountain lair, wings spread over hoarded gold, fiery breath illuminating cavern walls",
    "MYTHICAL CREATURES - Unicorn, enchanted forest glade, single horn catching moonlight, dew-kissed flowers bending softly",
    "MYTHICAL CREATURES - Phoenix, desert dawn, fiery plumage ignited by rising sun, ashes swirling in shimmering haze",
    "MYTHICAL CREATURES - Griffin, rocky highlands, eagle-lion hybrid perched majestically on a ledge, wind ruffling feathers",
    "MYTHICAL CREATURES - Mermaid, coastal lagoon, shimmering tail under rippling light, distant whale song drifting",
    "MYTHICAL CREATURES - Centaur, ancient grove, half-man half-horse stepping gracefully through dappled forest rays",
    "MYTHICAL CREATURES - Pegasus, rolling cloudscapes, white winged horse galloping through mist, sunrise pastels",
    "MYTHICAL CREATURES - Minotaur, labyrinth shadows, hulking half-bull figure in torchlit stone corridors, distant echoes",
    "MYTHICAL CREATURES - Kraken, stormy seas, colossal tentacles emerging from foaming waves, lightning reveals monstrous form",
    "MYTHICAL CREATURES - Basilisk, ruined temple interior, deadly gaze amid crumbling columns, dust motes in dim shafts",
    "MYTHICAL CREATURES - Elf, old-growth forest, elegant humanoid figure amid moss and ferns, pale sunlight through leaves",
    "MYTHICAL CREATURES - Werewolf, moonlit clearing, half-human half-wolf howling at full moon, gnarled trees in silhouette",

    # SEA ANIMALS (12)
    "SEA ANIMALS - Orca (Killer Whale), cold northern seas, black-and-white dorsal fin breaking surface, snowy peaks beyond",
    "SEA ANIMALS - Blue Whale, open ocean blue, immense gentle giant drifting beneath shimmering rays, silence and awe",
    "SEA ANIMALS - Humpback Whale, tropical shallows, breaching arcs in slow-motion, rainbow mist in blow spray",
    "SEA ANIMALS - Dolphin (Bottlenose), coastal waters, playful pod surfing sunlit waves, islands framing green horizons",
    "SEA ANIMALS - Sea Turtle (Hawksbill), coral labyrinth, ornate shell gliding gracefully, fish schooling in lively colors",
    "SEA ANIMALS - Manta Ray, pelagic blue, winged silhouette drifting elegantly, shifting sun patterns overhead",
    "SEA ANIMALS - Whale Shark, tropical plankton fields, spotted giant feeding near surface, snorkelers dwarfed in scale",
    "SEA ANIMALS - Seal (Harbor Seal), rocky coasts, sleek fur and curious gaze, gull calls echo in warm midday sun",
    "SEA ANIMALS - Seahorse, seagrass meadow, tiny curled tail grasping a blade, pastel hues drifting in mild current",
    "SEA ANIMALS - Jellyfish (Moon Jelly), calm lagoon, translucent bell pulsating softly, subtle beams dancing on sandy floor",
    "SEA ANIMALS - Coral (Staghorn), reef structures, branching arms swaying in gentle tide, schools of fish weaving through",
    "SEA ANIMALS - Pufferfish, reef crevice, inflating into a spiky orb, delicate light play on patterned scales"
]

DOMESTICATED_ANIMALS = [

    # DOGS (12) - Scenes capturing their companionship and varied settings
    "DOGS - Labrador Retriever, suburban backyard, tail wagging in afternoon sun, soft golden fur and a tennis ball",
    "DOGS - German Shepherd, spacious garden, alert stance and attentive ears, gentle breeze ruffling thick coat",
    "DOGS - Golden Retriever, lakeside dock at sunset, warm reflection on calm water, easygoing grin and wagging tail",
    "DOGS - French Bulldog, cozy apartment living room, soft lamplight, big ears casting playful shadows on the wall",
    "DOGS - Poodle (Standard), manicured lawn, elegant posture under dappled shade, curls catching gentle morning light",
    "DOGS - Shih Tzu, quiet front porch, silky hair parted by light breeze, pastel flower pots framing a sleepy afternoon",
    "DOGS - Husky, snowy backyard, piercing blue eyes and thick fur shimmering in crisp winter sun, soft crunch of snow",
    "DOGS - Beagle, country lane, nose to the ground tracking scents, soft sunlight filtering through orchard trees",
    "DOGS - Rottweiler, wide open field, confident stance with distant farm silhouettes, late-day warm glow touching glossy coat",
    "DOGS - Dachshund, city park, long body trotting happily over green grass, bright midday light and distant laughter",
    "DOGS - Great Dane, spacious family room, large silhouette beside a comfy sofa, window light highlighting gentle giant features",
    "DOGS - Poodle (Toy), window seat with indoor plants, cheerful eyes reflecting soft indoor lamps, houseplants adding fresh color",

    # CATS (12) - Comfortable indoor and garden spaces, subtle elegance
    "CATS - Siamese, sunny windowsill, slender form and blue eyes gazing outside, dust motes drifting in morning beam",
    "CATS - Persian, velvet sofa, luxurious fluffy coat, warm lamplight accentuating gentle facial features",
    "CATS - Maine Coon, wooden bookshelf edge, regal tufted ears, late afternoon sun stripes across hardwood floor",
    "CATS - Ragdoll, plush rug by fireplace, relaxed posture, soft crackle of wood creating a cozy golden hue",
    "CATS - Bengal, indoor jungle of houseplants, sleek spotted coat and inquisitive stare, late morning light filtered green",
    "CATS - British Shorthair, marble kitchen countertop, round amber eyes under warm pendant light, subtle flower vase aroma",
    "CATS - Sphynx, fleece blanket nest, furless skin warmed by reading lamp, delicate shadows and quiet purring",
    "CATS - Russian Blue, windowsill with rain outside, emerald eyes watching raindrops on glass, soft gray fur in dim light",
    "CATS - Abyssinian, sun-drenched porch, lean form and large ears, faint garden scents drifting through open door",
    "CATS - Scottish Fold, wicker chair in conservatory, folded ears and round eyes reflecting leafy patterns of sunlight",
    "CATS - American Shorthair, tiled entryway, sturdy build, calm gaze as soft midday light spills in from open door",
    "CATS - Oriental Shorthair, minimalist living space, sleek body and large ears catching every subtle household sound",

    # HORSES & EQUINES (12) - Pastoral landscapes, stables, gentle rural ambiance
    "HORSES & EQUINES - Arabian Horse, sunlit paddock, elegant arched neck, evening sun gilding chestnut coat",
    "HORSES & EQUINES - Thoroughbred, lush green pasture, strong frame silhouetted against rolling hills at sunrise",
    "HORSES & EQUINES - Quarter Horse, ranch corral, dust rising as it trots, wooden fence lines in warm midday light",
    "HORSES & EQUINES - Clydesdale, country lane, feathered hooves and massive build, soft rustling of trees in late afternoon",
    "HORSES & EQUINES - Appaloosa, open field, distinctive spotted rump, low sun angle highlighting dappled patterns",
    "HORSES & EQUINES - Shetland Pony, cottage garden, diminutive size nibbling clover, pastel blossoms and gentle breeze",
    "HORSES & EQUINES - Mustang, wide open range, mane flying in wind, distant mountains under clear blue sky",
    "HORSES & EQUINES - Lipizzaner, classical stable yard, poised stance, late-day light warming stone archways",
    "HORSES & EQUINES - Friesian Horse, meadow edge, flowing black mane and tail, golden sunset casting long shadows",
    "HORSES & EQUINES - Tennessee Walking Horse, tree-lined trail, smooth gait under green canopy, birdsong and dappled light",
    "HORSES & EQUINES - Paint Horse, fenced pasture, bold coat patterns catching sunbeams, distant barn silhouettes",
    "HORSES & EQUINES - Haflinger, alpine foothills, palomino coat gleaming, crisp mountain air and scattered wildflowers",

    # LIVESTOCK (12) - Farms, barns, rolling fields, rural warmth
    "LIVESTOCK - Cow (Holstein), grassy hillside, black-and-white patches under gentle morning sun, distant farmhouse",
    "LIVESTOCK - Sheep (Merino), meadow of wildflowers, soft wool backlit by a low sun, gentle bleating in tranquil stillness",
    "LIVESTOCK - Goat (Nubian), stone-walled enclosure, long ears and curious stare, mild afternoon warmth on straw bedding",
    "LIVESTOCK - Pig (Berkshire), muddy pen, pink snout sniffing fresh hay, shafts of sunlight through barn slats",
    "LIVESTOCK - Chicken (Hen), cozy coop yard, pecking at scattered feed, bright midday light and gentle clucking",
    "LIVESTOCK - Donkey, rustic stable yard, soft bray as warm straw scents rise, subtle shadows on old wooden beams",
    "LIVESTOCK - Llama, Andean-inspired paddock, elegant neck framed by rolling hills and a faint mountain line at dawn",
    "LIVESTOCK - Alpaca, small pasture, fluffy face under pastel sky, quiet murmurs and a sense of calm community",
    "LIVESTOCK - Turkey, farm orchard, fanned tail in afternoon glow, orchard blossoms drifting in gentle breeze",
    "LIVESTOCK - Duck (Domestic), farm pond, glossy feathers reflected in rippling water, subdued quacking amid reeds",
    "LIVESTOCK - Goose (Domestic), grassy orchard clearing, white plumage radiant in golden hour, orchard fruit scent",
    "LIVESTOCK - Mule, barn entrance, sturdy and patient, warm lamplight from stable interior, soft hoof sounds on straw",

    # BIRDS (12) - Domesticated birds, cages, aviaries, homesteads
    "BIRDS - Parrot (African Grey), indoor perch near window, inquisitive eyes, filtered morning light and subtle chatter",
    "BIRDS - Cockatiel, living room corner, gentle whistle echoing in warm lamplight, pastel crest feathers catching glow",
    "BIRDS - Budgerigar (Parakeet), small cage by sunny kitchen window, cheerful chirps mixing with coffee aroma",
    "BIRDS - Canary, Victorian-style aviary, vivid yellow plumage singing in soft midday beams, potted ferns surrounding",
    "BIRDS - Pigeon (Homing Pigeon), rooftop coop, cooing under a pale sky, city sounds muted by gentle sunrise",
    "BIRDS - Chicken (Hen), backyard coop, pecking at feed under warm afternoon sun, family voices in background",
    "BIRDS - Duck (Mallard Domestic), small garden pond, iridescent head and gentle quacks, reflections in still water",
    "BIRDS - Goose (Embden Goose), orchard path, white feathers glowing as it waddles by fallen apples in late day light",
    "BIRDS - Peacock (Domestic), manor garden, radiant tail fanned under clear blue sky, gravel path glistening",
    "BIRDS - Macaw (Blue-and-Gold), spacious indoor aviary, vivid plumage under skylight, subtle hum of household",
    "BIRDS - Finch (Zebra Finch), indoor cage with greenery, soft chattering, morning sun striping the tiled floor",
    "BIRDS - Quail (Domestic), small hutch in quiet backyard, gentle whirring calls, dusk settling softly over neat hedges",

    # SMALL MAMMALS (12) - Pet habitats, cozy home corners
    "SMALL MAMMALS - Rabbit (Netherland Dwarf), indoor pen by window, fluffy silhouette nibbling lettuce in gentle sun",
    "SMALL MAMMALS - Hamster (Syrian), terrarium with wood shavings, tiny paws holding sunflower seed, lamp glow overhead",
    "SMALL MAMMALS - Guinea Pig (Abyssinian), simple backyard hutch, soft chirrups, dappled shade from garden leaves",
    "SMALL MAMMALS - Ferret, cozy living room rug, curious slink and playful bounce, subdued lamplight in the evening",
    "SMALL MAMMALS - Chinchilla, quiet corner with climbing branches, plush grey fur in warm indoor lighting, muffled TV sounds",
    "SMALL MAMMALS - Rat (Fancy Rat), desk terrarium, whiskers twitching as it explores, late-night reading lamp",
    "SMALL MAMMALS - Mouse (White Mouse), small cage on a shelf, tiny pink feet scurrying, sunbeam highlighting delicate ears",
    "SMALL MAMMALS - Gerbil, sandy enclosure, constructing burrows and tunnels, subtle earthy scents in afternoon calm",
    "SMALL MAMMALS - Hedgehog (African Pygmy), wicker basket bed, snuffling quietly, warm house setting with soft music",
    "SMALL MAMMALS - Sugar Glider, indoor tree branch setup, nocturnal eyes reflecting moonlight from window",
    "SMALL MAMMALS - Degus, roomy pen with wooden platforms, gentle chattering, late afternoon light warming straw floor",
    "SMALL MAMMALS - Prairie Dog, suburban backyard enclosure, alert posture and chirpy calls, kids laughing nearby",

    # FISH (12) - Aquarium environments, indoor calm
    "FISH - Goldfish (Fancy), indoor aquarium near kitchen, orange fins flutter under LED tank light, soothing bubble hum",
    "FISH - Betta, small desktop tank, vibrant reds and blues flaring in gentle overhead illumination, a quiet office corner",
    "FISH - Guppy, planted nano-tank, schooling in tiny colorful bursts, soft green glow from aquatic plants",
    "FISH - Neon Tetra, lush community aquarium, electric blue-red streaks darting among waving aquatic ferns",
    "FISH - Angelfish (Freshwater), medium home aquarium, graceful fins gliding in lamplight, subtle reflections on glass",
    "FISH - Discus, warm-water tank, disc-shaped body with swirling patterns, ambient hush of filtration system",
    "FISH - Gourami (Dwarf), softly lit tank in living room, pastel body colors shifting with slight angle of view",
    "FISH - Molly (Black Molly), small community tank, dark form contrasting green plants, gentle morning window light",
    "FISH - Swordtail, aquarium with driftwood and pebbles, bright red tail catching LED shimmer, tranquil water current",
    "FISH - Corydoras Catfish, bottom of tank, whiskered face rummaging gently, subtle beams crossing gravel bed",
    "FISH - Cardinal Tetra, schooling under floating plants, vivid reds and blues in dim, naturalistic lighting",
    "FISH - Rasbora (Harlequin), tight schooling patterns, reflective silver-orange bodies in mild afternoon glow",

    # REPTILES (12) - Terrariums, controlled environments, gentle household ambiance
    "REPTILES - Leopard Gecko, terrarium with warm basking spot, spotted tail resting on sand, gentle red heat lamp",
    "REPTILES - Bearded Dragon, desert-themed enclosure, basking under UV light, calm gaze amid rock and branch decor",
    "REPTILES - Corn Snake, glass terrarium with leafy hide, orange scales patterning softly under overhead LED",
    "REPTILES - Ball Python, dim enclosure with warm hide, sleek coils, subtle reflections on smooth skin",
    "REPTILES - Red-eared Slider, turtle tank with platform, paddling in clear water, sunlight through nearby window",
    "REPTILES - Russian Tortoise, indoor pen with soft substrate, munching on greens, muted household sounds",
    "REPTILES - Greek Tortoise, sunny windowsill enclosure, shell patterns catching beams of natural daylight",
    "REPTILES - Chameleon (Veiled), arboreal terrarium, shifting green hues on branches, slow graceful movements",
    "REPTILES - Blue-tongued Skink, low terrarium, earthy décor, broad head turned towards mild warm lamp",
    "REPTILES - Painted Turtle, shallow water tank, mottled shell pattern in rippling light, mild hum of water filter",
    "REPTILES - Hermann's Tortoise, small indoor garden box, nibbling leaves under morning rays, gentle warmth",
    "REPTILES - African Sideneck Turtle, aquatic setup, quirky sideways head glimpses, gentle bubble stream rising",

    # INSECTS (12) - Insect enclosures or outdoor garden corners set up for them
    "INSECTS - Bee (Honeybee), small backyard hive box, buzzing around lavender blooms, sunlight on golden thorax",
    "INSECTS - Butterfly (Monarch), butterfly garden enclosure, orange wings flutter against purple flowers",
    "INSECTS - Ladybug, indoor herb planter on windowsill, red shell with black dots contrasting lush green leaves",
    "INSECTS - Praying Mantis, mesh terrarium with twigs, poised elegantly, overhead lamp creating delicate shadows",
    "INSECTS - Ant (Ant Farm), glass ant farm on study desk, tiny tunnels visible, soft lamp revealing busy movement",
    "INSECTS - Beetle (Hercules Beetle), hobbyist terrarium, strong horn silhouette in dim room light, subtle scratch sounds",
    "INSECTS - Stick Insect, simple foliage enclosure, twig mimic swaying gently as furnace hums quietly",
    "INSECTS - Cockroach (Madagascar Hissing), well-ventilated terrarium, faint hissing in subdued red heat lamp glow",
    "INSECTS - Cricket (House Cricket), small cricket keeper box, chirping rhythmically under a modest LED night light",
    "INSECTS - Silk Moth (Bombyx mori), silent bedroom corner, creamy wings resting on screen mesh, faint lamp behind",
    "INSECTS - Katydid, mini greenhouse setup, green leaf mimic blending seamlessly, filtered sunlight and faint indoor echoes",
    "INSECTS - Ant (Leafcutter), specialized enclosure with leaves and fungus gardens, tiny shapes carrying leaf fragments in soft lamplight",

    # PLANTS (12) - Indoor houseplants and simple gardens reflecting domestic serenity
    "PLANTS - Rose (Garden variety), small front yard garden bed, warm afternoon sun highlighting velvety petals",
    "PLANTS - Orchid (Phalaenopsis), windowsill pot, delicate white-pink blooms, soft natural light and gentle indoor hush",
    "PLANTS - Cactus (Golden Barrel), sunlit windowsill, spines glowing halo-like, subtle background hum of air conditioning",
    "PLANTS - Lavender, patio planter, purple blossoms swaying gently, bees visiting in late afternoon calm",
    "PLANTS - Bonsai (Juniper), living room shelf, miniature tree illuminated by a reading lamp, tranquil evening stillness",
    "PLANTS - Peace Lily, corner of office, broad green leaves and elegant white spathe in diffused skylight",
    "PLANTS - Monstera (Swiss Cheese Plant), apartment lounge, glossy leaves with characteristic holes, mild morning rays",
    "PLANTS - Snake Plant (Sansevieria), bedroom corner, vertical leaves catching soft bedside lamp glow, quiet serenity",
    "PLANTS - Jade Plant (Crassula), kitchen windowsill, plump green leaves reflecting mid-morning sun, peaceful domestic hum",
    "PLANTS - Fiddle-leaf Fig (Ficus), near balcony door, large leaves filtering bright afternoon beams, gentle urban skyline outside",
    "PLANTS - Spider Plant, hanging basket in living area, arching leaves and tiny plantlets, steady ambient indoor light",
    "PLANTS - Tomato (Garden bed), backyard plot, red fruit clusters ripening under full sun, distant laughter and rustling leaves"
]


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
    
    def build_gui(self):
        # Initialize ttk Style and apply theme
        style = ttk.Style()
        apply_theme(style, self.root)

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
        menubar = Menu(self.root, bg=COLOR_PALETTE["dark_navy"], fg=COLOR_PALETTE["white"])
        settings_menu = Menu(menubar, tearoff=0, bg=COLOR_PALETTE["dark_navy"], fg=COLOR_PALETTE["white"],
                             activebackground=COLOR_PALETTE["gold"], activeforeground=COLOR_PALETTE["dark_blue_black"])
        settings_menu.add_command(label="Set Output Directory", command=self.set_output_directory)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        self.root.config(menu=menubar)

        # Instructions
        self.instructions = tk.Label(
            self.root,
            text="Enter the desired prompt concept (max 5,000 characters):",
            bg=COLOR_PALETTE["dark_blue_black"],
            fg=COLOR_PALETTE["white"],
            font=('Helvetica', 14, 'bold')
        )
        self.instructions.grid(row=1, column=0, pady=10, sticky='ew', padx=20)

        # Input Frame
        self.input_frame = tk.Frame(self.root, bg=COLOR_PALETTE["dark_blue_black"])
        self.input_frame.grid(row=2, column=0, pady=5, padx=20, sticky='nsew')
        self.input_frame.columnconfigure(0, weight=1)

        self.input_text = ScrolledText(
            self.input_frame,
            font=('Helvetica', 12),
            wrap=tk.WORD,
            bg=COLOR_PALETTE["dark_navy"],
            fg=COLOR_PALETTE["white"],
            insertbackground=COLOR_PALETTE["white"],  # Cursor color
            bd=2,
            relief=tk.GROOVE
        )
        self.input_text.grid(row=0, column=0, sticky='nsew')
        self.input_frame.rowconfigure(0, weight=1)

        self.char_count_label = tk.Label(
            self.input_frame,
            text="0/5000 characters",
            bg=COLOR_PALETTE["dark_blue_black"],
            fg=COLOR_PALETTE["light_gold"],
            font=('Helvetica', 10, 'italic')
        )
        self.char_count_label.grid(row=1, column=0, sticky="e", pady=(2,0))

        self.input_text.bind("<KeyRelease>", self.check_input_text)

        # Buttons Frame
        self.buttons_frame = tk.Frame(self.root, bg=COLOR_PALETTE["dark_blue_black"])
        self.buttons_frame.grid(row=3, column=0, pady=20, padx=20, sticky='ew')
        self.buttons_frame.columnconfigure((0,1,2,3), weight=1)

        # Video Prompt Options Button
        self.video_prompt_options_button = ttk.Button(
            self.buttons_frame,
            text="Video Options - REQUIRED",
            command=self.show_video_prompt_options
        )
        self.video_prompt_options_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Generate Video Prompts Button
        self.generate_video_prompts_button = ttk.Button(
            self.buttons_frame,
            text="Generate Video Prompts",
            command=self.generate_video_prompts,
            state=tk.DISABLED
        )
        self.generate_video_prompts_button.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        # Output Text Area
        self.output_frame = tk.Frame(self.root, bg=COLOR_PALETTE["dark_blue_black"])
        self.output_frame.grid(row=4, column=0, pady=10, padx=20, sticky='nsew')
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)

        self.output_text = ScrolledText(
            self.output_frame,
            font=('Helvetica', 12),
            wrap=tk.WORD,
            bg=COLOR_PALETTE["dark_navy"],
            fg=COLOR_PALETTE["white"],
            insertbackground=COLOR_PALETTE["white"],  # Cursor color
            bd=2,
            relief=tk.GROOVE
        )
        self.output_text.grid(row=0, column=0, sticky='nsew', padx=(0,10))

        # Copy to Clipboard Link
        self.copy_link = tk.Label(
            self.output_frame,
            text="Copy to Clipboard",
            fg=COLOR_PALETTE["light_gold"],
            bg=COLOR_PALETTE["dark_blue_black"],
            cursor="hand2",
            font=('Helvetica', 12, 'underline')
        )
        self.copy_link.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.copy_link.bind("<Button-1>", lambda e: self.copy_to_clipboard(self.output_text.get("1.0", tk.END).strip()))

        # Buttons Row for Generate and Combine
        button_row_frame = tk.Frame(self.root, bg=COLOR_PALETTE["dark_blue_black"])
        button_row_frame.grid(row=5, column=0, pady=5, padx=5, sticky='ew')
        button_row_frame.columnconfigure((0,1,2), weight=1)

        # Generate Videos Button
        self.generate_videos_button = ttk.Button(
            button_row_frame,
            text="Generate Videos",
            command=self.generate_videos
        )
        self.generate_videos_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Footer Frame containing Logo
        self.footer_frame = tk.Frame(self.root, bg=COLOR_PALETTE["dark_blue_black"])
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
                bg=COLOR_PALETTE["dark_blue_black"],
                cursor="hand2"
            )
            logo_label.image = self.logo_photo
            logo_label.grid(row=0, column=1, sticky='e', padx=20, pady=5)
            logo_label.bind("<Button-1>", self.open_website)
        except Exception as e:
            print(f"Failed to download or display logo: {e}")
            messagebox.showerror("Logo Error", f"Failed to load logo: {e}")

        # Contact Information Label
        contact_info = "Sol@TemporalLab.com - Text 385-222-9920"
        contact_label = tk.Label(
            self.root,
            text=contact_info,
            bg=COLOR_PALETTE["dark_blue_black"],
            fg=COLOR_PALETTE["white"],
            font=('Helvetica', 10)
        )
        contact_label.grid(row=7, column=0, sticky='s', pady=5)

    def set_output_directory(self):
        """Implement the functionality to set the output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            DEFAULTS["save_path"] = directory
            logging.info(f"Output directory set to: {directory}")
            messagebox.showinfo("Output Directory Set", f"Output directory set to:\n{directory}")

    def show_video_prompt_options(self):
        """Implement the functionality to show video prompt options."""
        # Placeholder for actual implementation
        messagebox.showinfo("Video Options", "Video prompt options dialog will be here.")

    def generate_video_prompts(self):
        """Implement the functionality to generate video prompts."""
        # Placeholder for actual implementation
        messagebox.showinfo("Generate Video Prompts", "Video prompts have been generated.")

    def copy_to_clipboard(self, text):
        """Copy the provided text to the clipboard."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()  # Keeps the clipboard content after the window is closed
        messagebox.showinfo("Copied", "Output text copied to clipboard.")

    def open_website(self, event=None):
        """Open the Temporal Labs website in the default browser."""
        webbrowser.open("https://www.temporal-labs.com")  # Replace with your actual website

    def generate_videos(self):
        """Implement the functionality to generate videos."""
        # Placeholder for actual implementation
        messagebox.showinfo("Generate Videos", "Video generation has been initiated.")

    def check_input_text(self, event=None):
        """Check the input text length and update character count."""
        text = self.input_text.get("1.0", tk.END).strip()
        char_count = len(text)
        self.char_count_label.config(text=f"{char_count}/5000 characters")

        # Enable or disable buttons based on text length
        if 1 <= char_count <= 5000:
            self.generate_video_prompts_button.config(state=tk.NORMAL)
        else:
            self.generate_video_prompts_button.config(state=tk.DISABLED)

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

            # Activate Environment and Run the Utility
            self.activate_and_run_utility(cogvideo_path, temporal_script)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


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
        Integrates a foundational decade to set the cinematic aesthetics while allowing narrative traversal across multiple decades.
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

        # Retrieve the foundational decade from the dropdown
        foundational_decade = self.video_decade_var.get()


        if self.video_story_mode_var.get():
            # Story Mode: Generate a story outline first
            outline_generated = False
            max_outline_retries = 42  # Reduced for practicality
            outline_retry_count = 0

            while not outline_generated and outline_retry_count < max_outline_retries:
                try:
                    # Step 1: Generate a story outline with system prompt
                    outline_prompt = (
                            f"Create a well-thought out, organized and professionally crafted sequence of temporally coherent, engaging story beats from the following concept '{input_concept}'. Each story beat, or prompt set, will result in a video that takes place over a 5 second time-span, per prompt set. You are specifically dividing it into exactly {num_prompts} prompt seeds without repeating any prompt exactly within the same list. Each prompt seed should represent a distinct visual beat within the story sequence that advances the narrative without repeating previous ideas or deviating off-track from the original material. All things need to remain coherent and consistent throughout the story. '{input_concept}' may be about animals, objects, scenes, aliens or anything else that isn't human. IF it is then include specific descriptors to guide the subject towards the intended species and away from human charateristics where desired. Always make sure those describing factors are always present in other subsequent prompts to retain coherency across them. DO NOT ever list the PROS or CONS for a camera so token space is always maximized towards the actual scene itself. DO NOT GIVE ANY TECHNICAL SPECS, ADVICE OR EXPOSITION ABOUT THE CAMERA. \n"
                            f"Provide the outline as a numbered list, with each scene on a new line, starting from 1 up to {num_prompts}. Do not include any additional text before or after the outline. Do not create a scene that might promote or glorify illegal activities. Do not promote or glorify illegal activities EVER.\n"
                            f"With an acute awareness and keen judgement please leverage an awareness of prompt engineering to influence with specific terminology and stylistic elements relevant to the time period, environment, setting and other factors contributing to the visual scene for this concept '{input_concept}'.\n"
                            f"Never start with scene titling exposition like 'Scene 4...'. You are STRICTLY focus on visual elements and never describing sound, taste, feeling, vibe, context or provide exposition outside of visual descriptors. Do not reiterate the {input_concept} directly. Always describe the specific details for whatever is the subject. Never just say a vague term like 'baby alien', describe it in detail so the AI knows exactly what to generate. Never presume it knows best. It must explicitely be given visual direction before it can generate reliably. Do not provide additional exposition back towards the user.\n"
                            f"Ensure that each scene builds upon the previous one and that subjects remain consistent across scenes, you must include cohesive details for the subject across the scenes, maintaining a logical and engaging progression. Incorporate rising action, climax, and resolution appropriately to create a satisfying and always commercially viable story.\n"
                            f"Sometimes the input concept will be about animals, objects, scenes, aliens or something instead of humans. Use descriptive and evocative language to bring characters, aliens or whatever the subject is and it's setting to life.\n"
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
                        print(f"Temporal Story Outline still generating. Please be patient while I continue putting everything together for you... ({outline_retry_count})")
                except Exception as e:
                    outline_retry_count += 1
                    print(f"It looks like there has been an error generating Temporal Story Outline: {e}. This is not common. Let me go ahead and retry that for you... ({outline_retry_count}/{max_outline_retries})")

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

                        # Prepare a summary of previous scenes for continuity
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
                            f"Use '{previous_scenes_summary}' to inform you of what has happened upto this point in the sequence and then focusing on {prompt_index}:{scene_description}, create a full-featured video prompt. This prompt must always be detailed, commercially viable, inspiring, original and professionally crafted, spatially coherent and engaging. DO NOT DESCRIBE OR ACKNOWLEDGE ANY SOUND, TASTE, TOUCH OR SMELL IN THE SCENE AND ONLY PROVIDE A FOCUSED AND PROFESSIONAL QUALITY VISUAL PROMPTS. DO NOT ever list the PROS or CONS for a camera so token space is always maximized towards the actual scene itself. DO NOT GIVE ANY TECHNICAL SPECS, ADVICE OR EXPOSITION ABOUT THE CAMERA. IF it is about animals, objects, scenes, aliens or anything else that isn't human. then include specific descriptors to guide the subject towards the intended species and away from human charateristics where desired and always make sure those describing factors are always present in other subsequent prompts to retain coherency across them. DO NOT EVER USE VAGUE LANGUAGE LIKE 'of the era.' or 'Detailed visual description of the toddler-sized phoenix's face'. Always give the actual and thought-out details and never focus the prompt on the camera. ALWAYS focus on the subject and the scene itself, speak towards known tropes of the {foundational_decade} decade. Ensure that subjects remain consistent across scenes. Never provide any form of extra exposition at the start, end or middle in any form like 'Here are the Positive and Negative Prompt Sets for Scene 6:\n\n**Positive Prompt Set:' and ONLY ever start with 'Positive:' or 'Negative:' before providing the appropriate and respective prompt content.\n"
                            f"Focus on the scene's setting and subject and actually describe it in extreme visual detail. Never just list as 'subject' or 'character' and rather give specifics every single time. Any script notes or non-diagetic exposition is not required. Always follow a character's name with their outfit and obvious behavior if they're human. If they're not-human or the scene doesn't otherwise call for clothing then describe they're physical features and how it interacts with the environment. Always include these cohesive details for the subject across the scenes, maintaining a logical and engaging progression with setting and focus that aligns with the chosen story arc. Always follow the name of a location with a description of it. IT SHOULD ONLY BE ABOUT THE VISUAL SCENE ITSELF that you focus on. \n"
                            f"Always approach with an awareness of real-world physics. Lighting and shadows should be described to reflect natural light sources, considering their angle, intensity, and spectral properties, while shadows must exhibit accurate occlusion and scattering effects. Material properties should ne describe realistically to interact with that light, with metals showcasing reflectivity and color-dependent sheen, and surfaces like water demonstrating specular reflections and refraction. Environmental dynamics such as wind and fluid interactions must be modeled to influence elements that most make sense to the scene. Similarly, gravity and forces should govern object interactions, ensuring that items are naturally responding to air resistance. \n"
                            f"All content must be within PG-13 guidelines and always family-friendly. Each prompt should be a five-sentence description maximizing the token space for conveying the most information to the model as efficiently as possible. Always phrase the establishing sentence with details about the camera and decade as such 'Set in {foundational_decade}, shot on a {video_options['camera']}... '. ALWAYS phrase without including PROS or CONS like in this one example 'Shot on a Pathé-Baby built in 1922'. DO NOT ever list the PROS or CONS for a camera so token space is always maximized towards the actual scene itself. DO NOT GIVE ANY TECHNICAL SPECS, ADVICE OR EXPOSITION ABOUT THE CAMERA. It may specificy the time period of which you are supposed to represent as some concepts may not be restricted to only one decade. the decades mentioned across '{input_concept}' DO supercede the decade of the camera decade. The camera parameter itself '{video_options['camera']} only sets the visual tone and quality of the video itself, this way it enables time travel and stuff if requested but you are to work together as a team and use logic to determine what the user is actually requesting using common sense.\n"
                            f"Generate exactly one Positive Prompt and one Negative Prompt as a Prompt Set for Scene {prompt_index}. NEVER start with anything other than 'Positive:' OR 'Negative:'. NEVER give any exposition about what number the scene is. JUST start with 'Positive:'\n"
                            f"Positive: Set in {foundational_decade}, shot on a {video_options['camera']}... [Detailed visual description follows]\n"
                            f"Negative: Blurry background figures, misaligned or awkward features, deformed limbs, distracting backgrounds, cluttered scenes\n"
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
                            print(f"Scene {prompt_index} generated successfully.")
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

    # Non-Story Mode
        else:
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
                            f"Focusing on {input_concept}, create a full-featured video prompt. This prompt must always be commercially viable, inspiring, original and professionally crafted, spatially coherent, engaging, time-aware, physics aware. DO NOT DESCRIBE OR ACKNOWLEDGE ANY SOUND, TASTE, TOUCH OR SMELL IN THE SCENE AND ONLY PROVIDE A FOCUSED AND PROFESSIONAL QUALITY VISUAL PROMPTS. DO NOT ever list the PROS or CONS for a camera so token space is always maximized towards the actual scene itself. DO NOT GIVE ANY TECHNICAL SPECS, ADVICE OR EXPOSITION ABOUT THE CAMERA. {input_concept} may specify the time period of which you are required to place the prompts in, you are placing them in this by specifying which decade at the start of each prompt would be IF 'input_prompt' mentiones '1920's then you would say 'set in the 1920s' and IF 'input_prompt' says 1400 then you would start with 'set in the year 1400' THESE ARE EXAMPLES AND THE YEARS MENTIONED IN PROMPTS WILL VARY.The story sequence may not be restricted to only one decade so if '{input_concept}' specifies decades, years, months or any other form of time period then DO NOT just say 'Set in 2020s' because the camera decade is 2020. The camera parameter itself '{video_options['camera']} itself is from, this way it enables time travel and stuff if requested but you are to work together as a team and use logic to determine what the user is actually requesting using common sense. NEVER provide abstract or emotional descriptions, phrases or concepts. Every prompt must leverage a keen implementation to further show it is set in the the decade(s) mentioned in {input_concept}. IF it is about animals, objects, scenes, aliens or anything else that isn't human. then include specific descriptors to guide the subject towards the intended species and away from human charateristics where desired and always make sure those describing factors are always present in other subsequent prompts to retain coherency across them. DO NOT EVER USE VAGUE LANGUAGE LIKE 'of the era.' or 'Detailed visual description of the toddler-sized phoenix's face'. Always give the actual and thought-out details and never focus the prompt on the camera. ALWAYS focus on the subject and the scene itself, speak towards known tropes of the {foundational_decade} decade. Ensure that subjects remain consistent across scenes. Never provide any form of extra exposition at the start, end or middle in any form like 'Here are the Positive and Negative Prompt Sets for Scene 6:\n\n**Positive Prompt Set:' and ONLY ever start with 'Positive:' or 'Negative:' before providing the appropriate and respective prompt content.\n"
                            f"Focus on the scene's setting and subject and actually describe it in extreme visual detail. Never just list as 'subject' or 'character' and rather give specifics every single time. Any script notes or non-diagetic exposition is not required. Always follow a character's name with their outfit and obvious behavior if they're human. If they're not-human or the scene doesn't otherwise call for clothing then describe they're physical features and how it interacts with the environment. Always include these cohesive details for the subject across the scenes, maintaining a logical and engaging progression with setting and focus that aligns with the chosen story arc. Always follow the name of a location with a description of it. IT SHOULD ONLY BE ABOUT THE VISUAL SCENE ITSELF that you focus on. \n"
                            f"Always approach with an awareness of real-world physics. Lighting and shadows should be described to reflect natural light sources, considering their angle, intensity, and spectral properties, while shadows must exhibit accurate occlusion and scattering effects. Material properties should ne describe realistically to interact with that light, with metals showcasing reflectivity and color-dependent sheen, and surfaces like water demonstrating specular reflections and refraction. Environmental dynamics such as wind and fluid interactions must be modeled to influence elements that most make sense to the scene. Similarly, gravity and forces should govern object interactions, ensuring that items are naturally responding to air resistance. \n"
                            f"All content must be within PG-13 guidelines and always family-friendly. Each prompt should be a five-sentence description maximizing the token space for conveying the most information to the model as efficiently as possible. Always phrase the establishing sentence with details about the camera and decade as such 'Set in {foundational_decade}, shot on a {video_options['camera']}... '. ALWAYS phrase without including PROS or CONS like in this one example 'Shot on a Pathé-Baby built in 1922'. DO NOT ever list the PROS or CONS for a camera so token space is always maximized towards the actual scene itself. DO NOT GIVE ANY TECHNICAL SPECS, ADVICE OR EXPOSITION ABOUT THE CAMERA. It may specificy the time period of which you are supposed to represent as some concepts may not be restricted to only one decade. the decades mentioned across '{input_concept}' DO supercede the decade of the camera decade. The camera parameter itself '{video_options['camera']} only sets the visual tone and quality of the video itself, this way it enables time travel and stuff if requested but you are to work together as a team and use logic to determine what the user is actually requesting using common sense.\n"
                            f"Generate exactly one Positive Prompt and one Negative Prompt as a Prompt Set for Scene {prompt_index}. NEVER start with anything other than 'Positive:' OR 'Negative:'. NEVER give any exposition about what number the scene is. JUST start with 'Positive:'\n"
                            f"Positive: Set in {foundational_decade}, shot on a {video_options['camera']}... [Detailed visual description follows]\n"
                            f"Negative: Blurry background figures, misaligned or awkward features, deformed limbs, distracting backgrounds, cluttered scenes\n"
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

        positive: The positive aspects of the scene or shot in masterful {prompt_type} detail including specific features.
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
            print(f"Expected {num_prompts} scenes in the story, but got {len(scene_descriptions)}.")
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
