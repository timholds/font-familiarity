#!/usr/bin/env python
"""
Script to generate font images using TextRecognitionDataGenerator (TRDG).
"""

import os
import argparse
import random
from pathlib import Path
from typing import List
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from trdg.generators import GeneratorFromStrings
import importlib
from PIL import ImageFont
from functools import wraps

# Add compatibility for newer PIL versions that don't have getsize
def add_getsize_compatibility():
    """Add getsize method to FreeTypeFont class for compatibility with newer PIL versions."""
    if not hasattr(ImageFont.FreeTypeFont, 'getsize') and hasattr(ImageFont.FreeTypeFont, 'getbbox'):
        # Define a getsize method that uses getbbox
        def getsize(self, text):
            bbox = self.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
            
        # Add the method to the FreeTypeFont class
        ImageFont.FreeTypeFont.getsize = getsize
        print("Added getsize compatibility to PIL FreeTypeFont")

# Run the compatibility function
add_getsize_compatibility()

# Import FontMatcher from the same module used in create_font_pil_images.py
try:
    from match_fonts import FontMatcher
    has_font_matcher = True
except ImportError:
    has_font_matcher = False
    print("Warning: FontMatcher not available. Falling back to basic font matching.")

# Import TRDG generators

def check_fonts_availability(font_names: List[str], font_dir: str) -> List[str]:
    """Check which fonts are available and return the available ones."""
    available_fonts = []
    missing_fonts = []
    
    print(f"Checking availability of {len(font_names)} fonts in {font_dir}...")
    
    # List all font files in the directory
    all_font_files = [f for f in os.listdir(font_dir) 
                       if f.endswith(('.ttf', '.otf', '.TTF', '.OTF'))]
    print(f"Found {len(all_font_files)} font files in directory")
    
    # Sample output of some font files
    if all_font_files:
        print("Sample of available font files:")
        for file in sorted(all_font_files)[:5]:  # Show first 5 as sample
            print(f"  - {file}")
        if len(all_font_files) > 5:
            print(f"  - ... and {len(all_font_files) - 5} more")
    
    # Check each font
    for font_name in tqdm(font_names, desc="Checking fonts"):
        try:
            get_font_path(font_name, font_dir)
            available_fonts.append(font_name)
        except FileNotFoundError:
            missing_fonts.append(font_name)
    
    if missing_fonts:
        print(f"Warning: Could not find {len(missing_fonts)} fonts:")
        for font in missing_fonts[:10]:  # Show first 10 missing fonts
            print(f"  - {font}")
        if len(missing_fonts) > 10:
            print(f"  - ... and {len(missing_fonts) - 10} more")
    
    return available_fonts

def read_font_list(font_list_path: str) -> List[str]:
    """Read the list of fonts from a text file."""
    with open(font_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def read_text_file(text_file_path: str) -> str:
    """Read sample text from a text file."""
    with open(text_file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_font_path(font_name: str, font_dir: str, verbose: bool = False, font_matcher=None) -> str:
    """Get the path to a specific font file with more robust matching."""
    # Check using FontMatcher if available
    if font_matcher:
        try:
            # Use FontMatcher to get the font path
            font_path = font_matcher.get_font_path(font_name)
            if font_path and os.path.exists(font_path):
                if verbose:
                    print(f"FontMatcher found: {font_path}")
                return font_path
        except Exception as e:
            if verbose:
                print(f"FontMatcher failed for '{font_name}': {e}")
    
    if verbose:
        print(f"Attempting to find font: '{font_name}' in directory: {font_dir}")
    

    for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
        path = os.path.join(font_dir, font_name + ext)
        if os.path.exists(path):
            if verbose:
                print(f"Found exact match: {path}")
            return path
    
    # Generate variations of the font name
    variations = [
        font_name.replace(' ', ''),  # No spaces
        font_name.replace(' ', '-'),  # Hyphens
        font_name.replace(' ', '_'),  # Underscores
        font_name.lower(),  # Lowercase
        font_name.lower().replace(' ', ''),  # Lowercase, no spaces
        font_name.lower().replace(' ', '-'),  # Lowercase, hyphens
        font_name.lower().replace(' ', '_'),  # Lowercase, underscores
    ]
    
    # Try all variations with all extensions
    for variant in variations:
        for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
            path = os.path.join(font_dir, variant + ext)
            if os.path.exists(path):
                if verbose:
                    print(f"Found variant match: {path}")
                return path
    
    # Search for any file that contains the font name
    all_font_files = []
    for file in os.listdir(font_dir):
        if file.endswith(('.ttf', '.otf', '.TTF', '.OTF')):
            all_font_files.append(file)
            file_base = os.path.splitext(file)[0].lower()
            font_name_lower = font_name.lower()
            
            # Check if file contains font name or vice versa
            if (font_name_lower in file_base or 
                file_base in font_name_lower or
                font_name_lower.replace(' ', '') in file_base or
                font_name_lower.replace(' ', '-') in file_base or
                font_name_lower.replace(' ', '_') in file_base):
                path = os.path.join(font_dir, file)
                if verbose:
                    print(f"Found partial match: {path}")
                return path
    
    # Add a list of available fonts if verbose
    if verbose:
        print(f"Available font files in {font_dir}:")
        for file in sorted(all_font_files):
            print(f"  - {file}")
    
    raise FileNotFoundError(f"Font file for '{font_name}' not found in {font_dir}")

def get_text_samples(text_content: str, num_samples: int, sample_length: int = 100) -> List[str]:
    """Generate random text samples from the provided text content."""
    samples = []
    text_length = len(text_content)
    
    for _ in range(num_samples * 2):  # Generate more samples than needed for variety
        if text_length <= sample_length:
            samples.append(text_content)
        else:
            # Get a random starting position
            start_pos = random.randint(0, text_length - sample_length - 1)
            
            # Find the nearest space to start at a word boundary
            while start_pos > 0 and text_content[start_pos] != ' ':
                start_pos -= 1
            
            # Get the text sample
            end_pos = start_pos + sample_length
            while end_pos < text_length and text_content[end_pos] != ' ':
                end_pos += 1
            
            samples.append(text_content[start_pos:end_pos].strip())
    
    return samples

# Updated function:
def process_font(args):
    """Process a single font and generate images."""
    font_name, font_dir, text_samples, output_dir, samples_per_font, start_idx, font_matcher = args
    
    # Create output directory for this font
    font_output_dir = os.path.join(output_dir, font_name.lower().replace(' ', '_'))
    os.makedirs(font_output_dir, exist_ok=True)
    
    # Generate images for this font
    images_created = 0
    
    # Try to determine the font path
    actual_font_path = None
    if font_matcher:
        try:
            # Check if the font can be loaded with FontMatcher
            font = font_matcher.load_font(font_name, 24)
            # For TRDG, we might be able to just use the font name
            actual_font_path = font_name
            print(f"Successfully matched font: {font_name}")
            
            # But if we can get the actual path, that might be more reliable
            if hasattr(font, 'path'):
                actual_font_path = font.path
            elif hasattr(font, 'filename'):
                actual_font_path = font.filename
        except Exception as e:
            print(f"FontMatcher couldn't load '{font_name}', falling back to path search: {e}")
    
    # If we don't have a path yet, try to find it manually
    if not actual_font_path:
        try:
            actual_font_path = get_font_path(font_name, font_dir, verbose=True, font_matcher=None)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            return 0  # Skip this font
        
    for i in range(samples_per_font):
        if images_created >= samples_per_font:
            break
            
        # Get a random text sample
        text_sample = random.choice(text_samples)
        
        # Randomize parameters (similar to the PIL version)
        size = random.randint(24, 70)
        skewing_angle = random.uniform(0, 4) if random.random() > 0.5 else 0
        blur = random.uniform(0, 1.5) if random.random() > 0.7 else 0
        # TODO add backgrounds 
        background_type = random.choice([0, 1, 2])
        character_spacing = random.uniform(-0.1, 0.6)
        
        # Random text color (30% chance of custom color)
        text_color = None
        if random.random() < 0.3:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            text_color = f"#{r:02x}{g:02x}{b:02x}"
        
        try:
            print(f"Generating image for '{font_name}' with text: {text_sample[:20]}...")
            print(f"Font path: {actual_font_path}")
            
            # Create a generator for this image - using the correct parameters from TRDG documentation
            generator = GeneratorFromStrings(
                [text_sample],
                fonts=[actual_font_path],
                size=size,
                skewing_angle=int(skewing_angle),
                random_skew=False,
                blur=int(blur),
                random_blur=False,
                background_type=background_type,
                character_spacing=int(character_spacing),
                text_color="#000000" if text_color is None else text_color, 
                width=int(512),  # Use width instead of width/height
                fit=False   # Don't use tight fit
            )
            
            # Get one image from this generator
            for img, lbl in generator:
                try:
                    # Save the image
                    img_filename = f"sample_{start_idx + images_created:04d}.jpg"
                    img_path = os.path.join(font_output_dir, img_filename)
                    
                    # Try to save the image
                    img.save(img_path, quality=90)
                    print(f"Saved image {img_filename} for font '{font_name}'")
                    images_created += 1
                    break
                except Exception as e:
                    print(f"Error saving image for font '{font_name}': {e}")
                    # Try alternate saving method if the first fails
                    try:
                        img.save(img_path)  # Try without quality parameter
                    except Exception as inner_e:
                        print(f"Alternative save method failed: {inner_e}")
        except Exception as e:
            print(f"Error generating image for font '{font_name}': {e}")
    
    return images_created


def generate_font_images(
    font_names: List[str],
    text_file_path: str,
    output_dir: str,
    font_dir: str,
    font_list_path,
    samples_per_font: int = 10,
    num_workers: int = None
):
    """Generate font images using TRDG."""
    # Read fonts and text
    
    text_content = read_text_file(text_file_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    font_matcher = None
    if has_font_matcher:
        try:
            # The FontMatcher takes a font list file and fonts directory
            font_matcher = FontMatcher(font_list_path, font_dir)  # Adjust as needed
            print("Successfully initialized FontMatcher")
        except Exception as e:
            print(f"Failed to initialize FontMatcher: {e}")
    
    
    # Set number of workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create arguments for each font
    args_list = []
    for font_name in font_names:
        # Remove the fallback_path logic and just pass font_dir
        text_samples = get_text_samples(text_content, samples_per_font)
        # Pass the font_matcher and font_dir to the process_font function
        args_list.append((font_name, font_dir, text_samples, output_dir, samples_per_font, 0, font_matcher))
    # Process fonts in parallel
    total_fonts = len(args_list)
    print(f"Processing {total_fonts} fonts with {samples_per_font} samples each...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_font, args_list), total=total_fonts))
    
    # Create a dataset description file
    # description_path = os.path.join(output_dir, "deb/dataset_info.txt")
    # with open(description_path, 'w') as f:
    #     f.write(f"Dataset generated using TextRecognitionDataGenerator\n")
    #     f.write(f"Fonts processed: {total_fonts}\n")
    #     f.write(f"Samples per font: {samples_per_font}\n")
    #     f.write(f"Text source: {text_file_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of font images using TRDG")
    parser.add_argument('--font_list', type=str, default='available_fonts.txt', 
                        help='Path to file containing font names')
    parser.add_argument('--text_file', type=str, default='lorem_ipsum.txt', 
                        help='Path to text file for content')
    parser.add_argument('--output_dir', type=str, default='trdg-font-images', 
                        help='Directory to save generated images')
    parser.add_argument('--font_dir', type=str, default='fonts', 
                        help='Directory containing font files')
    parser.add_argument('--samples_per_font', type=int, default=10, 
                        help='Number of samples to generate per font')
    parser.add_argument('--num_workers', type=int, default=None, 
                        help='Number of worker processes (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    font_names = read_font_list(args.font_list)
    generate_font_images(
        font_names=font_names,
        text_file_path=args.text_file,
        output_dir=args.output_dir,
        font_dir=args.font_dir,
        font_list_path=args.font_list,
        samples_per_font=args.samples_per_font,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()