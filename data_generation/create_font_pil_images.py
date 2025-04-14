import os
import random
import logging
import argparse
import json
import time
import multiprocessing
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FontConfig:
    """Configuration for a single font instance."""
    name: str
    output_path: Path
    image_width: int = 512
    image_height: int = 512
    font_size: int = 24
    font_weight: int = 400      
    letter_spacing: float = 0
    line_height: float = 1.5   
    font_style: str = "normal"  
    text_color: str = "#000000"  
    bg_color: str = "#FFFFFF"   
    samples_per_font: int = 10
    sample_id: int = 0


class TextAugmentation:
    """Generates continuous text augmentation parameters for dataset diversity."""
    
    def __init__(self, 
                 font_size_range=(40, 80),
                 weight_primary_modes=[400, 700],
                 weight_primary_prob=0.7,
                 letter_spacing_range=(-0.1, 0.4),
                 line_height_range=(1.1, 1.9)):
        
        self.font_size_range = font_size_range
        self.weight_primary_modes = weight_primary_modes
        self.weight_primary_prob = weight_primary_prob
        self.letter_spacing_range = letter_spacing_range
        self.line_height_range = line_height_range
        
        # Valid font weights (100-900 in increments of 100)
        self.valid_weights = list(range(100, 1000, 100))
        
    def sample_font_size(self):
        """Sample a font size from the specified range."""
        return round(random.uniform(*self.font_size_range))
    
    def sample_font_weight(self):
        """Sample a font weight using a mixture model approach."""
        if random.random() < self.weight_primary_prob:
            # Sample from primary modes (regular or bold)
            return random.choice(self.weight_primary_modes)
        else:
            # Sample from full range for diversity
            return random.choice(self.valid_weights)
    
    def sample_letter_spacing(self):
        """Sample a letter spacing value as a continuous parameter."""
        return round(random.uniform(*self.letter_spacing_range), 2)
    
    def sample_line_height(self):
        """Sample a line height multiplier."""
        return round(random.uniform(*self.line_height_range), 2)
    
    def generate_config(self, font_name, output_dir, **kwargs):
        """Generate a FontConfig with augmented parameters."""
        font_size = self.sample_font_size()
        font_weight = self.sample_font_weight()
        letter_spacing = self.sample_letter_spacing()
        line_height = self.sample_line_height()
        
        sample_id = kwargs.get('sample_id', 0)
    
        # Create unique identifier for this configuration
        config_id = f"{font_name.lower().replace(' ', '_')}_sample{sample_id}"

        return FontConfig(
            name=font_name,
            output_path=Path(output_dir) / config_id,
            image_width=kwargs.get('image_width', 512),
            image_height=kwargs.get('image_height', 512),
            font_size=font_size,
            font_weight=font_weight,
            letter_spacing=letter_spacing,
            line_height=line_height,
            font_style=kwargs.get('font_style', 'normal'),
            text_color=kwargs.get('text_color', '#000000'),
            bg_color=kwargs.get('bg_color', '#FFFFFF'),
            samples_per_font=kwargs.get('samples_per_font', 10),
            sample_id=sample_id,
        )

class FontManager:
    """Download and manage font files."""
    
    def __init__(self, cache_dir="./fonts", use_local_fallbacks=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.font_cache = {}
        self.use_local_fallbacks = use_local_fallbacks
        
        # Create a directory for fallback fonts
        self.fallback_dir = self.cache_dir / "fallbacks"
        self.fallback_dir.mkdir(exist_ok=True)
        
        # Try to copy some system fonts as fallbacks
        self._setup_fallback_fonts()
    
    def _setup_fallback_fonts(self):
        """Copy some system fonts as fallbacks."""
        if not self.use_local_fallbacks:
            return
            
        # Common system font locations
        system_font_locations = [
            # Windows
            "C:/Windows/Fonts",
            # macOS
            "/Library/Fonts",
            "/System/Library/Fonts",
            # Linux
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            "~/.fonts"
        ]
        
        # Common fallback fonts to look for
        fallback_fonts = [
            "Arial.ttf", "arial.ttf",
            "Verdana.ttf", "verdana.ttf",
            "TimesNewRoman.ttf", "times.ttf",
            "DejaVuSans.ttf", "DejaVuSans.ttf",
            "Roboto-Regular.ttf", "Roboto.ttf",
            "NotoSans-Regular.ttf", "NotoSans.ttf",
            "OpenSans-Regular.ttf", "OpenSans.ttf"
        ]
        
        # Look for fallback fonts
        found_fallback = False
        for font_dir in system_font_locations:
            font_path = Path(os.path.expanduser(font_dir))
            if not font_path.exists():
                continue
                
            for font_name in fallback_fonts:
                system_font_path = font_path / font_name
                if system_font_path.exists():
                    try:
                        fallback_font_path = self.fallback_dir / font_name
                        if not fallback_font_path.exists():
                            # Copy the font to our fallbacks directory
                            shutil.copy(system_font_path, fallback_font_path)
                            logger.info(f"Copied fallback font: {font_name}")
                            found_fallback = True
                    except Exception as e:
                        logger.warning(f"Failed to copy fallback font {font_name}: {e}")
        
        if not found_fallback:
            logger.warning("No fallback fonts found. Text rendering may fail.")
    
    def _download_font(self, font_name, weight, style, output_path):
        """Instead of trying to download fonts, use fallbacks."""
        # We'll skip downloading and rely on fallbacks instead
        logger.warning(f"Font not found: {font_name}, weight: {weight}, style: {style}")
        return None

    def get_font_path(self, font_name, weight=400, style="normal"):
        """Get a path to a font, using fallbacks if necessary."""
        # First, check if we have this exact font cached
        normalized_name = font_name.replace(" ", "")
        weight_str = self._weight_to_string(weight)
        style_str = style.capitalize() if style != "normal" else ""
        
        # Try different filename variations
        possible_filenames = [
            f"{normalized_name}-{weight_str}{style_str}.ttf",
            f"{normalized_name}{weight_str}{style_str}.ttf",
            f"{normalized_name}-{weight_str}.ttf",
            f"{normalized_name}.ttf"
        ]
        
        # Remove empty components
        possible_filenames = [f for f in possible_filenames if "--" not in f]
        possible_filenames = list(set(possible_filenames))  # Remove duplicates
        
        # Check for exact match in cache directory
        for filename in possible_filenames:
            font_path = self.cache_dir / filename
            if font_path.exists():
                return str(font_path)
        
        # If we don't have the exact font, try to find a fallback
        return self._get_fallback_font(font_name, weight, style)
    
    def _weight_to_string(self, weight):
        """Convert numeric weight to string representation."""
        weight_map = {
            100: "Thin",
            200: "ExtraLight",
            300: "Light",
            400: "Regular",
            500: "Medium",
            600: "SemiBold",
            700: "Bold",
            800: "ExtraBold",
            900: "Black"
        }
        return weight_map.get(weight, "Regular")
    
    def _get_fallback_font(self, font_name, weight, style):
        """Find a suitable fallback font."""
        # Check fallbacks directory first
        fallback_fonts = list(self.fallback_dir.glob("*.ttf"))
        if fallback_fonts:
            return str(fallback_fonts[0])
        
        # If no fallbacks found, use PIL's default font
        logger.warning(f"No fallback font found for {font_name}. Using default.")
        return None
    
    def load_font(self, font_name, font_size, weight=400, style="normal"):
        """Load a font with the specified name, size, weight, and style."""
        cache_key = f"{font_name}_{font_size}_{weight}_{style}"

        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        # Simply use PIL's default font and warn the user
        logger.warning(f"Using default font for {font_name}")
        font = ImageFont.load_default()

        self.font_cache[cache_key] = font
        return font


class TextRenderer:
    """Render text as images with various augmentations."""
    
    def __init__(self, font_manager, backgrounds_dir=None, background_probability=0.5):
        self.font_manager = font_manager
        self.backgrounds_dir = backgrounds_dir
        self.background_probability = background_probability
        
        # Load background images if directory is provided
        self.background_images = []
        if backgrounds_dir:
            backgrounds_path = Path(backgrounds_dir)
            if backgrounds_path.exists():
                self.background_images = list(backgrounds_path.glob("*.jpg")) + list(backgrounds_path.glob("*.png"))
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _create_background(self, width, height, bg_color="#FFFFFF"):
        """Create a background image."""
        if self.backgrounds_dir and self.background_images and random.random() < self.background_probability:
            # Use a random background image
            bg_path = random.choice(self.background_images)
            try:
                with Image.open(bg_path) as bg:
                    bg = bg.convert("RGB")
                    bg = bg.resize((width, height), Image.LANCZOS)
                    return bg
            except Exception as e:
                logger.error(f"Error loading background image {bg_path}: {e}")
        
        # Fall back to solid color if no background image or an error occurred
        bg_color_rgb = self._hex_to_rgb(bg_color)
        return Image.new("RGB", (width, height), bg_color_rgb)
    
    def _apply_augmentations(self, image, config):
        """Apply various augmentations to the image."""
        # Gaussian blur
        # if random.random() < 0.3:  # 30% chance of applying blur
        #     blur_radius = random.uniform(0, 1.5)
        #     image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Slight rotation
        if random.random() < 0.3:  # 30% chance of rotation
            rotation_angle = random.uniform(-5, 5)
            image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)
        
        # Add slight noise
        if random.random() < 0.2:  # 20% chance of noise
            noise_level = random.uniform(5, 15)
            img_array = np.array(image)
            noise = np.random.normal(0, noise_level, img_array.shape)
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_array)
        
        # Adjust brightness/contrast slightly
        if random.random() < 0.3:  # 30% chance of brightness/contrast adjustment
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
            
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        return image
    
    def _get_char_bounding_boxes(self, text, draw, font, start_pos, letter_spacing):
        """Get the bounding boxes for each character in the text."""
        boxes = []
        x, y = start_pos
        
        for char in text:
            if char.isspace():
                # For space, we need to estimate its width
                space_width = font.getlength(" ")
                x += space_width + letter_spacing
                continue
            
            # Get character dimensions
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                
                # Create the bounding box (x1, y1, x2, y2)
                box = (x, y + bbox[1], x + char_width, y + bbox[3])
            except AttributeError:
                # Fallback for older PIL versions
                char_width = font.getlength(char)
                ascent, descent = font.getmetrics()
                char_height = ascent + descent
                
                # Create the bounding box (x1, y1, x2, y2)
                box = (x, y, x + char_width, y + char_height)
            
            boxes.append((char, box))
            
            # Move to the next character position
            x += char_width + letter_spacing
        
        return boxes
    
    def render_text_image(self, text, config):
        """Render text as an image with the specified configuration."""
        # Create background
        image = self._create_background(config.image_width, config.image_height, config.bg_color)
        draw = ImageDraw.Draw(image)
        
        # Load font
        font = self.font_manager.load_font(
            config.name, 
            config.font_size, 
            config.font_weight, 
            config.font_style
        )
        
        # Get text metrics
        try:
            # For newer PIL versions
            text_width = font.getbbox(text)[2]
            ascent, descent = font.getmetrics()
            text_height = ascent + descent
        except AttributeError:
            # Fallback for older PIL versions
            text_width = font.getlength(text)
            ascent, descent = font.getmetrics()
            text_height = ascent + descent
        
        # Calculate text position (centered horizontally, near the top vertically)
        start_x = (config.image_width - text_width) // 2
        start_y = config.image_height // 4  # Positioned at 1/4 down from the top
        
        # Check if text is too wide for the image
        if text_width > config.image_width * 0.9:
            # Break the text into multiple lines
            words = text.split()
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                try:
                    word_width = font.getbbox(word + " ")[2]
                except AttributeError:
                    word_width = font.getlength(word + " ")
                    
                if current_width + word_width <= config.image_width * 0.9:
                    current_line.append(word)
                    current_width += word_width
                else:
                    if current_line:  # Only append if there are words
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    try:
                        current_width = font.getbbox(word)[2]
                    except AttributeError:
                        current_width = font.getlength(word)
                    
            if current_line:  # Add the last line if it exists
                lines.append(" ".join(current_line))
                
            # Render each line
            y = start_y
            all_char_boxes = []
            
            for line in lines:
                try:
                    line_width = font.getbbox(line)[2]
                except AttributeError:
                    line_width = font.getlength(line)
                    
                line_x = (config.image_width - line_width) // 2
                
                draw.text((line_x, y), line, font=font, fill=self._hex_to_rgb(config.text_color))
                
                line_char_boxes = self._get_char_bounding_boxes(
                    line, 
                    draw, 
                    font, 
                    (line_x, y), 
                    config.letter_spacing
                )
                all_char_boxes.extend(line_char_boxes)
                
                y += text_height * config.line_height
                
            char_boxes = all_char_boxes
        else:
            # Draw single line text
            draw.text((start_x, start_y), text, font=font, fill=self._hex_to_rgb(config.text_color))
            
            # Get character bounding boxes
            char_boxes = self._get_char_bounding_boxes(
                text, 
                draw, 
                font, 
                (start_x, start_y), 
                config.letter_spacing
            )
        
        # Apply augmentations
        image = self._apply_augmentations(image, config)
        
        return image, char_boxes


class FontDatasetGenerator:
    """Generate a dataset of font images."""
    
    def __init__(self, 
                 fonts_file="fonts.txt", 
                 text_file="lorem_ipsum.txt",
                 output_dir="font-images",
                 num_samples_per_font=10,
                 image_size=(512, 512),
                 backgrounds_dir=None,
                 background_probability=0.5):
        
        self.fonts_file = fonts_file
        self.text_file = text_file
        self.output_dir = Path(output_dir)
        self.num_samples_per_font = num_samples_per_font
        self.image_size = image_size
        self.backgrounds_dir = backgrounds_dir
        self.background_probability = background_probability
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up components
        self.font_manager = FontManager()
        self.text_renderer = TextRenderer(
            self.font_manager,
            backgrounds_dir,
            background_probability
        )
        self.augmenter = TextAugmentation()
        
        # Load fonts and text
        self.fonts = self._load_fonts()
        self.text = self._load_text()
    
    def _load_fonts(self):
        """Load font names from the fonts file."""
        try:
            with open(self.fonts_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Fonts file not found: {self.fonts_file}")
            raise
    
    def _load_text(self):
        """Load text from the text file."""
        try:
            with open(self.text_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Text file not found: {self.text_file}")
            raise
    
    def _get_text_sample(self, length=50):
        """Get a sample of text with the specified length."""
        if len(self.text) <= length:
            return self.text
        
        # Get a random starting position
        start_pos = random.randint(0, len(self.text) - length - 1)
        
        # Find the nearest space to start at a word boundary
        while start_pos > 0 and self.text[start_pos] != ' ':
            start_pos -= 1
        
        # Get the text sample
        end_pos = start_pos + length
        while end_pos < len(self.text) and self.text[end_pos] != ' ':
            end_pos += 1
        
        return self.text[start_pos:end_pos].strip()
    
    def _process_font(self, font_name, sample_id):
        """Process a single font to generate a sample image."""
        try:
            # Create a new font manager for this process to avoid multiprocessing issues
            process_font_manager = FontManager()
            process_text_renderer = TextRenderer(
                process_font_manager,
                self.backgrounds_dir,
                self.background_probability
            )
            
            # Generate a configuration for this font sample
            config = self.augmenter.generate_config(
                font_name,
                str(self.output_dir),
                image_width=self.image_size[0],
                image_height=self.image_size[1],
                samples_per_font=self.num_samples_per_font,
                sample_id=sample_id
            )
            
            # Get a text sample
            text_sample = self._get_text_sample()
            
            # Render the text image
            image, char_boxes = process_text_renderer.render_text_image(text_sample, config)
            
            # Create output directories
            font_dir = self.output_dir / font_name.lower().replace(' ', '_')
            font_dir.mkdir(exist_ok=True)
            
            annotations_dir = font_dir / "annotations"
            annotations_dir.mkdir(exist_ok=True)
            
            # Save the image
            image_filename = f"sample_{sample_id:04d}.jpg"
            image_path = font_dir / image_filename
            image.save(image_path, quality=90)
            
            # Save annotations
            self._save_annotations(annotations_dir, sample_id, text_sample, char_boxes, self.image_size)
            
            logger.info(f"Generated sample {sample_id} for font {font_name}")
            return True
        except Exception as e:
            logger.error(f"Error generating sample for font {font_name}: {e}")
            return False
    
    def _save_annotations(self, annotations_dir, sample_id, text, char_boxes, image_size):
        """Save annotations for the image."""
        # Save YOLO format annotations
        yolo_path = annotations_dir / f"sample_{sample_id:04d}.txt"
        json_path = annotations_dir / f"sample_{sample_id:04d}.json"
        
        width, height = image_size
        
        # Generate YOLO annotations
        yolo_lines = []
        char_mapping = {}
        
        for char, box in char_boxes:
            x1, y1, x2, y2 = box
            
            # Skip if character is whitespace
            if char.isspace():
                continue
            
            # Convert to YOLO format (class_id, x_center, y_center, width, height)
            char_code = ord(char)
            char_class = char_code % 256  # Simple mapping
            
            # Add to mapping
            char_mapping[char_class] = char
            
            # Calculate normalized coordinates
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            # Ensure values are within bounds
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            box_width = max(0, min(1, box_width))
            box_height = max(0, min(1, box_height))
            
            yolo_line = f"{char_class} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Save YOLO annotations
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Save JSON annotations for reference
        json_data = {
            'image': f"sample_{sample_id:04d}.jpg",
            'font': text,
            'char_boxes': [
                {'char': char, 'box': box}
                for char, box in char_boxes if not char.isspace()
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save character mapping
        mapping_path = annotations_dir / "classes.txt"
        with open(mapping_path, 'w') as f:
            for class_id, char in sorted(char_mapping.items()):
                f.write(f"{class_id} {char}\n")
    
    def generate_dataset(self):
        """Generate the dataset."""
        logger.info(f"Starting dataset generation with {len(self.fonts)} fonts")
        
        # Generate configurations for all font samples
        font_configs = []
        for font in self.fonts:
            for sample_id in range(self.num_samples_per_font):
                font_configs.append((font, sample_id))
        
        logger.info(f"Generated {len(font_configs)} font configurations")
        
        # Process font configurations in parallel
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for font_name, sample_id in font_configs:
                futures.append(executor.submit(self._process_font, font_name, sample_id))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Font processing failed: {e}")
        
        self._create_dataset_description()
    
    def _create_dataset_description(self):
        """Create a description file for the dataset."""
        description_path = self.output_dir / "dataset_info.txt"
        with open(description_path, 'w') as f:
            f.write(f"Dataset generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Fonts processed: {len(self.fonts)}\n")
            f.write(f"Samples per font: {self.num_samples_per_font}\n")
            f.write(f"Image size: {self.image_size[0]}x{self.image_size[1]}\n")
            f.write(f"Text source: {self.text_file}\n")
            if self.backgrounds_dir:
                f.write(f"Background images: {self.backgrounds_dir}\n")
                f.write(f"Background probability: {self.background_probability}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of font images")
    parser.add_argument('--text_file', default='lorem_ipsum.txt', help='Path to text file for content')
    parser.add_argument('--font_file', default='fonts.txt', help='Path to font names file')
    parser.add_argument('--output_dir', default='font-images', help='Directory to save generated images')
    parser.add_argument('--samples_per_class', type=int, default=10, help='Number of samples per font')
    parser.add_argument('--image_resolution', type=int, default=512, help='Image width and height')
    parser.add_argument('--backgrounds_dir', default=None, help='Directory containing background images')
    parser.add_argument('--background_probability', type=float, default=0.5, 
                      help='Probability of using a background (0-1)')
    
    args = parser.parse_args()
    
    generator = FontDatasetGenerator(
        fonts_file=args.font_file,
        text_file=args.text_file,
        output_dir=args.output_dir,
        num_samples_per_font=args.samples_per_class,
        image_size=(args.image_resolution, args.image_resolution),
        backgrounds_dir=args.backgrounds_dir,
        background_probability=args.background_probability
    )
    
    try:
        generator.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)


if __name__ == "__main__":
    main()