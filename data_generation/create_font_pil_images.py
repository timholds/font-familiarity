import os
import gc
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
from match_fonts import FontMatcher
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_rotation_matrix(width, height, thetaX=0, thetaY=0, thetaZ=0):
    """Provide a rotation matrix about the center of a rectangle with
    a given width and height.
    
    Args:
        width: The width of the rectangle
        height: The height of the rectangle
        thetaX: Rotation about the X axis (in radians)
        thetaY: Rotation about the Y axis (in radians)
        thetaZ: Rotation about the Z axis (in radians)
        
    Returns:
        A 3x3 transformation matrix
    """
    # Translation to center
    translate1 = np.array([
        [1, 0, width / 2],
        [0, 1, height / 2],
        [0, 0, 1]
    ])
    
    # Rotation around X axis
    rotX = np.array([
        [1, 0, 0],
        [0, np.cos(thetaX), -np.sin(thetaX)],
        [0, np.sin(thetaX), np.cos(thetaX)]
    ])
    
    # Rotation around Y axis
    rotY = np.array([
        [np.cos(thetaY), 0, np.sin(thetaY)],
        [0, 1, 0],
        [-np.sin(thetaY), 0, np.cos(thetaY)]
    ])
    
    # Rotation around Z axis
    rotZ = np.array([
        [np.cos(thetaZ), -np.sin(thetaZ), 0],
        [np.sin(thetaZ), np.cos(thetaZ), 0],
        [0, 0, 1]
    ])
    
    # Translation back
    translate2 = np.array([
        [1, 0, -width / 2],
        [0, 1, -height / 2],
        [0, 0, 1]
    ])
    
    # Combine transformations
    M = np.dot(translate1, np.dot(rotX, np.dot(rotY, np.dot(rotZ, translate2))))
    return M[:2, :]  # Return 2x3 matrix for OpenCV warpAffine

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
                 font_size_range=(18, 60),
                 weight_primary_modes=[400, 700],
                 weight_primary_prob=0.3,
                 letter_spacing_range=(-0.1, 0.6),
                 line_height_range=(.8, 1.2),
                 color_probability=0.3):
        
        self.font_size_range = font_size_range
        self.weight_primary_modes = weight_primary_modes
        self.weight_primary_prob = weight_primary_prob
        self.letter_spacing_range = letter_spacing_range
        self.line_height_range = line_height_range
        self.color_probability = color_probability
        
        # Valid font weights (100-900 in increments of 100)
        self.valid_weights = list(range(100, 1000, 100))
        
    def sample_color(self):
        """Sample a random color in hex format."""
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def sample_colors_with_contrast(self, min_contrast=125):
        """Sample text and background colors with sufficient contrast."""
        # Generate first color
        color1 = self.sample_color()
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)

        # Keep generating second color until we have sufficient contrast
        while True:
            color2 = self.sample_color()
            r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
            
            # Calculate contrast based on luminance difference
            lum1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
            lum2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
            contrast = abs(lum1 - lum2)
            
            if contrast > min_contrast:
                break

        # Decide if we want dark text on light background or vice versa
        if random.random() < 0.5:  # 50% chance to swap
            return color1, color2
        else:
            return color2, color1

    def sample_font_size(self):
        """Sample a font size using a distribution that reflects real-world usage."""
        return round(random.uniform(*self.font_size_range))
        
        # rand = random.random()

        # # Small text (10% of samples) - footnotes, captions
        # if rand < 0.1:
        #     return round(random.uniform(self.font_size_range[0], 14))
        # # Body text (50% of samples) - most common size range
        # elif rand < 0.4:
        #     return round(random.uniform(14, 32))
        # # Heading text (20% of samples)
        # elif rand < 0.7:
        #     return round(random.uniform(32, 48))
        # # Display text (20% of samples) - large, decorative text
        # else:
        #     return round(random.uniform(48, self.font_size_range[1]))
    
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
        text_color = '#000000'
        bg_color = '#FFFFFF'

        # 50% chance to use custom colors with good contrast
        if random.random() < self.color_probability:
            text_color, bg_color = self.sample_colors_with_contrast()
        
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
            text_color=text_color,
            bg_color=bg_color,
            samples_per_font=kwargs.get('samples_per_font', 10),
            sample_id=sample_id,
        )

# In create_font_pil_images.py, replace the current FontManager class with this:

class FontManager:
    """Manage font files for PIL text rendering."""
    
    def __init__(self, font_matcher=None, cache_dir="./fonts"):
        self.font_matcher = font_matcher
        self.cache_dir = Path(cache_dir)
        self.font_cache = {}  # Cache for loaded fonts
        
        # Create a directory for fallback fonts if not using font_matcher
        if not font_matcher:
            raise ValueError("FontMatcher is required to initialize FontManager")
   
    
    def load_font(self, font_name, font_size):
        """
        Load a font with the specified name and size.
        
        Args:
            font_name: Font name as it appears in the font list file
            font_size: Size of the font to load
            
        Returns:
            PIL ImageFont object
            
        Raises:
            ValueError: If the font is not found or cannot be loaded
        """
        try:
            return self.font_matcher.load_font(font_name, font_size)
        except Exception as e:
            logger.error(f"Failed to load font '{font_name}' at size {font_size}: {e}")
            raise ValueError(f"Failed to load font '{font_name}': {e}")
        # cache_key = f"{font_name}_{font_size}"

        # if cache_key in self.font_cache:
        #     return self.font_cache[cache_key]

        # # Get the font file path
        # try:
        #     font_path = self.get_font_file(font_name)
        # except ValueError as e:
        #     logger.error(f"Font file not found for '{font_name}': {e}")
        #     raise

        # # Verify the font file exists
        # if not os.path.exists(font_path):
        #     error_msg = f"Font file '{font_path}' does not exist for font '{font_name}'"
        #     logger.error(error_msg)
        #     raise ValueError(error_msg)

        # try:
        #     # Load the font
        #     font = ImageFont.truetype(font_path, font_size)
            
        #     # Verify the font loaded correctly by testing a method
        #     test_text = "Test"
        #     try:
        #         # Handle different PIL versions
        #         if hasattr(font, 'getbbox'):
        #             _ = font.getbbox(test_text)
        #         elif hasattr(font, 'getsize'):
        #             _ = font.getsize(test_text)
        #         else:
        #             error_msg = f"Cannot determine text metrics for font '{font_name}'"
        #             logger.error(error_msg)
        #             raise ValueError(error_msg)
        #     except Exception as e:
        #         error_msg = f"Font '{font_name}' loaded but failed metrics test: {e}"
        #         logger.error(error_msg)
        #         raise ValueError(error_msg)
                
        #     # If we got here, the font is valid
        #     self.font_cache[cache_key] = font
        #     return font
        # except Exception as e:
        #     error_msg = f"Failed to load font '{font_name}' from {font_path}: {e}"
        #     logger.error(error_msg)
        #     raise ValueError(error_msg)
    

class TextRenderer:
    """Render text as images with various augmentations."""
    
    def __init__(self, font_manager, backgrounds_dir=None, 
                 background_probability=0.25,
                 transform_probability=0.25):
        self.font_manager = font_manager
        self.backgrounds_dir = backgrounds_dir
        self.background_probability = background_probability
        self.transform_probability = transform_probability
        
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
        #     _blur_radius = random.uniform(0, 1.5)
        #     image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        width, height = image.size
        img_array = np.array(image)

        # Choose transformation strategy
        # transform_type = random.choices(
        #     ["affine_only", "perspective_only", "3d_rotation_only", "2d_rotation_only", "combined"],
        #     weights=[0.1, 0.1, 0.1, 0.6, 0.1]  # 30% chance of combined transforms
        # )[0]

        # if "affine" in transform_type or transform_type == "combined":
        #     # Apply affine transformation (with gentler values for combined case)
        #     scale_factor = 0.1 if transform_type == "combined" else .5
        #     rotation = random.uniform(-3, 3) * scale_factor
        #     scale_x = random.uniform(0.97, 1.03)
        #     scale_y = random.uniform(0.97, 1.03)
        #     shear_x = random.uniform(-0.03, 0.03) * scale_factor
        #     shear_y = random.uniform(-0.02, 0.02) * scale_factor
            
        #     # Create affine matrix
        #     M = np.float32([
        #         [scale_x * np.cos(np.radians(rotation)), 
        #          scale_x * (np.sin(np.radians(rotation)) + shear_x), 0],
        #         [scale_y * (-np.sin(np.radians(rotation)) + shear_y), 
        #          scale_y * np.cos(np.radians(rotation)), 0]
        #     ])
            
        #     img_array = cv2.warpAffine(img_array, M, (width, height), 
        #                                 borderMode=cv2.BORDER_REPLICATE)

        # if "perspective" in transform_type or transform_type == "combined":
        #     # Apply perspective transformation (gentler for combined case)
        #     scale_factor = 0.6 if transform_type == "combined" else 1.0
        #     src_points = np.array([
        #         [0, 0], [width, 0], [width, height], [0, height]
        #     ], dtype=np.float32)
            
        #     max_shift = min(width, height) * 0.03 * scale_factor
        #     dst_points = np.array([
        #         [0 + random.uniform(-max_shift, max_shift), 
        #             0 + random.uniform(-max_shift, max_shift)],
        #         [width + random.uniform(-max_shift, max_shift), 
        #             0 + random.uniform(-max_shift, max_shift)],
        #         [width + random.uniform(-max_shift, max_shift), 
        #             height + random.uniform(-max_shift, max_shift)],
        #         [0 + random.uniform(-max_shift, max_shift), 
        #             height + random.uniform(-max_shift, max_shift)]
        #     ], dtype=np.float32)
            
        #     M = cv2.getPerspectiveTransform(src_points, dst_points)
        #     img_array = cv2.warpPerspective(img_array, M, (width, height), 
        #                                     borderMode=cv2.BORDER_REPLICATE)

        # if "3d_rotation" in transform_type:
        #     # Apply 3D rotation projection
        #     thetaX = random.uniform(-0.002, 0.002)
        #     thetaY = random.uniform(-0.002, 0.002) 
        #     thetaZ = random.uniform(-0.005, 0.005)
            
        #     M = get_rotation_matrix(width, height, thetaX, thetaY, thetaZ)
        #     img_array = cv2.warpAffine(img_array, M, (width, height), 
        #                                 borderMode=cv2.BORDER_REPLICATE)

        # Convert back to PIL
        image = Image.fromarray(img_array)
        # Slight rotation
        # if "2d_rotation" in transform_type or transform_type == "combined":
        # if random.random() < .3:  # 30% chance of rotation
        #     rotation_angle = random.uniform(-4, 4)
        #     image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)
        
        # Add slight noise
        # if random.random() < 0.2:  # 20% chance of noise
        #     noise_level = random.uniform(5, 15)
        #     img_array = np.array(image)
        #     noise = np.random.normal(0, noise_level, img_array.shape)
        #     noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        #     image = Image.fromarray(noisy_array)
        
        # Adjust brightness/contrast slightly
        # if random.random() < 0.3:  # 30% chance of brightness/contrast adjustment
        #     enhancer = ImageEnhance.Brightness(image)
        #     factor = random.uniform(0.8, 1.2)
        #     image = enhancer.enhance(factor)
            
        #     enhancer = ImageEnhance.Contrast(image)
        #     factor = random.uniform(0.8, 1.2)
        #     image = enhancer.enhance(factor)
        
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
        
        # Calculate text position (centered horizontally, adjusted vertically with padding)
        start_x = (config.image_width - text_width) // 2
        # start_y = config.image_height // 8  # Positioned at 1/8 down from the top
        padding_bottom= int(config.image_height * 0.2)
        start_y = (config.image_height - text_height - padding_bottom) // 8
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
                 background_probability=0.5,
                 color_probability=0.25,
                 transform_probability=0.25):
        
        self.fonts_file = fonts_file
        self.text_file = text_file
        self.output_dir = Path(output_dir)
        self.num_samples_per_font = num_samples_per_font
        self.image_size = image_size
        self.backgrounds_dir = backgrounds_dir
        self.background_probability = background_probability
        self.color_probability = color_probability
        self.transform_probability = transform_probability

        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up components
        font_matcher = FontMatcher(self.fonts_file, "fonts")
        self.font_manager = FontManager(font_matcher=font_matcher)
        self.text_renderer = TextRenderer(
            self.font_manager,
            backgrounds_dir,
            background_probability
        )
        self.augmenter = TextAugmentation(color_probability=color_probability)
        
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
    
    def _get_text_sample(self, length=150):
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
            process_font_manager = FontManager(font_matcher=self.font_manager.font_matcher)
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
            
            # Save the image
            image_filename = f"sample_{sample_id:04d}.jpg"
            image_path = font_dir / image_filename
            image.save(image_path, quality=90)
     
            #logger.info(f"Generated sample {sample_id} for font {font_name}")
            return True
        except Exception as e:
            logger.error(f"Error generating sample for font {font_name}: {e}")
            return False
    
   
    def generate_dataset(self):
        """Generate the dataset by processing one font at a time."""
        logger.info(f"Starting dataset generation with {len(self.fonts)} fonts")
        
        num_workers = max(1, multiprocessing.cpu_count())
        batch_size = 256 
        
        # Process one font at a time
        for font_idx, font in enumerate(self.fonts):
            logger.info(f"Processing font {font_idx+1}/{len(self.fonts)}: {font}")
            
            # Create font directory (if needed)
            font_dir = self.output_dir / font.lower().replace(' ', '_')
            font_dir.mkdir(exist_ok=True)
            
            # Process this font in batches
            for batch_start in range(0, self.num_samples_per_font, batch_size):
                batch_end = min(batch_start + batch_size, self.num_samples_per_font)
                sample_ids = range(batch_start, batch_end)
                
                logger.info(f"  Processing samples {batch_start+1}-{batch_end} for font {font}")
                
                # Use a fresh pool for each batch
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for sample_id in sample_ids:
                        futures.append(executor.submit(self._process_font, font, sample_id))
                    
                    # Process all futures in this batch
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Font processing failed: {e}")
                
                # Clean up between batches
                gc.collect()
                
                # Optional: Log progress after each batch
                logger.info(f"  Completed batch for font {font} ({batch_end}/{self.num_samples_per_font})")
        
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
            f.write(f"Color variation probability: {self.color_probability}\n")
            f.write(f"Perspective transformation probability: {self.transform_probability}\n")
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
    parser.add_argument('--background_probability', type=float, default=0.2, 
                      help='Probability of using a background (0-1)')
    parser.add_argument('--color_probability', type=float, default=0.2,
                      help='Probability of using custom text and background colors (0-1)')
    parser.add_argument('--transform_probability', type=float, default=0.15,
                      help='Probability of using custom text and background colors (0-1)')
    args = parser.parse_args()
    
    generator = FontDatasetGenerator(
        fonts_file=args.font_file,
        text_file=args.text_file,
        output_dir=args.output_dir,
        num_samples_per_font=args.samples_per_class,
        image_size=(args.image_resolution, args.image_resolution//2),
        backgrounds_dir=args.backgrounds_dir,
        background_probability=args.background_probability,
        color_probability=args.color_probability,
        transform_probability=args.transform_probability
    )
    
    try:
        generator.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)


if __name__ == "__main__":
    main()