from pathlib import Path
from typing import List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import os
from PIL import Image
import io
import json
import random

from flask import Flask, render_template
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
import argparse

# Running main takes the list of fonts passed in and 
# renders the text file passed in each font
# it will generate screenshots of the text to save
# for a machine learning task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_text_multiplier(text: str, 
                          font_size: int,
                          line_height: float,
                          container_height: int,
                          samples_per_class: int) -> int:
    """
    Calculate how many times to repeat the text to ensure enough content for all samples.
    
    Args:
        text: Base text content
        font_size: Font size in pixels
        line_height: Line height multiplier (e.g. 1.5)
        container_height: Height of the container in pixels
        samples_per_class: Number of samples needed per font
        
    Returns:
        int: Number of times to repeat the text
    """
    # Calculate approximate height of one line
    line_height_px = font_size * line_height
    
    # Calculate lines visible in one container
    lines_per_container = container_height / line_height_px
    
    # Calculate approximate characters per line 
    # Assuming average char width is 0.6 * font_size
    chars_per_line = container_height / (font_size * 0.6)
    
    # Calculate total lines needed for all samples
    # Add one extra container worth of lines to ensure enough content
    total_lines_needed = lines_per_container * (samples_per_class + 1)
    
    # Calculate total characters needed
    total_chars_needed = total_lines_needed * chars_per_line
    
    # Calculate multiplier (rounding up)
    text_length = len(text)
    multiplier = int(total_chars_needed / text_length) + 1
    
    return max(1, multiplier)  # Ensure at least one copy

def prepare_text_content(filename: str, **kwargs) -> str:
    """
    Load and prepare text content for rendering.
    
    Args:
        filename: Path to text file
        **kwargs: Arguments to pass to calculate_text_multiplier
        
    Returns:
        str: Prepared text content
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            raise ValueError("Text file is empty")
            
        # Calculate required repetitions
        multiplier = calculate_text_multiplier(text, **kwargs)
        prepared_text = text * multiplier
        
        return prepared_text
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {filename}")

@dataclass
class FontConfig:
    """Configuration for font rendering"""
    name: str
    output_path: Path
    image_width: int = 512
    image_height: int = 512
    font_size: int = 24
    samples_per_font: int = 10


class FontRenderer:
    def __init__(self, 
                 fonts_file: str = 'fonts.txt',
                 text_file: str = 'lorem_ipsum.txt',
                 output_dir: str = 'font-images',
                 template_dir: str = 'templates',
                 port: int = 5100,
                 image_size: tuple = (256, 256),  # Reduced from 512x512
                 image_quality: int = 80, # JPEG quality (0-100))
                 num_samples_per_font: int = 10,
                 font_size: int = 24,
                 line_height: float = 1.5):       
        
        self.font_size = font_size
        self.line_height = line_height
        self.image_size = image_size
        self.image_quality = image_quality
        self.num_samples_per_font = num_samples_per_font
        self.scroll_height = 400
        self.flask_app = None
        self.server_thread = None
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.port = port

        self.fonts = self._load_fonts(fonts_file)
        self.text = prepare_text_content(
            text_file,
            font_size=self.font_size,
            line_height=self.line_height,
            container_height=self.image_size[1],
            samples_per_class=self.num_samples_per_font
        )
        #self.text = self._load_text(text_file)
        
        # Ensure output directory exists
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    
    def _load_fonts(self, filename: str) -> List[str]:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                fonts = [line.strip() for line in f if line.strip()]
            if not fonts:
                raise ValueError("No fonts found in fonts file")
            logger.info(f"Loaded {len(fonts)} fonts from {filename}")
            return fonts
        except FileNotFoundError:
            raise FileNotFoundError(f"Fonts file not found: {filename}")

    def _setup_webdriver(self) -> webdriver.Chrome:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        # Create service object
        service = Service()
        return webdriver.Chrome(service=service, options=chrome_options)

    def start_flask(self):
        """Start Flask server in a controlled way"""
        if self.flask_app is None:
            self.flask_app = Flask(__name__, template_folder=str(self.template_dir.absolute()))
            
            @self.flask_app.route('/font/<font_name>')
            def render_font(font_name):
                font_config = FontConfig(
                    name=font_name,
                    output_path=self.output_dir / font_name.lower().replace(' ', '_'),
                    image_width=self.image_size[0],
                    image_height=self.image_size[1],
                    font_size=self.font_size,
                    samples_per_font=self.num_samples_per_font
                )
                return render_template(
                    'single_font_detection.html',
                    font=font_config,
                    text=self.text,
                    font_size=self.font_size,
                    line_height=self.line_height
                )
            
            from threading import Thread
            self.server_thread = Thread(target=lambda: self.flask_app.run(port=self.port, debug=False))
            self.server_thread.daemon = True
            self.server_thread.start()
            time.sleep(2)  # Wait for server to start
            logger.info("Flask server started")

    def _save_optimized_screenshot(self, element, filename: Path, format='JPEG') -> None:
        """Take and save an optimized screenshot"""
        # Get screenshot as PNG bytes
        png_data = element.screenshot_as_png
        
        # Open with Pillow
        with Image.open(io.BytesIO(png_data)) as img:
            # Convert to grayscale (since we're dealing with text)
            img = img.convert('L')
            
            # Resize if needed
            if img.size != self.image_size:
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Save with compression
            if format.upper() == 'JPEG':
                img.save(filename, format=format, quality=self.image_quality, optimize=True)
            elif format.upper() == 'PNG':
                img.save(filename, format=format, optimize=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
            

   
    def _capture_font_screenshots(self, font: str, num_samples: int = 10) -> None:
        """Capture screenshots for a single font with improved annotation coverage"""
        driver = None
        try:
            driver = self._setup_webdriver()
            font_dir = self.output_dir / font.lower().replace(' ', '_')
            font_dir.mkdir(parents=True, exist_ok=True)
            
            # Create directories
            images_dir = font_dir / "images"
            annotations_dir = font_dir / "annotations"
            images_dir.mkdir(exist_ok=True)
            annotations_dir.mkdir(exist_ok=True)
            
            url = f"http://localhost:{self.port}/font/{font.replace(' ', '%20')}"
            logger.info(f"Loading URL: {url}")
            driver.get(url)
            
            # Wait for elements and fonts to load
            wait = WebDriverWait(driver, 10, 0.1)
            container = wait.until(
                EC.presence_of_element_located((By.ID, 'container'))
            )
            text_block = wait.until(
                EC.presence_of_element_located((By.ID, 'text-block'))
            )
            
            # Wait additional time for fonts to load
            time.sleep(1)
            
            # Get total height of text content
            total_height = driver.execute_script(
                "return document.getElementById('text-block').scrollHeight"
            )
            
            # Calculate visible area and scroll requirements
            viewport_height = self.image_size[1]
            content_height = total_height
            
            if content_height <= viewport_height:
                logger.warning(f"Content for font '{font}' fits in one screen, will generate fewer samples")
                # Even if content fits, we can still generate multiple samples with different parts visible
                scroll_step = viewport_height / (num_samples + 1)  # Smaller steps to show different parts
            else:
                # Calculate how much we can scroll
                scrollable_height = content_height - viewport_height
                logger.info(f"Content height: {content_height}px, Viewport: {viewport_height}px, Scrollable: {scrollable_height}px")
                scroll_step = scrollable_height / (num_samples - 1) if num_samples > 1 else 0
            
            # Track which samples succeeded
            successful_samples = 0
            max_attempts = num_samples * 2  # Allow extra attempts if needed
            attempt = 0
            
            while successful_samples < num_samples and attempt < max_attempts:
                try:
                    # Calculate scroll position with some jitter to avoid duplicates
                    base_position = (attempt % num_samples) * scroll_step
                    jitter = random.randint(-10, 10) if attempt >= num_samples else 0
                    scroll_position = max(0, int(base_position + jitter))
                    
                    # Apply scroll with translateY transform
                    driver.execute_script(
                        f"document.getElementById('text-block').style.transform = 'translateY(-{scroll_position}px)'"
                    )
                    
                    # Allow time for rendering after scroll
                    time.sleep(0.3)
                    
                    # Take screenshot
                    sample_idx = successful_samples  # Use successful count as index
                    filename = images_dir / f"sample_{sample_idx:04d}.jpg"
                    
                    # IMPORTANT: Make sure any debug boxes are removed before taking screenshot
                    driver.execute_script(
                        "document.querySelectorAll('.debug-box').forEach(box => box.remove())"
                    )
                    
                    self._save_optimized_screenshot(container, filename)
                    
                    # Collect character positions
                    char_positions = driver.execute_script("return window.getCharPositions()")
                    
                    # CHANGED: Accept images with any number of characters, but log warnings
                    if not char_positions:
                        logger.warning(f"No characters detected for {font} sample {sample_idx}, but keeping the image")
                    elif len(char_positions) < 5:
                        logger.warning(f"Only {len(char_positions)} characters detected for {font} sample {sample_idx}")
                    
                    # Save annotation even with few characters
                    annotation_file = annotations_dir / f"sample_{sample_idx:04d}.json"
                    with open(annotation_file, 'w') as f:
                        json.dump({
                            'font': font,
                            'image': filename.name,
                            'image_width': self.image_size[0],
                            'image_height': self.image_size[1],
                            'scroll_position': scroll_position,
                            'characters': char_positions if char_positions else []
                        }, f, indent=2)
                    
                    logger.info(f"Generated sample {sample_idx+1}/{num_samples} for font {font} with {len(char_positions) if char_positions else 0} characters")
                    
                    # Optional: Debug the first sample to verify positioning
                    if sample_idx == 0 and logger.level <= logging.DEBUG:
                        driver.execute_script("return window.debugCharBoxes()")
                        debug_screenshot = annotations_dir / f"debug_sample_{sample_idx:04d}.jpg"
                        self._save_optimized_screenshot(container, debug_screenshot)
                        logger.debug(f"Saved debug screenshot for {font}")
                    
                    successful_samples += 1
                    
                except Exception as e:
                    logger.error(f"Error processing attempt {attempt} for font {font}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                attempt += 1
                
            if successful_samples < num_samples:
                logger.warning(f"Only generated {successful_samples}/{num_samples} samples for font {font}")
                    
        finally:
            if driver:
                driver.quit()
    def generate_dataset(self) -> None:
        logger.info("Starting dataset generation")
        self.start_flask()
        
        for font in self.fonts:
            try:
                logger.info(f"Processing font: {font}")
                self._capture_font_screenshots(font, num_samples=self.num_samples_per_font)
            except Exception as e:
                logger.error(f"Failed to process font {font}: {e}")
                continue
        
        logger.info("Dataset generation complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', default='data_generation/lorem_ipsum.txt', help='Text to render in the various fonts')
    parser.add_argument('--font_file', default='data_generation/fonts.txt', help='File containing list of fonts to render')
    parser.add_argument('--output_dir', default='font-images3', help='Directory to save rendered images')
    parser.add_argument('--samples_per_class', default=10, type=int, help='Number of images per font')
    parser.add_argument('--image_resolution', default=256, type=int)
    parser.add_argument('--image_quality', default=10, type=int, help='JPEG quality (0-100)')
    parser.add_argument('--port', default=5100, type=int)
    parser.add_argument('--font_size', default=16, type=int)
    parser.add_argument('--line_height', default=1.5, type=float)
    args = parser.parse_args()

    # suggest new arguments copilot!
    renderer = FontRenderer(
        fonts_file=args.font_file,
        text_file=args.text_file,
        output_dir=args.output_dir,
        template_dir='templates',
        port=args.port,
        image_size=(args.image_resolution, args.image_resolution),  # Smaller size
        image_quality=args.image_quality,
        num_samples_per_font=args.samples_per_class,
        font_size=args.font_size,
        line_height=args.line_height
    )
    
    try:
        renderer.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")

if __name__ == "__main__":
    main()