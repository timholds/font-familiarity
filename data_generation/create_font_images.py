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
                 image_size: tuple = (256, 256),  
                 image_quality: int = 80, # JPEG quality (0-100))
                 num_samples_per_font: int = 10,
                 font_size: int = 24,
                 line_height: float = 1.5,
                 detection_mode: bool = False):  
    
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
        self.detection_mode = detection_mode
        
        self.fonts = self._load_fonts(fonts_file)
    
        # Process text content with potential repetition for scrolling
        self.text = prepare_text_content(
            text_file,
            font_size=self.font_size,
            line_height=self.line_height,
            container_height=self.image_size[1],
            samples_per_class=self.num_samples_per_font
        )
            
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
            app = Flask(__name__, template_folder=str(self.template_dir.absolute()))
            self.flask_app = app
            
            # Store class instance as app config variable
            app.config['RENDERER'] = self
            
            @app.route('/font/<font_name>')
            def render_font(font_name):
                # Get renderer from app config
                renderer = app.config['RENDERER']
                
                font_config = FontConfig(
                    name=font_name,
                    output_path=renderer.output_dir / font_name.lower().replace(' ', '_'),
                    image_width=renderer.image_size[0],
                    image_height=renderer.image_size[1],
                    font_size=renderer.font_size,
                    samples_per_font=renderer.num_samples_per_font
                )
                
                if renderer.detection_mode:
                    # For detection mode with annotations
                    return render_template(
                        'single_font_detection.html',
                        font=font_config,
                        text=renderer.text,
                        font_size=renderer.font_size,
                        line_height=renderer.line_height
                    )
                else:
                    # For standard text mode
                    return render_template(
                        'single_font.html',
                        font=font_config,
                        text=renderer.text,
                        font_size=renderer.font_size,
                        line_height=renderer.line_height
                    )
            
            from threading import Thread
            self.server_thread = Thread(target=lambda: app.run(port=self.port, debug=False))
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
            # img = img.convert('L')
            
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
        """Capture screenshots for a single font with scrolling"""
        driver = None
        try:
            driver = self._setup_webdriver()
            font_dir = self.output_dir / font.lower().replace(' ', '_')
            font_dir.mkdir(parents=True, exist_ok=True)
            
            url = f"http://localhost:{self.port}/font/{font.replace(' ', '%20')}"
            logger.info(f"Loading URL: {url}")
            driver.get(url)
            
            wait = WebDriverWait(driver, 10, .1)
            container = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, 'container'))
            )
            text_block = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, 'text-block'))
            )
            
            # Get total height of text
            total_height = driver.execute_script(
                "return document.querySelector('.text-block').scrollHeight"
            )
            
            content_height = total_height - self.image_size[1]
            if content_height <= 0:
                raise ValueError(f"Not enough text content for font {font}")
                
            scroll_step = content_height / (num_samples - 1)
            
            for i in range(num_samples):
                try:
                    # Calculate scroll position
                    scroll_position = int(i * scroll_step)
                    
                    # Scroll to position
                    driver.execute_script(
                        f"document.querySelector('.text-block').style.transform = 'translateY(-{scroll_position}px)';"
                    )
                    time.sleep(0.1)
                    
                    # Take and save optimized screenshot
                    filename = font_dir / f"sample_{i:04d}.jpg"
                    self._save_optimized_screenshot(container, filename)
                    
                    if i % 100 == 0:
                        logger.info(f"Generated {i}/{num_samples} samples for font {font}")
                        
                except Exception as e:
                    logger.error(f"Error capturing screenshot {i} for font {font}: {e}")
                    raise
                    
        finally:
            if driver:
                driver.quit()
                
    def _capture_font_screenshots_with_detection(self, font: str, num_samples: int = 10) -> None:
        """Capture screenshots for a single font with character-level detection annotations"""
        driver = None
        try:
            driver = self._setup_webdriver()
            font_dir = self.output_dir / font.lower().replace(' ', '_')
            font_dir.mkdir(parents=True, exist_ok=True)
            
            # Create annotations directory
            annotations_dir = font_dir / "annotations"
            annotations_dir.mkdir(exist_ok=True)
            
            url = f"http://localhost:{self.port}/font/{font.replace(' ', '%20')}"
            logger.info(f"Loading URL: {url}")
            driver.get(url)
            
            wait = WebDriverWait(driver, 10, .5)
            container = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, 'container'))
            )
            
            # Wait for text block to be ready
            text_block = wait.until(
                EC.presence_of_element_located((By.ID, 'text-block'))
            )
            
            # Wait for initial character detection to complete
            time.sleep(1.0)  # Additional time for initial rendering
            
            # Get total height of text
            total_height = driver.execute_script(
                "return document.querySelector('#text-block').scrollHeight"
            )
            
            content_height = total_height - self.image_size[1]
            if content_height <= 0:
                raise ValueError(f"Not enough text content for font {font}")
                
            scroll_step = content_height / (num_samples - 1)
            
            for i in range(num_samples):
                try:
                    # Calculate scroll position
                    scroll_position = int(i * scroll_step)
                    
                    # Scroll to position
                    driver.execute_script(
                        f"document.querySelector('#text-block').style.transform = 'translateY(-{scroll_position}px)';"
                    )
                    time.sleep(0.5)  # Additional time for rendering and measuring
                    
                    # Trigger measurement after scrolling
                    driver.execute_script("window.measureCharacterPositions();")
                    time.sleep(0.5)
                    
                    # Get detection data from JavaScript
                    detection_data = driver.execute_script("return window.detectionData;")
                    
                    # Take and save optimized screenshot
                    image_filename = f"sample_{i:04d}.jpg"
                    image_path = font_dir / image_filename
                    self._save_optimized_screenshot(container, image_path)
                    
                    # Process detection data
                    if detection_data and 'characters' in detection_data:
                        # Filter to only characters visible in the current viewport
                        visible_chars = [
                            char for char in detection_data['characters']
                            if (0 <= char['y'] <= self.image_size[1] - char['height'] - 5 and 
                                0 <= char['x'] <= self.image_size[0] - char['width'] - 5 and
                                char['y'] + char['height'] > 5 and
                                char['x'] + char['width'] > 5 and
                                char['width'] > 3 and char['height'] > 3)  # Minimum size requirement
                        ]
                        
                        # Generate annotations in YOLO format
                        annotation_lines = []
                        char_mapping = {}  # For label file
                        
                        for char in visible_chars:
                            # Skip if character is empty or whitespace
                            if not char['char'] or char['char'].isspace():
                                continue
                                
                            # Convert to YOLO format: class x_center y_center width height
                            # (all normalized to 0-1)
                            char_code = ord(char['char']) 
                            char_class = char_code % 256  # Simple mapping for now
                            
                            # Add to mapping
                            char_mapping[char_class] = char['char']
                            
                            # Calculate normalized coordinates
                            x_center = (char['x'] + char['width']/2) / self.image_size[0]
                            y_center = (char['y'] + char['height']/2) / self.image_size[1]
                            width_norm = char['width'] / self.image_size[0]
                            height_norm = char['height'] / self.image_size[1]
                            
                            # Skip if bounding box is invalid
                            if width_norm <= 0 or height_norm <= 0:
                                continue
                            
                            # Ensure values are bounded between 0 and 1
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width_norm = max(0, min(1, width_norm))
                            height_norm = max(0, min(1, height_norm))
                            
                            line = f"{char_class} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                            annotation_lines.append(line)
                        
                        # Save YOLO annotations
                        annotation_path = annotations_dir / f"{image_filename.split('.')[0]}.txt"
                        with open(annotation_path, 'w') as f:
                            f.write('\n'.join(annotation_lines))
                        
                        # Also save raw JSON data for reference
                        json_path = annotations_dir / f"{image_filename.split('.')[0]}.json"
                        with open(json_path, 'w') as f:
                            json.dump({
                                'image': image_filename,
                                'font': font,
                                'scroll_position': scroll_position,
                                'characters': visible_chars
                            }, f, indent=2)
                        
                        # Update character mapping file
                        mapping_path = annotations_dir / "classes.txt"
                        with open(mapping_path, 'w') as f:
                            for class_id, char in sorted(char_mapping.items()):
                                f.write(f"{class_id} {char}\n")
                        
                        logger.info(f"Generated image {image_filename} with {len(visible_chars)} character annotations")
                    else:
                        logger.warning(f"No detection data available for image {i}")
                    
                    if i % 10 == 0:
                        logger.info(f"Generated {i+1}/{num_samples} samples for font {font}")
                        
                except Exception as e:
                    logger.error(f"Error capturing screenshot {i} for font {font}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
        finally:
            if driver:
                driver.quit()
                
        # Create a classes file for the entire font
        classes_path = font_dir / "classes.txt"
        
        # Compile all character codes seen in this font
        all_chars = set()
        
        # Look through all annotation files
        anno_files = list(annotations_dir.glob("*.txt"))
        for anno_file in anno_files:
            try:
                with open(anno_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_chars.add(int(parts[0]))
            except Exception as e:
                logger.error(f"Error reading annotation file {anno_file}: {e}")
        
        # Create a comprehensive mapping file
        with open(classes_path, 'w') as f:
            for class_id in sorted(all_chars):
                # Ensure we use the actual character, not just % 256
                char = chr(class_id)
                f.write(f"{class_id} {char}\n")
    def generate_dataset(self) -> None:
        logger.info("Starting dataset generation")
        self.start_flask()
        
        for font in self.fonts:
            try:
                logger.info(f"Processing font: {font}")
                
                if self.detection_mode:
                    # Use detection-specific capture method
                    self._capture_font_screenshots_with_detection(
                        font, 
                        num_samples=self.num_samples_per_font
                    )
                else:
                    # Use standard capture method
                    self._capture_font_screenshots(
                        font, 
                        num_samples=self.num_samples_per_font
                    )
                    
            except Exception as e:
                logger.error(f"Failed to process font {font}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info("Dataset generation complete")
        
        # Create dataset description file
        description_path = self.output_dir / "dataset_info.txt"
        with open(description_path, 'w') as f:
            f.write(f"Dataset generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of fonts: {len(self.fonts)}\n")
            f.write(f"Samples per font: {self.num_samples_per_font}\n")
            f.write(f"Image size: {self.image_size[0]}x{self.image_size[1]}\n")
            f.write(f"Font size: {self.font_size}px\n")
            
            if self.detection_mode:
                f.write(f"Mode: Detection with character annotations\n")
                f.write(f"Annotation format: YOLO\n")
            else:
                f.write(f"Mode: Full text\n")


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
    parser.add_argument('--detection_mode', action='store_true',
                      help='Generate images with character-level detection annotations')
    
    args = parser.parse_args()

    renderer = FontRenderer(
        fonts_file=args.font_file,
        text_file=args.text_file,
        output_dir=args.output_dir,
        template_dir='templates',
        port=args.port,
        image_size=(args.image_resolution, args.image_resolution),
        image_quality=args.image_quality,
        num_samples_per_font=args.samples_per_class,
        font_size=args.font_size,
        line_height=args.line_height,
        detection_mode=args.detection_mode
    )
    
    # Print mode information
    if args.detection_mode:
        logger.info("Running in detection mode: generating images with character annotations")
    else:
        logger.info("Running in standard mode: generating paragraph images")
    
    try:
        renderer.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}", exc_info=True)

if __name__ == "__main__":
    main()