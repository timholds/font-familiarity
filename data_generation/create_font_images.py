import logging
import time
import os
from PIL import Image
import io
import json
import socket
import argparse
import random
import uuid

from pathlib import Path
from typing import List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from flask import Flask, render_template
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.service import Service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_available_port(start_port=5100, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available
                return port
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

# Custom expected conditions
class style_contains:
    """Wait for CSS style to contain specific value"""
    def __init__(self, locator, style_property, value):
        self.locator = locator
        self.style_property = style_property
        self.value = value

    def __call__(self, driver):
        try:
            element = driver.find_element(*self.locator)
            return self.value in element.value_of_css_property(self.style_property)
        except (StaleElementReferenceException, NoSuchElementException):
            return False


class javascript_returns_true:
    """Wait for JavaScript expression to return True"""
    def __init__(self, script):
        self.script = script

    def __call__(self, driver):
        return driver.execute_script(f"return ({self.script});")

def calculate_text_multiplier(text: str, font_size: int, line_height: float, 
                            container_height: int, samples_per_class: int) -> int:
    line_height_px = font_size * line_height
    lines_per_container = container_height / line_height_px
    chars_per_line = container_height / (font_size * 0.6)
    total_lines_needed = lines_per_container * (samples_per_class + 1)
    total_chars_needed = total_lines_needed * chars_per_line
    multiplier = int(total_chars_needed / len(text)) + 1
    return max(1, multiplier)

def prepare_text_content(filename: str, **kwargs) -> str:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        multiplier = calculate_text_multiplier(text, **kwargs)
        return (text * multiplier)[:10000]  # Cap text length
    except Exception as e:
        logger.error(f"Text preparation failed: {e}")
        raise

@dataclass
class FontConfig:
    name: str
    output_path: Path
    image_width: int = 512
    image_height: int = 512
    font_size: int = 24
    font_weight: int = 400      
    letter_spacing: str = 0
    line_height: float = 1.5   
    font_style: str = "normal"  
    text_color: str = "#000000"  
    bg_color: str = "#FFFFFF"   
    samples_per_font: int = 10

class TextAugmentation:
    """Generates continuous text augmentation parameters for dataset diversity."""
    
    def __init__(self, 
                 font_size_range=(16, 36),
                 weight_primary_modes=[400, 700],  # Common font weights
                 weight_primary_prob=0.7,          # Probability of using common weights
                 letter_spacing_range=(-0.1, 0.4), # em units
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
        """Sample a font weight using a mixture model approach.
        
        This uses a bimodal distribution favoring common weights (400, 700)
        while still occasionally sampling from the full range for robustness.
        """
        if random.random() < self.weight_primary_prob:
            # Sample from primary modes (regular or bold)
            return random.choice(self.weight_primary_modes)
        else:
            # Sample from full range for diversity
            return random.choice(self.valid_weights)
    
    def sample_letter_spacing(self):
        """Sample a letter spacing value as a continuous parameter.
        
        Returns a CSS value like '0.03em' or '-0.02em'.
        """
        spacing = random.uniform(*self.letter_spacing_range)
        return f"{spacing:.2f}em"
    
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
            #samples_per_font=1
        )


def process_font_config(font_config, **kwargs):
    """Process a font with specified configuration parameters."""
    try:
        logger.info(f"Starting worker for font config: {font_config.name} (size: {font_config.font_size}, weight: {font_config.font_weight})")
        
        # Update kwargs to include font config parameters
        font_worker_kwargs = {
            'font': font_config.name,
            'port': kwargs['port'],
            'output_dir': kwargs['output_dir'],
            'num_samples': font_config.samples_per_font,
            'detection_mode': kwargs['detection_mode'],
            'image_size': kwargs['image_size'],
            'image_quality': kwargs['image_quality'],
            'font_size': font_config.font_size,
            'line_height': font_config.line_height,
            'font_weight': font_config.font_weight,
            'letter_spacing': font_config.letter_spacing,
            'font_style': font_config.font_style,
            'text_color': font_config.text_color,
            'bg_color': font_config.bg_color,
            'output_subdir': font_config.name.lower().replace(' ', '_'), 
            'backgrounds_dir': kwargs.get('backgrounds_dir', None),
            'background_probability': kwargs.get('background_probability', 0.0),
            'sample_id': kwargs.get('sample_id', 0),
            'config_id': kwargs.get('config_id'),  
        }
        
        worker = FontRendererWorker(**font_worker_kwargs)
        worker.process()
        
        logger.info(f"Completed processing for font config: {font_config.name} (size: {font_config.font_size})")
        return True
    except Exception as e:
        import traceback
        error_msg = f"Error processing font {font_config.name}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise RuntimeError(f"Processing font config '{font_config.name}' failed: {e}")
    
class FontRenderer:
    def __init__(self, fonts_file: str = 'fonts.txt', text_file: str = 'lorem_ipsum.txt',
                 output_dir: str = 'font-images', template_dir: str = 'templates',
                 port: int = 5100, image_size: tuple = (256, 256), image_quality: int = 80,
                 num_samples_per_font: int = 10, font_size: int = 24, line_height: float = 1.5,
                 detection_mode: bool = False, backgrounds_dir: str = None,
                 background_blend_mode: str = 'overlay', background_probability: float = 0.5):
        
        self.font_size = font_size
        self.line_height = line_height
        self.image_size = image_size
        self.image_quality = image_quality
        self.num_samples_per_font = num_samples_per_font
        self.port = port
        self.detection_mode = detection_mode
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.flask_app = None
        self.server_thread = None

        self.backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None
        self.background_blend_mode = background_blend_mode  # 'overlay', 'multiply', etc.
        self.background_probability = background_probability  # Chance of using a background (0-1)
        # self.text_contrast = text_contrast  # Enhancing text contrast before overlaying

        self.config_registry = {}
        self.port = find_available_port(port)
        if self.port != port:
            logger.info(f"Port {port} was in use. Using port {self.port} instead.")

        self.fonts = self._load_fonts(fonts_file)
        self.text = prepare_text_content(
            text_file,
            font_size=self.font_size,
            line_height=self.line_height,
            container_height=self.image_size[1],
            samples_per_class=self.num_samples_per_font
        )

        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_fonts(self, filename: str) -> List[str]:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Fonts file not found: {filename}")
            raise

    def start_flask(self):
        app = Flask(__name__, template_folder=str(self.template_dir.absolute()))
        self.flask_app = app

        @app.route('/health')
        def health_check():
            return "OK"

        @app.route('/font/<font_name>/<config_id>')
        def render_font(font_name, config_id):
            font_config = self.config_registry.get(config_id)
            if not font_config:    
                font_config = FontConfig(
                    name=font_name,
                    output_path=self.output_dir / font_name.lower().replace(' ', '_'),
                    image_width=self.image_size[0],
                    image_height=self.image_size[1],
                    font_size=self.font_size,
                    samples_per_font=self.num_samples_per_font
            )

            template = 'single_font_detection.html' if self.detection_mode else 'single_font.html'
            return render_template(
                template,
                font=font_config,
                text=self.text,
                font_size=self.font_size,
                line_height=self.line_height
            )

        from threading import Thread
        from werkzeug.serving import make_server

        class ServerThread(Thread):
            def __init__(self, app, port):
                Thread.__init__(self)
                self.server = make_server('0.0.0.0', port, app, threaded=True)
                self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.daemon = True
                
            def run(self):
                self.server.serve_forever()
                
            def shutdown(self):
                self.server.shutdown()
    
        server = ServerThread(app, self.port)
        server.start()
        self.server_thread = server
        time.sleep(1)  # Wait for server to start


    def generate_dataset(self):
        self.start_flask()
        logger.info(f"Starting parallel processing with {len(self.fonts)} fonts")
        augmenter = TextAugmentation(
            font_size_range=(16, 36),
            weight_primary_modes=[400, 700],
            weight_primary_prob=0.7,
            letter_spacing_range=(-0.05, 0.1),
            line_height_range=(1.1, 1.9)
        )

        versions_per_font = self.num_samples_per_font
        all_font_configs = []
        for font in self.fonts:
            # Generate a unique configuration for each sample
            for sample_id in range(versions_per_font):
                config = augmenter.generate_config(
                    font, 
                    str(self.output_dir),
                    image_width=self.image_size[0],
                    image_height=self.image_size[1],
                    samples_per_font=1,
                    sample_id=sample_id,
                )
                all_font_configs.append(config)
        logger.info(f"Generated {len(all_font_configs)} total font configurations")
      
        for config in all_font_configs:
            config_id = f"{config.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            self.config_registry[config_id] = config

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for config_id, config in self.config_registry.items():
                futures.append(executor.submit(
                    process_font_config,
                    font_config=config,
                    port=self.port,
                    output_dir=str(self.output_dir),
                    detection_mode=self.detection_mode,
                    image_size=self.image_size,
                    image_quality=self.image_quality,
                    config_id=config_id,
                    sample_id=getattr(config, 'sample_id', 0)  # Safely get sample_id if it exists
                ))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Font processing failed: {e}")

        self._create_dataset_description()

    def _create_dataset_description(self):
        description_path = self.output_dir / "dataset_info.txt"
        with open(description_path, 'w') as f:
            f.write(f"Dataset generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Fonts processed: {len(self.fonts)}\n")
            f.write(f"Samples per font: {self.num_samples_per_font}\n")

class FontRendererWorker:
    def __init__(self, **kwargs):
        self.font = kwargs['font']
        self.port = kwargs['port']
        self.num_samples = kwargs['num_samples']
        self.detection_mode = kwargs['detection_mode']
        self.image_size = kwargs['image_size']
        self.image_quality = kwargs['image_quality']
        self.font_size = kwargs['font_size']
        self.line_height = kwargs['line_height']
        self.font_weight = kwargs.get('font_weight', 400)
        self.letter_spacing = kwargs.get('letter_spacing', 'normal')
        self.font_style = kwargs.get('font_style', 'normal')
        self.text_color = kwargs.get('text_color', '#000000')
        self.bg_color = kwargs.get('bg_color', '#FFFFFF')
        
        # self.output_dir = Path(kwargs['output_dir'])
        # self.output_subdir = kwargs.get('output_subdir', None)  # For unique subdirectory
        base_output_dir = Path(kwargs['output_dir']) 
        self.output_subdir = kwargs.get('output_subdir', self.font.lower().replace(' ', '_'))
        self.font_dir = base_output_dir / self.output_subdir
        
        self.sample_id = kwargs.get('sample_id', 0)
        self.config_id = kwargs.get('config_id')
        self.output_subdir = kwargs.get('output_subdir', None)
    
        
        # Background settings
        self.backgrounds_dir = kwargs.get('backgrounds_dir', None)
        self.background_probability = kwargs.get('background_probability', 0.0)


        self.driver = self._setup_driver()

    def _setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument(f'--window-size={self.image_size[0]},{self.image_size[1]}')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        service = Service()
        return webdriver.Chrome(service=service, options=chrome_options)

    def process(self):
        try:
            if self.detection_mode:
                self._capture_with_detection()
            else:
                self._capture_standard()
        finally:
            self.driver.quit()

    def _capture_standard(self):
        font_dir = self.font_dir
        annotations_dir = font_dir / "annotations"

        font_dir.mkdir(parents=True, exist_ok=True)

        self.driver.get(f"http://localhost:{self.port}/font/{self.font.replace(' ', '%20')}/{self.config_id}")

        # Add a small delay to ensure the page is fully loaded
        time.sleep(1)

        container = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'container'))
        )
        text_block = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'text-block'))
        )

        total_height = self.driver.execute_script(
            "return arguments[0].scrollHeight", text_block
        )

        # Ensure we have enough content to scroll
        visible_height = self.image_size[1]
        if total_height <= visible_height:
            logger.warning(f"Text content for font {self.font} is too short for {self.num_samples} samples")
            # Just take one screenshot if content is too short
            filename = f"sample_{self.sample_id:04d}_0000.jpg"  # Fixed: use 0 when no scroll
            self._save_screenshot(container, font_dir / filename)
            return
            
        scroll_step = (total_height - visible_height) / max(1, self.num_samples - 1)

        for i in range(self.num_samples):
            scroll_pos = int(i * scroll_step)
            
            # Apply scroll transform
            self.driver.execute_script(
                f"arguments[0].style.transform = 'translateY(-{scroll_pos}px)';",
                text_block
            )
            
            # Small delay
            time.sleep(0.1)
            
            # Verify via JavaScript
            scroll_complete = self.driver.execute_script(
                """
                var style = window.getComputedStyle(arguments[0]);
                var transform = style.getPropertyValue('transform');
                return transform !== 'none' && transform !== '';
                """, 
                text_block
            )
            
            if not scroll_complete:
                logger.warning(f"Scroll verification failed for font {self.font} at position {scroll_pos}")
            
            filename = f"sample_{i:04d}.jpg"
            self._save_screenshot(container, font_dir / filename)
            
    def _capture_with_detection(self):
        font_dir = self.output_dir / self.font.lower().replace(' ', '_')
        annotations_dir = font_dir / "annotations"
        font_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)

        WebDriverWait(self.driver, 10).until(
            javascript_returns_true("typeof window.detectionData !== 'undefined' && window.detectionData.characters.length > 0")
        )

        self.driver.get(f"http://localhost:{self.port}/font/{self.font.replace(' ', '%20')}")
        container = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'container'))
        )
        text_block = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, 'text-block'))
        )

        total_height = self.driver.execute_script(
            "return arguments[0].scrollHeight", text_block
        )
        scroll_step = (total_height - self.image_size[1]) / (self.num_samples - 1)

        for i in range(self.num_samples):
            scroll_pos = int(i * scroll_step)
            self.driver.execute_script(
                f"arguments[0].style.transform = 'translateY(-{scroll_pos}px)';",
                text_block
            )
            WebDriverWait(self.driver, 2).until(
                style_contains((By.ID, 'text-block'), "transform", f"translateY(-{scroll_pos}px)")
            )
            self.driver.execute_script("window.measureCharacterPositions();")
            WebDriverWait(self.driver, 5).until(
                javascript_returns_true("window.detectionData?.characters?.length > 0")
            )
            filename = f"sample_{i:04d}.jpg"
            self._save_screenshot(container, font_dir / filename)
            self._save_annotations(annotations_dir, i, self.sample_id)

    def _save_screenshot(self, element, path):
        png_data = element.screenshot_as_png
        with Image.open(io.BytesIO(png_data)) as text_img:
            if text_img.size != self.image_size:
                text_img = text_img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            if self.backgrounds_dir is None or random.random() > self.background_probability:
                text_img.save(path, quality=self.image_quality, optimize=True)
                return
            # Process the text image to create a mask (white text on transparent background)
            # Convert to RGBA if not already
            if text_img.mode != 'RGBA':
                text_img = text_img.convert('RGBA')
                
            # Enhance contrast if specified
            # if self.text_contrast > 1:
            #     from PIL import ImageEnhance
            #     enhancer = ImageEnhance.Contrast(text_img)
            #     text_img = enhancer.enhance(self.text_contrast)
            
            # Create a mask where white pixels (text) are opaque and background is transparent
            r, g, b, a = text_img.split()
            # Create mask where white (255,255,255) becomes opaque (255) and everything else transparent
            mask = Image.eval(r, lambda px: 255 if px > 240 else 0)
            
            # Choose a random background
            background_files = list(self.backgrounds_dir.glob('*.jpg')) + \
                                list(self.backgrounds_dir.glob('*.png'))
            
            if not background_files:
                logger.warning(f"No background images found in {self.backgrounds_dir}")
                text_img.save(path, quality=self.image_quality, optimize=True)
                return
                
            background_file = random.choice(background_files)
            
            try:
                with Image.open(background_file) as bg:
                    # Resize background to match our dimensions
                    bg = bg.resize(self.image_size, Image.Resampling.LANCZOS)
                    
                    # Create a new image with the background
                    result = bg.copy()
                    
                    # Composite the text onto the background using the mask
                    result.paste(text_img, (0, 0), mask)
                    
                    # Save the final image
                    result.save(path, quality=self.image_quality, optimize=True)
            except Exception as e:
                logger.error(f"Error processing background image: {e}")
                # Fallback to original image
                text_img.save(path, quality=self.image_quality, optimize=True)

    def _save_annotations(self, annotations_dir, index, sample_id):
        detection_data = self.driver.execute_script("return window.detectionData;")

        if not detection_data or 'characters' not in detection_data:
            logger.warning(f"No detection data available for {self.font} sample {index}")
            return

        yolo_path = annotations_dir / f"sample_{index:04d}.txt"
        json_path = annotations_dir / f"sample_{index:04d}.json"
        visible_chars = [
            char for char in detection_data['characters']
            if (0 <= char['y'] <= self.image_size[1] - char['height'] - 5 and 
                0 <= char['x'] <= self.image_size[0] - char['width'] - 5 and
                char['y'] + char['height'] > 5 and
                char['x'] + char['width'] > 5 and
                char['width'] > 3 and char['height'] > 3)
        ]

        # Generate annotations in YOLO format
        annotation_lines = []
        char_mapping = {}  # For label file

        for char in visible_chars:
            # Skip if character is empty or whitespace
            if not char['char'] or char['char'].isspace():
                continue
                
            # Convert to YOLO format: class x_center y_center width height
            char_code = ord(char['char']) 
            char_class = char_code % 256  # Simple mapping
            
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
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(annotation_lines))

        # Also save raw JSON data for reference
        with open(json_path, 'w') as f:
            json.dump({
                'image': f"sample_{index:04d}.jpg",
                'font': self.font,
                'characters': visible_chars
            }, f, indent=2)

        # Update character mapping file
        mapping_path = annotations_dir / "classes.txt"
        with open(mapping_path, 'w') as f:
            for class_id, char in sorted(char_mapping.items()):
                f.write(f"{class_id} {char}\n")

def process_font_worker(**kwargs):
    try:
        logger.info(f"Starting worker for font: {kwargs['font']}")
        worker = FontRendererWorker(**kwargs)
        worker.process()
        logger.info(f"Completed processing for font: {kwargs['font']}")
        return True
    except Exception as e:
        import traceback
        error_msg = f"Error processing font {kwargs['font']}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        # Re-raise with more detailed message to be caught by the main process
        raise RuntimeError(f"Processing font '{kwargs['font']}' failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', default='data_generation/lorem_ipsum.txt')
    parser.add_argument('--font_file', default='data_generation/fonts.txt')
    parser.add_argument('--output_dir', default='font-images')
    parser.add_argument('--samples_per_class', type=int, default=10)
    parser.add_argument('--image_resolution', type=int, default=256)
    parser.add_argument('--image_quality', type=int, default=80)
    parser.add_argument('--port', type=int, default=5100)
    parser.add_argument('--font_size', type=int, default=24)
    parser.add_argument('--line_height', type=float, default=1.5)
    parser.add_argument('--detection_mode', action='store_true')
    parser.add_argument('--backgrounds_dir', default=None, help='Directory containing background images')
    parser.add_argument('--background_probability', type=float, default=1.0,
                    help='Probability of using a background (0-1)')
    # parser.add_argument('--text_contrast', type=float, default=1.2,
    #                 help='Text contrast enhancement factor')

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

    try:
        renderer.generate_dataset()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()