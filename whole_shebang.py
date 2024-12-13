from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import base64
from PIL import Image
import io
from typing import List
import os

def load_fonts(filename: str) -> List[str]:
    """Load font names from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def get_google_font_url(font_name: str) -> str:
    """Generate Google Fonts URL for a given font"""
    # Replace spaces with + for URL formatting
    formatted_name = font_name.replace(' ', '+')
    return f"https://fonts.googleapis.com/css2?family={formatted_name}&display=swap"

def setup_webdriver():
    """Configure headless Chrome for font rendering"""
    print("Setting up headless Chrome...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=512,512")  # Make this a power of 2 for ML
    chrome_options.add_argument("--no-sandbox")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("Webdriver setup complete")
    return driver

def render_font_to_image(driver, html: str, output_path: Path, font_name: str):
    """Render HTML with font to image and save it"""
    try:
        print(f"\nProcessing {font_name}:")
        
        abs_output_path = output_path.absolute()
        abs_output_path.mkdir(parents=True, exist_ok=True)
        
        html_b64 = base64.b64encode(html.encode()).decode()
        driver.get(f"data:text/html;base64,{html_b64}")
        
        print("Waiting for font to load...")
        # Updated to use text-block class from your template
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "text-block"))
        )
        time.sleep(1)  # Wait for font to render
        
        screenshot = element.screenshot_as_png
        image = Image.open(io.BytesIO(screenshot))
        
        save_path = abs_output_path / f"{font_name.replace(' ', '_')}.png"
        print(f"Saving to: {save_path}")
        image.save(save_path)
        
        if save_path.exists():
            file_size = os.path.getsize(save_path)
            print(f"✓ Image saved successfully ({file_size} bytes)")
            return True
        else:
            print("✗ Image file not found after saving")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {font_name}: {str(e)}")
        return False

def generate_dataset(font_names: List[str], template_content: str, sample_text: str, output_dir: Path):
    """Generate the complete image dataset"""
    print(f"\nOutput directory: {output_dir.absolute()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    driver = setup_webdriver()
    successful_fonts = []
    
    try:
        for font_name in font_names:
            print(f"\n{'='*50}")
            print(f"Processing font: {font_name}")
            
            # Create a font context that matches your template
            font_context = {
                'name': font_name,
                'url': get_google_font_url(font_name)
            }
            
            # Generate HTML for this font
            html = template_content
            html = html.replace("{{ font.name }}", font_context['name'])
            html = html.replace("{{ font.url }}", font_context['url'])
            html = html.replace("{{ text }}", sample_text)
            
            if render_font_to_image(driver, html, output_dir, font_name):
                successful_fonts.append(font_name)
                print(f"Added {font_name} to successful fonts list")
    
    finally:
        driver.quit()
    
    success_path = output_dir / 'successful_fonts.txt'
    with open(success_path, 'w') as f:
        for font in successful_fonts:
            f.write(f"{font}\n")
    
    return successful_fonts

def main():
    print("Starting font dataset generation...")
    
    template_path = Path('templates/single_font.html')
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found at: {template_path.absolute()}"
        )
    
    data_dir = Path('dataset')
    
    font_names = load_fonts('fonts.txt')
    print(f"Loaded {len(font_names)} fonts: {font_names}")
    
    with open(template_path) as f:
        template_content = f.read()
    
    sample_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789
    """
    
    successful_fonts = generate_dataset(font_names, template_content, sample_text, data_dir)
    
    print(f"\nDataset generation complete!")
    print(f"Successfully rendered: {len(successful_fonts)} fonts")
    print(f"Failed to render: {len(font_names) - len(successful_fonts)} fonts")
    print(f"Images saved in: {data_dir.absolute()}")
    
    if successful_fonts:
        print("\nSuccessful fonts:")
        for font in successful_fonts:
            print(f"- {font}")

if __name__ == "__main__":
    main()