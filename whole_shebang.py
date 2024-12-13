from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
from PIL import Image
import io
from typing import List
import traceback

def load_fonts(filename: str) -> List[str]:
    """Load font names from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def setup_webdriver():
    """Configure headless Chrome for font rendering"""
    print("Setting up headless Chrome...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    print("Webdriver setup complete")
    return driver

def render_font_to_image(driver, html: str, output_path: Path, font_name: str):
    """Render HTML with font to image and save it"""
    try:
        print(f"\nProcessing {font_name}:")
        print("1. Loading HTML...")
        html_b64 = base64.b64encode(html.encode()).decode()
        driver.get(f"data:text/html;base64,{html_b64}")
        
        print("2. Waiting for font to load...")
        # Wait for font to load
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "text-block"))
        )
        
        # Additional wait for font rendering
        time.sleep(1)  # Increased wait time
        
        print("3. Taking screenshot...")
        screenshot = element.screenshot_as_png
        
        print("4. Converting to PIL Image...")
        image = Image.open(io.BytesIO(screenshot))
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_path = output_path / f"{font_name.replace(' ', '_')}.png"
        print(f"5. Saving to {save_path}")
        image.save(save_path)
        
        # Verify the file was saved
        if save_path.exists():
            print("✓ Image saved successfully")
            return True
        else:
            print("✗ Image file not found after saving")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {font_name}:")
        print(traceback.format_exc())
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
            
            # Generate HTML for this font
            html = template_content.replace(
                "{{font_name}}", font_name
            ).replace(
                "{{text}}", sample_text
            )
            
            # Render and save image
            if render_font_to_image(driver, html, output_dir, font_name):
                successful_fonts.append(font_name)
                print(f"Added {font_name} to successful fonts list")
    
    finally:
        driver.quit()
    
    # Save list of successful fonts
    success_path = output_dir / 'successful_fonts.txt'
    print(f"\nSaving successful fonts list to {success_path}")
    with open(success_path, 'w') as f:
        for font in successful_fonts:
            f.write(f"{font}\n")
    
    return successful_fonts

def main():
    print("Starting font dataset generation...")
    
    # Ensure we're using the existing templates
    template_path = Path('templates/single_font.html')
    if not template_path.exists():
        raise FileNotFoundError(
            "single_font.html not found! Make sure it's in the same directory as this script."
        )
    
    # Setup output directory
    data_dir = Path('dataset')
    print(f"Output directory will be: {data_dir.absolute()}")
    
    # Load fonts and template
    font_names = load_fonts('fonts.txt')
    print(f"Loaded {len(font_names)} fonts: {font_names}")
    
    with open(template_path) as f:
        template_content = f.read()
    print("Loaded template file")
    
    # Sample text
    sample_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789
    """
    
    # Generate dataset
    successful_fonts = generate_dataset(font_names, template_content, sample_text, data_dir)
    
    print(f"\nDataset generation complete!")
    print(f"Successfully rendered: {len(successful_fonts)} fonts")
    print(f"Failed to render: {len(font_names) - len(successful_fonts)} fonts")
    print(f"Images saved in: {data_dir.absolute()}")
    
    if successful_fonts:
        print("\nSuccessful fonts:")
        for font in successful_fonts:
            print(f"- {font}")
    else:
        print("\nNo fonts were successfully rendered!")
        
    if not successful_fonts:
        print("\nPossible issues to check:")
        print("1. Is Chrome and ChromeDriver installed?")
        print("2. Check the HTML template - does it have the 'font-sample' class?")
        print("3. Are the font names in fonts.txt exactly as they appear on Google Fonts?")

if __name__ == "__main__":
    main()