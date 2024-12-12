from flask import Flask, render_template_string
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import json
import os
import time
from lorem import get_paragraph
breakpoint()

# Flask app to render fonts
app = Flask(__name__)

@app.route('/')
def home():
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Font Dataset Generator</title>
        <style>
            .text-block {
                padding: 20px;
                margin: 20px;
                background: white;
                min-height: 200px;
            }
        </style>
    </head>
    <body style="background-color: #f0f0f0;">
        {% for font in fonts %}
            <link href="https://fonts.googleapis.com/css2?family={{ font.family|replace(' ', '+') }}&display=swap" rel="stylesheet">
            <div id="{{ font.family|replace(' ', '_') }}" class="text-block" style="font-family: '{{ font.family }}', {{ font.category }};">
                <h3>{{ font.family }}</h3>
                {{ text }}
            </div>
        {% endfor %}
    </body>
    </html>
    '''
    return render_template_string(html_template, fonts=get_fonts()[:10], text=generate_text())

def get_fonts(api_key=None):
    """Fetch fonts from Google Fonts API"""
    if api_key:
        url = f'https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}'
    else:
        # Fallback to local sample if no API key
        return [
            {'family': 'Roboto', 'category': 'sans-serif'},
            {'family': 'Open Sans', 'category': 'sans-serif'},
            {'family': 'Lato', 'category': 'sans-serif'},
            {'family': 'Montserrat', 'category': 'sans-serif'},
            {'family': 'Playfair Display', 'category': 'serif'}
        ]
    
    response = requests.get(url)
    return response.json()['items']

def generate_text(paragraphs=3):
    """Generate Lorem Ipsum text"""
    return '\n'.join([get_paragraph() for _ in range(paragraphs)])

def setup_webdriver():
    """Configure Chrome WebDriver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=chrome_options)

def capture_font_screenshots(output_dir='font_dataset', batch_size=10):
    """Capture screenshots for each font rendered on the page"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    driver = setup_webdriver()
    fonts = get_fonts()
    
    # Process fonts in batches
    for i in range(0, len(fonts), batch_size):
        batch = fonts[i:i+batch_size]
        
        # Start Flask server for this batch
        driver.get('http://localhost:5000')
        
        # Wait for fonts to load
        time.sleep(2)
        
        # Capture screenshots for each font in the batch
        for font in batch:
            font_id = font['family'].replace(' ', '_')
            element = driver.find_element(By.ID, font_id)
            
            # Scroll element into view
            driver.execute_script("arguments[0].scrollIntoView();", element)
            time.sleep(0.5)  # Allow time for any animations
            
            # Save screenshot
            element.screenshot(f'{output_dir}/{font_id}.png')
            
            # Save metadata
            with open(f'{output_dir}/{font_id}.json', 'w') as f:
                json.dump(font, f)
    
    driver.quit()

if __name__ == '__main__':
    # Run Flask app in a separate thread
    from threading import Thread
    server = Thread(target=app.run, kwargs={'debug': False})
    server.daemon = True
    server.start()
    
    # Start capturing screenshots
    capture_font_screenshots()