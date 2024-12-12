# app.py
from flask import Flask, render_template
import os

app = Flask(__name__)

def load_text_file(filename):
    """Load text content from a file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def load_fonts(filename):
    """Load font names from a file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

@app.route('/')
def home():
    # Load the Lorem Ipsum text
    lorem_text = load_text_file('lorem_ipsum.txt')
    
    # Load the list of fonts
    fonts = load_fonts('fonts.txt')
    
    # Create font data for template
    font_data = []
    for font in fonts:
        # Convert font name to format needed for Google Fonts URL
        url_font_name = font.replace(' ', '+')
        font_data.append({
            'name': font,
            'url': f'https://fonts.googleapis.com/css2?family={url_font_name}&display=swap'
        })
    
    return render_template(
        'base.html',
        fonts=font_data,
        text=lorem_text
    )

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)