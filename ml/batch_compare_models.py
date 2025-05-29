# batch_compare.py
from flask import Flask, render_template_string
import os
import torch
import numpy as np
from PIL import Image
import argparse
import base64
import io
import traceback
import logging
from datetime import datetime
import uuid
import torch.nn.functional as F
from char_model import CRAFTFontClassifier
import compare_models  # Import the existing compare_models module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HTML template for the batch report
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Font Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            padding: 20px;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        .image-container {
            margin-bottom: 40px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        .image-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .image-preview {
            max-width: 100%;
            max-height: 300px;
            display: block;
            margin: 0 auto 20px;
            border: 1px solid #ddd;
        }
        
        .comparison-container {
            display: flex;
            gap: 20px;
        }
        
        .model-column {
            flex: 1;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e1e4e8;
        }
        
        .model-header {
            background-color: #e9ecef;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
            font-weight: bold;
            border-left: 4px solid #5b9504;
        }
        
        .section-header h3 {
            margin: 0;
            padding: 10px 0;
            border-bottom: 1px solid #e1e4e8;
        }
        
        .result-item {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .font-name {
            font-weight: bold;
            font-size: 18px;
        }
        
        .score {
            font-weight: bold;
            color: #4285f4;
        }
        
        .comparison-container {
            display: flex;
            gap: 20px;
            align-items: stretch;
        }

        .model-column {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

            /* Container for the results sections */
        .results-section-container {
            display: flex;
            flex-direction: column;
            flex: 1;
        }

            /* Ensure each result item has a consistent height */
        .result-item {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            flex-direction: column;
            height: 260px; /* same height for all result items */
        }

        .font-sample {
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #e1e4e8;
            font-size: 24px;
            line-height: 1.5;
            overflow-wrap: break-word;
            flex: 1; /* Let it grow to fill available space */
            overflow: auto; /* Add scrollbar if text overflows */
        }

        .font-actions {
            margin-top: auto; /* Push to bottom of container */
            padding-top: 8px;
        }
        
        .copy-btn, .font-link {
            padding: 6px 12px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        
        .copy-btn {
            background-color: #eaeaea;
            border: 1px solid #ccc;
            color: #333;
            cursor: pointer;
        }
        
        .copy-btn:hover {
            background-color: #d5d5d5;
        }
        
        .font-link {
            background-color: #4285f4;
            color: white;
            border: none;
            text-decoration: none;
        }
        
        .font-link:hover {
            background-color: #3367d6;
        }
        
        @media (max-width: 768px) {
            .comparison-container {
                flex-direction: column;
            }
        }
        
        .font-error .font-sample {
            font-family: sans-serif !important;
            color: #999;
        }
        
        .font-error-message {
            display: none;
            color: #e74c3c;
            font-size: 12px;
            margin-top: 5px;
        }
        
        .font-error .font-error-message {
            display: block;
        }
    </style>
    <script>
        const fontList = {{ all_fonts|tojson }};
        const fontCapitalizationMap = {};
        
        // Create initial mapping
        fontList.forEach(font => {
            fontCapitalizationMap[font.toLowerCase()] = font;
        });
        
        document.addEventListener('DOMContentLoaded', function() {
            // Preload all fonts
            fontList.forEach(font => {
                preloadGoogleFont(font);
            });
            
            // Check for font loading errors after a delay
            setTimeout(checkFontLoadingErrors, 2000);
        });
        
        function formatFontName(fontName) {
            const lookupName = fontName.toLowerCase();
            
            if (fontCapitalizationMap[lookupName]) {
                return fontCapitalizationMap[lookupName];
            }
            
            return lookupName.split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
        }
        
        function preloadGoogleFont(fontName) {
            const displayName = formatFontName(fontName);
            const fontId = displayName.replace(/\s+/g, '_').toLowerCase();
            
            // Check if already loaded
            if (document.getElementById(`font-${fontId}`)) {
                return;
            }
            
            const fontUrl = getGoogleFontURL(displayName);
            
            const link = document.createElement('link');
            link.id = `font-${fontId}`;
            link.rel = 'stylesheet';
            link.href = fontUrl;
            document.head.appendChild(link);
        }

        // Font mapping utilities
        const fontWeightMap = {
            'buda': '300',
            'opensanscondensed': '300',
            'unifrakturcook': '700',
        };
        
        const fontStyleMap = {
            'molle': 'ital@1'
        };
        
        const fontRenameMap = {
            'codacaption': 'Coda'
        };

        function equalizeHeights() {
            // Get all classifier prediction sections
            const leftClassifiers = document.querySelectorAll('.model-column:nth-child(1) .results-section-container:nth-child(3) .result-item');
            const rightClassifiers = document.querySelectorAll('.model-column:nth-child(2) .results-section-container:nth-child(3) .result-item');

            // Get all similar fonts sections
            const leftSimilar = document.querySelectorAll('.model-column:nth-child(1) .results-section-container:nth-child(5) .result-item');
            const rightSimilar = document.querySelectorAll('.model-column:nth-child(2) .results-section-container:nth-child(5) .result-item');

            // Set all boxes to the same height
            const resultItems = document.querySelectorAll('.result-item');
            resultItems.forEach(item => {
                item.style.height = '160px';
            });
        }

        
        // Load all fonts when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Collect all fonts from the page
            const allFonts = new Set();
            document.querySelectorAll('.font-data').forEach(el => {
                allFonts.add(el.dataset.font);
            });
            
            // Preload all fonts
            allFonts.forEach(fontName => {
                preloadGoogleFont(fontName);
            });
            
            // Check for font loading errors after a delay
            setTimeout(checkFontLoadingErrors, 1500);
        });
        
        function preloadGoogleFont(fontName) {
            const fontId = fontName.replace(/\s+/g, '_').toLowerCase();
            
            // Check if we already loaded this font
            if (document.getElementById(`font-${fontId}`)) {
                return;
            }
            
            const fontUrl = getGoogleFontURL(fontName);
            
            const link = document.createElement('link');
            link.id = `font-${fontId}`;
            link.rel = 'stylesheet';
            link.href = fontUrl;
            document.head.appendChild(link);
        }
        
        function getGoogleFontURL(fontName) {
            const fontLower = fontName.toLowerCase().replace(/\s+/g, '');
            let formattedName = fontName;
            
            if (fontRenameMap[fontLower]) {
                formattedName = fontRenameMap[fontLower];
            }
            
            const googleFontParam = formattedName.replace(/\s+/g, '+');
            
            if (fontWeightMap[fontLower]) {
                return `https://fonts.googleapis.com/css2?family=${googleFontParam}:wght@${fontWeightMap[fontLower]}&display=swap`;
            }
            
            if (fontStyleMap[fontLower]) {
                return `https://fonts.googleapis.com/css2?family=${googleFontParam}:${fontStyleMap[fontLower]}&display=swap`;
            }
            
            return `https://fonts.googleapis.com/css2?family=${googleFontParam}&display=swap`;
        }
        
        function checkFontLoadingErrors() {
            document.querySelectorAll('.result-item').forEach(function(item) {
                const fontSample = item.querySelector('.font-sample');
                if (fontSample) {
                    const fontName = fontSample.style.fontFamily.split(',')[0].replace(/['"]+/g, '');
                    if (!document.fonts.check(`1em "${fontName}"`)) {
                        item.classList.add('font-error');
                        if (!item.querySelector('.font-error-message')) {
                            const errorMsg = document.createElement('div');
                            errorMsg.className = 'font-error-message';
                            errorMsg.textContent = 'Font failed to load - displaying fallback';
                            item.appendChild(errorMsg);
                        }
                    }
                }
            });
        }
    </script>
</head>
<body>
    <h1>Batch Font Analysis Report</h1>
    <p>Generated on: {{ timestamp }}</p>
    <p>Total images processed: {{ total_images }}</p>
    
    {% for item in results %}
    <div class="image-container">
        <div class="image-header">
            <h2>Image: {{ item.filename }}</h2>
        </div>
        
        <img src="data:image/png;base64,{{ item.image_data }}" alt="{{ item.filename }}" class="image-preview">
        
        <div class="comparison-container">
            <!-- Model A column -->
            <div class="model-column">
                <div class="model-header">{{ model_a_name }}</div>
                <div class="section-header">
                    <h3>Classifier Predictions</h3>
                </div>
                <div>
                    {% for result in item.results_a.classifier_predictions %}
                    <div class="result-item">
                        <div class="result-header">
                            <span class="font-name">{{ result.font }}</span>
                            <span class="score">{{ "%.1f"|format(result.probability*100) }}%</span>
                        </div>
                        <div class="font-sample font-data" data-font="{{ result.font }}" style="font-family: '{{ result.font }}', sans-serif;">
                            The quick brown fox jumps over the lazy dog. 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz !@$%^&*()_+-=[]|;:,.<>?/
                        </div>
                        <div class="font-actions">
                            <button class="copy-btn" onclick="navigator.clipboard.writeText('{{ result.font }}').then(() => { this.textContent = 'Copied!'; setTimeout(() => { this.textContent = 'Copy font name'; }, 1500); })">Copy font name</button>
                            <a href="https://fonts.google.com/specimen/{{ result.font.replace(' ', '+') }}" target="_blank" class="font-link">View on Google Fonts</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section-header">
                    <h3>Similar Fonts</h3>
                </div>
                <div>
                    {% for result in item.results_a.embedding_similarity %}
                    <div class="result-item">
                        <div class="result-header">
                            <span class="font-name">{{ result.font }}</span>
                            <span class="score">{{ "%.1f"|format(result.similarity*100) }}%</span>
                        </div>
                        <div class="font-sample font-data" data-font="{{ result.font }}" style="font-family: '{{ result.font }}', sans-serif;">
                            The quick brown fox jumps over the lazy dog. 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz !@$%^&*()_+-=[]|;:,.<>?/
                        </div>
                        <div class="font-actions">
                            <button class="copy-btn" onclick="navigator.clipboard.writeText('{{ result.font }}').then(() => { this.textContent = 'Copied!'; setTimeout(() => { this.textContent = 'Copy font name'; }, 1500); })">Copy font name</button>
                            <a href="https://fonts.google.com/specimen/{{ result.font.replace(' ', '+') }}" target="_blank" class="font-link">View on Google Fonts</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- Model B column -->
            <div class="model-column">
                <div class="model-header">{{ model_b_name }}</div>
                <div class="section-header">
                    <h3>Classifier Predictions</h3>
                </div>
                <div>
                    {% for result in item.results_b.classifier_predictions %}
                    <div class="result-item">
                        <div class="result-header">
                            <span class="font-name">{{ result.font }}</span>
                            <span class="score">{{ "%.1f"|format(result.probability*100) }}%</span>
                        </div>
                        <div class="font-sample font-data" data-font="{{ result.font }}" style="font-family: '{{ result.font }}', sans-serif;">
                            The quick brown fox jumps over the lazy dog. 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz !@$%^&*()_+-=[]|;:,.<>?/
                        </div>
                        <div class="font-actions">
                            <button class="copy-btn" onclick="navigator.clipboard.writeText('{{ result.font }}').then(() => { this.textContent = 'Copied!'; setTimeout(() => { this.textContent = 'Copy font name'; }, 1500); })">Copy font name</button>
                            <a href="https://fonts.google.com/specimen/{{ result.font.replace(' ', '+') }}" target="_blank" class="font-link">View on Google Fonts</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section-header">
                    <h3>Similar Fonts</h3>
                </div>
                <div>
                    {% for result in item.results_b.embedding_similarity %}
                    <div class="result-item">
                        <div class="result-header">
                            <span class="font-name">{{ result.font }}</span>
                            <span class="score">{{ "%.1f"|format(result.similarity*100) }}%</span>
                        </div>
                        <div class="font-sample font-data" data-font="{{ result.font }}" style="font-family: '{{ result.font }}', sans-serif;">
                            The quick brown fox jumps over the lazy dog. 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz !@$%^&*()_+-=[]|;:,.<>?/
                        </div>
                        <div class="font-actions">
                            <button class="copy-btn" onclick="navigator.clipboard.writeText('{{ result.font }}').then(() => { this.textContent = 'Copied!'; setTimeout(() => { this.textContent = 'Copy font name'; }, 1500); })">Copy font name</button>
                            <a href="https://fonts.google.com/specimen/{{ result.font.replace(' ', '+') }}" target="_blank" class="font-link">View on Google Fonts</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
    {% endfor %}
</body>
</html>
"""

def process_image_directory(directory_path, model_a, model_b, class_embeddings_a, 
                            class_embeddings_b, label_mapping_a, label_mapping_b,
                            device, model_a_name, model_b_name, font_mapping):
    """Process all images in the given directory with both models."""
    results = []
    
    # List all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) and 
                  os.path.splitext(f.lower())[1] in image_extensions]
    
    total_images = len(image_files)
    logger.info(f"Found {total_images} images in directory: {directory_path}")
    
    for i, image_file in enumerate(image_files):
        try:
            file_path = os.path.join(directory_path, image_file)
            logger.info(f"Processing image {i+1}/{total_images}: {image_file}")
            
            # Open and prepare the image
            image = Image.open(file_path).convert('RGB')
            
            # Convert to base64 for displaying in HTML
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare image tensor for model
            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
            
            # Get predictions from both models
            results_a = compare_models.predict_with_model(
                model_a, class_embeddings_a, label_mapping_a, image_tensor, False
            )
            
            results_b = compare_models.predict_with_model(
                model_b, class_embeddings_b, label_mapping_b, image_tensor, False
            )

            for result in results_a['embedding_similarity'] + results_a['classifier_predictions']:
                result['font'] = format_font_name(result['font'], font_mapping)
                
            for result in results_b['embedding_similarity'] + results_b['classifier_predictions']:
                result['font'] = format_font_name(result['font'], font_mapping)
            
            
            # Add to results list
            results.append({
                'filename': image_file,
                'image_data': image_base64,
                'results_a': results_a,
                'results_b': results_b
            })
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")
            logger.error(traceback.format_exc())
    
    return results, total_images

def generate_report(results, total_images, model_a_name, model_b_name, output_path, font_mapping):
    """Generate HTML report with all results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a minimal Flask app for templating
    app = Flask(__name__)

    all_fonts = set()
    for item in results:
        for result in (item['results_a']['embedding_similarity'] + 
                      item['results_a']['classifier_predictions'] +
                      item['results_b']['embedding_similarity'] + 
                      item['results_b']['classifier_predictions']):
            all_fonts.add(result['font'])
    
    # Use app context for rendering
    with app.app_context():
        # Render the template
        html_content = render_template_string(
            REPORT_TEMPLATE,
            results=results,
            total_images=total_images,
            timestamp=timestamp,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            all_fonts=list(all_fonts)
        )
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report generated at: {output_path}")
    return output_path


def format_font_name(model_font_name, font_mapping):
    """Format font name from model format to display format."""
    lookup_name = model_font_name.replace('_', ' ').lower()
    
    if lookup_name in font_mapping:
        return font_mapping[lookup_name]
    
    # Fallback to capitalization
    return ' '.join(word.capitalize() for word in lookup_name.split())


def main():
    """Run the batch processing script."""
    parser = argparse.ArgumentParser(description="Process a directory of images with two font models")
    parser.add_argument("--image_dir", required=True, help="Directory containing images to process")
    parser.add_argument("--model_a_path", required=True, help="Path to model A .pt file")
    parser.add_argument("--model_b_path", required=True, help="Path to model B .pt file")
    parser.add_argument("--data_dir", required=True, help="Directory containing embeddings and label mappings")
    parser.add_argument("--embeddings_a_path", required=True, help="Path to embeddings for model A")
    parser.add_argument("--embeddings_b_path", required=True, help="Path to embeddings for model B")
    parser.add_argument("--label_mapping_a", default="label_mapping.npy", help="Label mapping file for model A")
    parser.add_argument("--label_mapping_b", default="label_mapping.npy", help="Label mapping file for model B")
    parser.add_argument("--output_html", default="compare_models.html", help="Path to save the HTML report")
    parser.add_argument("--serve", action="store_true", help="Start a webserver to view the report")
    parser.add_argument("--port", type=int, default=8080, help="Port for the webserver")
    
    args = parser.parse_args()

    # Load font capitalization mapping
    font_mapping = {}
    font_mapping_path = os.path.join(os.path.dirname(__file__), 'static', 'available_fonts.txt')
    
    try:
        if os.path.exists(font_mapping_path):
            with open(font_mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    fontname = line.strip()
                    if fontname:
                        font_mapping[fontname.lower()] = fontname
            logger.info(f"Loaded {len(font_mapping)} fonts from mapping file")
        else:
            logger.warning(f"Font mapping file not found: {font_mapping_path}")
    except Exception as e:
        logger.error(f"Error loading font mapping: {e}")

    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare paths
    label_mapping_path_a = os.path.join(args.data_dir, args.label_mapping_a)
    label_mapping_path_b = os.path.join(args.data_dir, args.label_mapping_b)
    
    # Load models (reusing code from compare_models.py)
    logger.info(f"Loading model A: {os.path.basename(args.model_a_path)}")
    model_a, class_embeddings_a, label_mapping_a = compare_models.load_model(
        args.model_a_path, args.embeddings_a_path, label_mapping_path_a, device
    )
    
    logger.info(f"Loading model B: {os.path.basename(args.model_b_path)}")
    model_b, class_embeddings_b, label_mapping_b = compare_models.load_model(
        args.model_b_path, args.embeddings_b_path, label_mapping_path_b, device
    )
    
    model_a_name = os.path.basename(args.model_a_path)
    model_b_name = os.path.basename(args.model_b_path)
    
    # Process all images in the directory
    results, total_images = process_image_directory(
        args.image_dir, model_a, model_b, 
        class_embeddings_a, class_embeddings_b,
        label_mapping_a, label_mapping_b, 
        device, model_a_name, model_b_name, font_mapping
    )
    
    # Generate the HTML report
    report_path = generate_report(
        results, total_images, model_a_name, model_b_name, 
        args.output_html, font_mapping
    )
    
    print(f"Report generated: {report_path}")
    
    # Optionally serve the report
    if args.serve:
        app = Flask(__name__)
        
        @app.route('/')
        def serve_report():
            with open(args.output_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        
        print(f"Starting server at http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()