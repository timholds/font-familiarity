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
from ml.char_model import CRAFTFontClassifier
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
        }
        
        .score {
            font-weight: bold;
            color: #4285f4;
        }
        
        @media (max-width: 768px) {
            .comparison-container {
                flex-direction: column;
            }
        }
    </style>
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

def process_image_directory(directory_path, model_a, model_b, class_embeddings_a, class_embeddings_b, 
                           label_mapping_a, label_mapping_b, device, model_a_name, model_b_name):
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

def generate_report(results, total_images, model_a_name, model_b_name, output_path):
    """Generate HTML report with all results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a minimal Flask app for templating
    app = Flask(__name__)
    
    # Use app context for rendering
    with app.app_context():
        # Render the template
        html_content = render_template_string(
            REPORT_TEMPLATE,
            results=results,
            total_images=total_images,
            timestamp=timestamp,
            model_a_name=model_a_name,
            model_b_name=model_b_name
        )
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report generated at: {output_path}")
    return output_path

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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare paths
    label_mapping_path_a = os.path.join(args.data_dir, args.label_mapping_a)
    label_mapping_path_b = os.path.join(args.data_dir, args.label_mapping_b)
    
    # Load models (reusing code from compare_models.py)
    logger.info(f"Loading model A: {os.path.basename(args.model_a_path)}")
    model_a, class_embeddings_a, label_mapping_a = compare_models.load_model(
        args.model_a_path, args.embeddings_a_path, label_mapping_path_a
    )
    
    logger.info(f"Loading model B: {os.path.basename(args.model_b_path)}")
    model_b, class_embeddings_b, label_mapping_b = compare_models.load_model(
        args.model_b_path, args.embeddings_b_path, label_mapping_path_b
    )
    
    model_a_name = os.path.basename(args.model_a_path)
    model_b_name = os.path.basename(args.model_b_path)
    
    # Process all images in the directory
    results, total_images = process_image_directory(
        args.image_dir, model_a, model_b, 
        class_embeddings_a, class_embeddings_b,
        label_mapping_a, label_mapping_b, 
        device, model_a_name, model_b_name
    )
    
    # Generate the HTML report
    report_path = generate_report(
        results, total_images, model_a_name, model_b_name, args.output_html
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