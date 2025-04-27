from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import io
import torch.nn.functional as F
import logging
import traceback
import argparse
import time
import json
import base64
from ml.char_model import CRAFTFontClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model state
model1 = None
model2 = None
embeddings1 = None
embeddings2 = None
device = None
label_mapping = None
model1_name = None
model2_name = None
is_initialized = False

def sanitize_font_name(font_name):
    """Clean and prepare font name for Google Fonts API."""
    import re
    
    # Remove any version numbers, weights or styles in parentheses
    clean_name = re.sub(r'\s*\([^)]*\)', '', font_name)
    
    # Remove common weight/style suffixes
    for suffix in [' Regular', ' Bold', ' Italic', ' Light', ' Medium', ' Black', ' Thin']:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
    
    # Keep alphanumerics, spaces, and some safe characters
    clean_name = re.sub(r'[^a-zA-Z0-9 \-_]', '', clean_name)
    
    return clean_name.strip()

def generate_font_html(font_name):
    """Generate HTML to display sample text in the specified font with fallbacks."""
    # Original font name (for display)
    original_name = font_name
    
    # Clean the font name for Google Fonts
    sanitized_font = sanitize_font_name(font_name)
    encoded_font_name = sanitized_font.replace(' ', '+')
    
    # Create a unique ID for font loading detection
    import hashlib
    font_id = hashlib.md5(original_name.encode()).hexdigest()[:8]
    
    # Create the HTML with font loading detection
    html = f"""
    <div class="font-preview" id="font-{font_id}">
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family={encoded_font_name}&display=swap">
        <div class="font-header">
            <span class="font-name-display">{original_name}</span>
        </div>
        <div class="font-sample-wrapper">
            <p class="font-sample" style="font-family: '{sanitized_font}', sans-serif;">
                The quick brown fox jumps over the lazy dog. 0123456789
            </p>
        </div>
        <div class="font-fallback hidden">
            <p class="fallback-message">Font preview unavailable</p>
            <p class="fallback-sample">Sample: The quick brown fox jumps over the lazy dog. 0123456789</p>
        </div>
    </div>
    """
    return html

def get_top_k_similar_fonts(query_embedding, class_embeddings, k=5):
    """Find k most similar fonts using embedding similarity."""
    # Normalize query embedding for cosine similarity
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute cosine similarity with all class embeddings
    similarities = torch.mm(query_embedding, class_embeddings.t())
    
    # Get top k similarities and indices
    top_k_similarities, top_k_indices = similarities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()

def get_top_k_predictions(logits, k=5):
    """Get top k predictions from classifier."""
    probabilities = F.softmax(logits, dim=1)
    top_k_probs, top_k_indices = probabilities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()

def load_models(model1_path, model2_path, embeddings1_path, embeddings2_path, label_mapping_path):
    """Initialize both models and load pre-computed embeddings."""
    global model1, model2, embeddings1, embeddings2, device, label_mapping, is_initialized, model1_name, model2_name
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load label mapping
        if not os.path.isfile(label_mapping_path):
            raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")
        
        logger.info(f"Loading label mapping from {label_mapping_path}")
        label_mapping_raw = np.load(label_mapping_path, allow_pickle=True).item()
        
        # Invert the mapping to go from index -> font name
        label_mapping = {v: k for k, v in label_mapping_raw.items()}
        
        # Load Model 1
        logger.info(f"Loading model 1 from {model1_path}")
        model1_name = os.path.basename(model1_path)
        state1 = torch.load(model1_path, map_location=device)
        state_dict1 = state1['model_state_dict']
        
        # Determine number of classes from classifier
        classifier_key = 'font_classifier.font_classifier.weight'
        if classifier_key in state_dict1:
            num_fonts = state_dict1[classifier_key].shape[0]
        else:
            # Try to find classifier key
            classifier_keys = [k for k in state_dict1.keys() if 'classifier' in k and 'weight' in k]
            if classifier_keys:
                classifier_key = classifier_keys[0]
                num_fonts = state_dict1[classifier_key].shape[0]
            else:
                raise ValueError("Could not determine number of font classes from model 1")
        
        # Determine embedding dimension
        embedding_dim = 512  # Default fallback
        for key in state_dict1.keys():
            if 'projection' in key and 'weight' in key:
                embedding_dim = state_dict1[key].shape[0]
                break
        
        # Initialize model 1
        model1 = CRAFTFontClassifier(
            num_fonts=num_fonts,
            device=device,
            patch_size=32, 
            embedding_dim=embedding_dim,
            craft_fp16=False,
            use_precomputed_craft=False
        )
        
        # Load weights
        model1.load_state_dict(state_dict1)
        model1 = model1.to(device)
        model1.eval()
        
        # Load embeddings for model 1
        logger.info(f"Loading embeddings for model 1 from {embeddings1_path}")
        embeddings1 = torch.from_numpy(np.load(embeddings1_path)).to(device)
        
        # Load Model 2
        logger.info(f"Loading model 2 from {model2_path}")
        model2_name = os.path.basename(model2_path)
        state2 = torch.load(model2_path, map_location=device)
        state_dict2 = state2['model_state_dict']
        
        # Determine number of classes from classifier
        if classifier_key in state_dict2:
            num_fonts = state_dict2[classifier_key].shape[0]
        else:
            # Try to find classifier key
            classifier_keys = [k for k in state_dict2.keys() if 'classifier' in k and 'weight' in k]
            if classifier_keys:
                classifier_key = classifier_keys[0]
                num_fonts = state_dict2[classifier_key].shape[0]
            else:
                raise ValueError("Could not determine number of font classes from model 2")
        
        # Initialize model 2
        model2 = CRAFTFontClassifier(
            num_fonts=num_fonts,
            device=device,
            patch_size=32, 
            embedding_dim=embedding_dim,
            craft_fp16=False,
            use_precomputed_craft=False
        )
        
        # Load weights
        model2.load_state_dict(state_dict2)
        model2 = model2.to(device)
        model2.eval()
        
        # Load embeddings for model 2
        logger.info(f"Loading embeddings for model 2 from {embeddings2_path}")
        embeddings2 = torch.from_numpy(np.load(embeddings2_path)).to(device)
        
        is_initialized = True
        logger.info("Models initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_app(model1_path, model2_path, embeddings1_path, embeddings2_path, 
               label_mapping_path, uploads_dir="uploads"):
    """Factory function to create and configure Flask app instance."""
    app = Flask(__name__)
    uploads_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), uploads_dir)
    os.makedirs(uploads_path, exist_ok=True)
    logger.info(f"Uploads directory: {uploads_path}")
    
    # Store path in app config for access in routes
    app.config['UPLOADS_PATH'] = uploads_path

    # Load models
    load_models(model1_path, model2_path, embeddings1_path, embeddings2_path, label_mapping_path)
    
    @app.route('/test')
    def test():
        """Test route to verify Flask is working."""
        return "Flask server is running!"

    @app.route('/')
    def index():
        """Serve the main page."""
        css_version = int(time.time())
        return render_template('compare_models_frontend.html', 
                               css_version=css_version,
                               model1_name=model1_name,
                               model2_name=model2_name)

    @app.route('/compare', methods=['POST'])
    def compare():
        """Endpoint for comparing font predictions from both models."""
        if not is_initialized:
            return jsonify({'error': 'Models not initialized'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        research_mode = request.form.get('research_mode', 'false').lower() == 'true'
        
        try:
            # Get and preprocess image
            image_bytes = request.files['image'].read()
            
            # Save the uploaded image (optional)
            image_path = os.path.join(app.config['UPLOADS_PATH'], f"compare_{int(time.time())}.png")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Convert uploaded image bytes to tensor
            original_image = Image.open(io.BytesIO(image_bytes))
            image = original_image.convert('RGB')
            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
            
            response_data = {
                'model1_name': model1_name,
                'model2_name': model2_name
            }
            
            # Generate visualizations if in research mode
            if research_mode:
                logger.info("Generating visualizations for research mode")
                try:
                    # Model 1 visualization
                    visual_image1 = model1.visualize_craft_detections(
                        images=image_tensor,
                        label_mapping=label_mapping,
                        targets=None,
                        save_path=None
                    )
                    
                    # Model 2 visualization
                    visual_image2 = model2.visualize_craft_detections(
                        images=image_tensor,
                        label_mapping=label_mapping,
                        targets=None,
                        save_path=None
                    )
                    
                    # Convert visualizations to base64
                    buffer1 = io.BytesIO()
                    visual_image1.save(buffer1, format='PNG')
                    encoded_img1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
                    
                    buffer2 = io.BytesIO()
                    visual_image2.save(buffer2, format='PNG')
                    encoded_img2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
                    
                    response_data['visualization1'] = encoded_img1
                    response_data['visualization2'] = encoded_img2
                    
                    logger.info("Visualizations generated successfully")
                except Exception as vis_error:
                    logger.error(f"Visualization error: {str(vis_error)}")
                    response_data['visualization_error'] = str(vis_error)
            
            # Process with both models
            with torch.no_grad():
                # Model 1 predictions
                outputs1 = model1(image_tensor)
                embedding1 = outputs1['font_embedding']
                logits1 = outputs1['logits']
                
                # Model 2 predictions
                outputs2 = model2(image_tensor)
                embedding2 = outputs2['font_embedding']
                logits2 = outputs2['logits']
                
                # Get predictions using embedding similarity
                top_k_indices1, top_k_similarities1 = get_top_k_similar_fonts(embedding1, embeddings1, k=5)
                top_k_indices2, top_k_similarities2 = get_top_k_similar_fonts(embedding2, embeddings2, k=5)
                
                # Get predictions using classifier
                top_k_cls_indices1, top_k_probs1 = get_top_k_predictions(logits1, k=5)
                top_k_cls_indices2, top_k_probs2 = get_top_k_predictions(logits2, k=5)
                
                embedding_results1 = []
                for idx, score in zip(top_k_indices1, top_k_similarities1):
                    font_name = label_mapping.get(idx, f'Unknown Font ({idx})')
                    embedding_results1.append({
                        'font': font_name,
                        'similarity': float(score),
                        'html': generate_font_html(font_name) if 'Unknown Font' not in font_name else ''
                    })

                classifier_results1 = []
                for idx, prob in zip(top_k_cls_indices1, top_k_probs1):
                    font_name = label_mapping.get(idx, f'Unknown Font ({idx})')
                    classifier_results1.append({
                        'font': font_name,
                        'probability': float(prob),
                        'html': generate_font_html(font_name) if 'Unknown Font' not in font_name else ''
                    })

                embedding_results2 = []
                for idx, score in zip(top_k_indices2, top_k_similarities2):
                    font_name = label_mapping.get(idx, f'Unknown Font ({idx})')
                    embedding_results2.append({
                        'font': font_name,
                        'similarity': float(score),
                        'html': generate_font_html(font_name) if 'Unknown Font' not in font_name else ''
                    })

                classifier_results2 = []
                for idx, prob in zip(top_k_cls_indices2, top_k_probs2):
                    font_name = label_mapping.get(idx, f'Unknown Font ({idx})')
                    classifier_results2.append({
                        'font': font_name,
                        'probability': float(prob),
                        'html': generate_font_html(font_name) if 'Unknown Font' not in font_name else ''
                    })
                
                # Combine results
                response_data.update({
                    'model1': {
                        'embedding_similarity': embedding_results1,
                        'classifier_predictions': classifier_results1
                    },
                    'model2': {
                        'embedding_similarity': embedding_results2,
                        'classifier_predictions': classifier_results2
                    },
                    'research_mode': research_mode
                })
                
                return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error processing comparison: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f"Error processing image: {str(e)}",
                'details': traceback.format_exc()
            }), 500

    return app

def main():
    """Initialize and run the Flask application when run directly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_path", required=True, help="Path to first model .pt file")
    parser.add_argument("--model2_path", required=True, help="Path to second model .pt file")
    parser.add_argument("--embeddings1_path", required=True, help="Path to first model embeddings")
    parser.add_argument("--embeddings2_path", required=True, help="Path to second model embeddings")
    parser.add_argument("--label_mapping_path", required=True, help="Path to label mapping file")
    parser.add_argument("--uploads_dir", default="uploads", help="Directory to save uploaded images")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    app = create_app(
        args.model1_path, 
        args.model2_path, 
        args.embeddings1_path, 
        args.embeddings2_path, 
        args.label_mapping_path,
        uploads_dir=args.uploads_dir
    )
    app.run(host='0.0.0.0', port=args.port, debug=True)

if __name__ == '__main__':
    main()