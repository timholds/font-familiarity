from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import io
import argparse
import time
import json
import io as bio
import base64
import re
import traceback
import logging
from datetime import datetime
import uuid
import torch.nn.functional as F
from char_model import CRAFTFontClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for both models
model_a = None
model_b = None
class_embeddings_a = None
class_embeddings_b = None
device = None
label_mapping_a = None
label_mapping_b = None

def load_model(model_path, embeddings_path, label_mapping_path, device):
    """Load a single model and return the model, embeddings, and label mapping."""
    try:
        # Load label mapping
        label_mapping_raw = np.load(label_mapping_path, allow_pickle=True).item()
        label_mapping = {v: k for k, v in label_mapping_raw.items()}
        
        # Load model state
        state = torch.load(model_path, map_location=device)
        
        # Check if we have the full model object (new format)
        if 'model' in state and hasattr(state['model'], 'eval'):
            # New format: full model object
            model = state['model']
            model = model.to(device)
            model.eval()
            
            # For inference with raw images, ensure precomputed CRAFT is disabled
            if hasattr(model, 'use_precomputed_craft'):
                model.use_precomputed_craft = False
                
            logger.info("Loaded full model object from checkpoint")
        else:
            # This should not happen with the new format, but keeping for safety
            raise ValueError("Expected full model object in checkpoint, but found old format. Please retrain your model.")
        
        
        # Load embeddings
        class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
        
        return model, class_embeddings, label_mapping
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def predict_with_model(model, class_embeddings, label_mapping, image_tensor, research_mode=False):
    """Process an image with the given model and return formatted results."""
    results = {}
    
    # Generate visualization if in research mode
    if research_mode and isinstance(model, CRAFTFontClassifier):
        try:
            visual_image = model.visualize_craft_detections(
                images=image_tensor,
                label_mapping=label_mapping,
                targets=None,
                save_path=None
            )
            
            buffer = bio.BytesIO()
            visual_image.save(buffer, format='PNG')
            encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            results['visualization'] = encoded_img
        except Exception as vis_error:
            logger.error(f"Visualization error: {str(vis_error)}")
            results['visualization_error'] = str(vis_error)
    
    # Run model inference
    with torch.no_grad():
        if isinstance(model, CRAFTFontClassifier):
            outputs = model(image_tensor)
            embedding = outputs['font_embedding']
            logits = outputs['logits']
        else:
            embedding = model.get_embedding(image_tensor)
            logits = model.classifier(embedding)
        
        # Get embedding similarity results
        similarities = torch.mm(F.normalize(embedding, p=2, dim=1), 
                                F.normalize(class_embeddings, p=2, dim=1).t())
        top_k_similarities, top_k_indices = similarities[0].topk(5)
        
        # Get classifier prediction results
        probabilities = F.softmax(logits, dim=1)
        top_k_probs, top_k_cls_indices = probabilities[0].topk(5)
        
        # Format results
        embedding_results = [
            {
                'font': label_mapping.get(idx.item(), f'Unknown Font ({idx.item()})'),
                'similarity': float(score)
            }
            for idx, score in zip(top_k_indices, top_k_similarities)
        ]
        
        classifier_results = [
            {
                'font': label_mapping.get(idx.item(), f'Unknown Font ({idx.item()})'),
                'probability': float(prob)
            }
            for idx, prob in zip(top_k_cls_indices, top_k_probs)
        ]
        
        results['embedding_similarity'] = embedding_results
        results['classifier_predictions'] = classifier_results
    
    return results

def create_app(model_path_a, model_path_b, data_dir, 
               embeddings_path_a, embeddings_path_b,
               label_mapping_file_a, label_mapping_file_b, 
               uploads_dir="uploads"):
    """Create Flask app instance for model comparison."""
    global model_a, model_b, class_embeddings_a, class_embeddings_b, label_mapping_a, label_mapping_b, device
    
    app = Flask(__name__)
    uploads_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), uploads_dir)
    os.makedirs(uploads_path, exist_ok=True)
    logger.info(f"Uploads directory: {uploads_path}")

    model_a_name = os.path.basename(model_path_a)
    model_b_name = os.path.basename(model_path_b)
    
    app.config['UPLOADS_PATH'] = uploads_path
    app.config['MODEL_A_NAME'] = model_a_name
    app.config['MODEL_B_NAME'] = model_b_name
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare paths
    label_mapping_path_a = os.path.join(data_dir, label_mapping_file_a)
    label_mapping_path_b = os.path.join(data_dir, label_mapping_file_b)
    
    # Load models
    logger.info(f"Loading model A: {model_a_name}")
    model_a, class_embeddings_a, label_mapping_a = load_model(
        model_path_a, embeddings_path_a, label_mapping_path_a, device,
    )
    
    logger.info(f"Loading model B: {model_b_name}")
    model_b, class_embeddings_b, label_mapping_b = load_model(
        model_path_b, embeddings_path_b, label_mapping_path_b, device,
    )
    
    # Define routes
    @app.route('/')
    def index():
        """Serve the comparison page."""
        css_version = int(time.time())
        return render_template('compare_models.html', 
                              css_version=css_version,
                              model_a_name=app.config['MODEL_A_NAME'],
                              model_b_name=app.config['MODEL_B_NAME'])
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Process image with both models."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        research_mode = request.form.get('research_mode', 'false').lower() == 'true'
        
        try:
            # Process the uploaded image
            image_bytes = request.files['image'].read()
            
            # Get IP address
            if request.headers.getlist("X-Forwarded-For"):
                ip_address = request.headers.getlist("X-Forwarded-For")[0]
            else:
                ip_address = request.remote_addr
            
           
            # Prepare image tensor
            original_image = Image.open(io.BytesIO(image_bytes))
            image = original_image.convert('RGB')
            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
            
            # Get predictions from both models
            results_a = predict_with_model(model_a, class_embeddings_a, label_mapping_a, image_tensor, research_mode)
            results_b = predict_with_model(model_b, class_embeddings_b, label_mapping_b, image_tensor, research_mode)
            
            # Combine results
            prediction_results = {
                model_a_name: results_a,
                model_b_name: results_b,
                'research_mode': research_mode
            }
            
            
            # Return combined results
            return jsonify({
                app.config['MODEL_A_NAME']: results_a,
                app.config['MODEL_B_NAME']: results_b,
            })
            
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f"Error processing image: {str(e)}",
                'details': traceback.format_exc()
            }), 500
    
    return app

def main():
    """Initialize and run the Flask application."""
    parser = argparse.ArgumentParser(description="Compare two font models in a web interface")
    parser.add_argument("--model_a_path", required=True, help="Path to model A .pt file")
    parser.add_argument("--model_b_path", required=True, help="Path to model B .pt file")
    parser.add_argument("--data_dir", required=True, help="Directory containing embeddings and label mappings")
    parser.add_argument("--embeddings_a_path", required=True, help="Path to embeddings for model A")
    parser.add_argument("--embeddings_b_path", required=True, help="Path to embeddings for model B")
    parser.add_argument("--label_mapping_a", default="label_mapping.npy", help="Label mapping file for model A")
    parser.add_argument("--label_mapping_b", default="label_mapping.npy", help="Label mapping file for model B")
    parser.add_argument("--uploads_dir", default="uploads", help="Directory to save uploaded images")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the Flask server on")
    
    args = parser.parse_args()
    
    app = create_app(
        model_path_a=args.model_a_path,
        model_path_b=args.model_b_path,
        data_dir=args.data_dir,
        embeddings_path_a=args.embeddings_a_path,
        embeddings_path_b=args.embeddings_b_path,
        label_mapping_file_a=args.label_mapping_a,
        label_mapping_file_b=args.label_mapping_b,
        uploads_dir=args.uploads_dir,
    )
    
    app.run(host='0.0.0.0', port=args.port, debug=True)

if __name__ == '__main__':
    main()