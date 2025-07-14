from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
from ml.font_model import SimpleCNN
from ml.char_model import CRAFTFontClassifier
import torch.nn.functional as F
import traceback
import logging
import argparse
import time
import json
import io as bio
import base64
import uuid 
from datetime import datetime
from ml.utils import get_params_from_model_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Register additional image format support
HEIC_SUPPORT = False
AVIF_SUPPORT = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    logger.info("HEIC/HEIF support enabled")
except ImportError:
    logger.warning("HEIC/HEIF support not available - install pillow-heif")

try:
    import pillow_avif
    AVIF_SUPPORT = True
    logger.info("AVIF support enabled") 
except ImportError:
    logger.warning("AVIF support not available - install pillow-avif-plugin")

# Global variables for model state
model = None
class_embeddings = None
device = None
label_mapping = None
is_initialized = False

# def preprocess_image(image_bytes: bytes) -> torch.Tensor:
#     """Convert uploaded image bytes to tensor."""
#     try:
#         # Open image and convert to grayscale
#         image = Image.open(io.BytesIO(image_bytes)).convert('L')
#         logger.info(f"Loaded image, size: {image.size}")
        
#         # Define preprocessing
#         transform = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#         ])
        
#         # Apply preprocessing and add batch dimension
#         tensor = transform(image).unsqueeze(0)
#         logger.info(f"Preprocessed tensor shape: {tensor.shape}")
        
#         return tensor.to(device)
        
#     except Exception as e:
#         logger.error(f"Error preprocessing image: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise

def save_uploaded_image(image_bytes, save_dir, ip_address=None, prediction_data=None):
    """Save uploaded image with a unique filename for future analysis."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Detect actual image format
    try:
        img = Image.open(io.BytesIO(image_bytes))
        format_name = img.format.lower() if img.format else 'png'
        # Map PIL format names to file extensions
        format_extensions = {
            'jpeg': 'jpg',
            'jpg': 'jpg', 
            'png': 'png',
            'webp': 'webp',
            'heif': 'heic',
            'heic': 'heic',
            'avif': 'avif',
            'bmp': 'bmp',
            'gif': 'gif'
        }
        extension = format_extensions.get(format_name, 'png')
    except Exception as e:
        logger.warning(f"Could not detect image format: {e}, defaulting to .png")
        extension = 'png'
    
    # Generate unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    filename = f"{timestamp}_{unique_id}.{extension}"

    image_path = os.path.join(save_dir, filename)
    metadata_path = os.path.join(save_dir, f"{timestamp}_{unique_id}_metadata.json")
    
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    # Save metadata including IP address
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'ip_address': ip_address,
        'detected_format': extension
    }
    
    # Add prediction data if provided
    if prediction_data:
        metadata['prediction_data'] = prediction_data
    
    # Save metadata to JSON file
    metadata_path = os.path.join(save_dir, f"{timestamp}_{unique_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved uploaded image to: {image_path}")
    return image_path, metadata_path


def update_metadata(metadata_path, prediction_data):
    """Update metadata file with prediction results."""
    try:
        # Read existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add prediction data
        metadata['prediction_data'] = prediction_data
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Updated metadata at: {metadata_path} with prediction results")
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}")


def get_top_k_similar_fonts(query_embedding: torch.Tensor, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find k most similar fonts using embedding similarity."""
    # Add debug logging
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Class embeddings shape: {class_embeddings.shape}")
    
    # Normalize query embedding for cosine similarity
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute cosine similarity with all class embeddings
    similarities = torch.mm(query_embedding, class_embeddings.t())
    
    # Get top k similarities and indices
    top_k_similarities, top_k_indices = similarities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()

def get_top_k_predictions(logits: torch.Tensor, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Get top k predictions from classifier."""
    probabilities = F.softmax(logits, dim=1)
    top_k_probs, top_k_indices = probabilities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()


def load_char_model_and_embeddings(model_path: str, 
                                   embeddings_path: str, 
                                   label_mapping_path: str) -> None:
    """Initialize character-based model and load pre-computed embeddings."""
    global model, class_embeddings, device, label_mapping, is_initialized
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Verify all required files exist
        for path, name in [(model_path, 'Model'), 
                          (embeddings_path, 'Embeddings'), 
                          (label_mapping_path, 'Label mapping')]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
            logger.info(f"Found {name} file: {path}")
            
        # Load label mapping
        logger.info(f"\nLoading label mapping from {label_mapping_path}")
        label_mapping_raw = np.load(label_mapping_path, allow_pickle=True).item()
        
        # Invert the mapping to go from index -> font name
        label_mapping = {v: k for k, v in label_mapping_raw.items()}
        
        # Load model state
        logger.info(f"\nLoading character model from {model_path}")
        state = torch.load(model_path, map_location=device)
        
        # Check if we have the full model object (new format)
        if 'model' in state and hasattr(state['model'], 'eval'):
            # New format: full model object
            model = state['model']
            model = model.to(device)
            model.eval()
            logger.info("Loaded full model object from checkpoint")
        else:
            # This should not happen with the new format, but keeping for safety
            raise ValueError("Expected full model object in checkpoint, but found old format. Please retrain your model.")
        
        # Load embeddings
        logger.info(f"\nLoading embeddings from {embeddings_path}")
        class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
        
        is_initialized = True
        logger.info("\nCharacter-based model initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during character model initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def load_model_and_embeddings(model_path: str, 
                            embeddings_path: str, 
                            label_mapping_path: str) -> None:
    """Initialize model and load pre-computed class embeddings."""
    global model, class_embeddings, device, label_mapping, is_initialized
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Verify all required files exist
        for path, name in [(model_path, 'Model'), 
                          (embeddings_path, 'Embeddings'), 
                          (label_mapping_path, 'Label mapping')]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
            logger.info(f"Found {name} file: {path}")
            
        # Load label mapping
        logger.info(f"\nLoading label mapping from {label_mapping_path}")
        label_mapping_raw = np.load(label_mapping_path, allow_pickle=True).item()
        
        # Invert the mapping to go from index -> font name
        label_mapping = {v: k for k, v in label_mapping_raw.items()}
        
        # Load model state
        logger.info(f"\nLoading model from {model_path}")
        state = torch.load(model_path, map_location=device)
        state_dict = state['model_state_dict']
        
        # Get model dimensions
        embedding_weight = state_dict['embedding_layer.0.weight']
        classifier_weight = state_dict['classifier.weight']
        embedding_dim = embedding_weight.shape[0]
        num_classes = classifier_weight.shape[0]
        initial_channels = state_dict['features.0.weight'].shape[0]
        
        # Initialize model
        model = SimpleCNN(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            initial_channels=initial_channels
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        
        # Load embeddings
        logger.info(f"\nLoading embeddings from {embeddings_path}")
        class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
        
        is_initialized = True
        logger.info("\nInitialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_app(model_path=None, data_dir=None, embeddings_path=None, 
               label_mapping_file=None, use_char_model=True, uploads_dir="uploads"):
    """Factory function to create and configure Flask app instance."""
    app = Flask(__name__)
    uploads_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), uploads_dir)
    os.makedirs(uploads_path, exist_ok=True)
    logger.info(f"Uploads directory: {uploads_path}")
    
    # Store path in app config for access in routes
    app.config['UPLOADS_PATH'] = uploads_path

    label_mapping_path = os.path.join(data_dir, label_mapping_file)

    if use_char_model:
        # For character-based model
        load_char_model_and_embeddings(model_path, embeddings_path, label_mapping_path)
    else:
        # For traditional SimpleCNN
        load_model_and_embeddings(model_path, embeddings_path, label_mapping_path)
    
    # Handle both factory and command-line initialization
    # if model_path and data_dir and embeddings_path:
    #     label_mapping_path = os.path.join(data_dir, label_mapping_file)
    #     load_model_and_embeddings(model_path, embeddings_path, label_mapping_path)
    
    @app.route('/test')
    def test():
        """Test route to verify Flask is working."""
        return "Flask server is running!"

    @app.route('/')
    def index():
        """Serve the main page."""
        css_version = int(time.time())
        return render_template('frontend.html', css_version=css_version)

    @app.route('/predict', methods=['POST'])
    def predict():
        """Endpoint for font prediction using both similarity and classification approaches."""
        if not is_initialized:
            return jsonify({'error': 'Model not initialized'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        research_mode = request.form.get('research_mode', 'false').lower() == 'true'
        try:

            # TODO figure out exactly what format we need our images in 
            # to visualize them with visualize_craft_detections

            # Get and preprocess image
            image_bytes = request.files['image'].read()
            
            # Get IP address - handle proxies 
            if request.headers.getlist("X-Forwarded-For"):
                # If behind a proxy, get the real IP
                ip_address = request.headers.getlist("X-Forwarded-For")[0]
            else:
                ip_address = request.remote_addr
            # TODO get uploads path and prediction data and pass
            image_path, metadata_path = save_uploaded_image(
                image_bytes, 
                app.config['UPLOADS_PATH'], 
                ip_address
            )

            
            # Convert uploaded image bytes to tensor 
            # Todo this needs to match the dataloader
            try:
                original_image = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Original image mode: {original_image.mode}, size: {original_image.size}")
                image = original_image.convert('RGB')
                logger.info(f"Converted image mode: {image.mode}, size: {image.size}")
            except Exception as e:
                if "cannot identify image file" in str(e):
                    supported_formats = ['JPEG', 'PNG', 'WebP', 'BMP', 'GIF']
                    if HEIC_SUPPORT:
                        supported_formats.append('HEIC')
                    if AVIF_SUPPORT:
                        supported_formats.append('AVIF')
                    
                    error_msg = f'Unsupported image format. Please use: {", ".join(supported_formats)}.'
                    if not HEIC_SUPPORT:
                        error_msg += ' If you have an iPhone image (HEIC), please convert it to JPEG first.'
                    
                    return jsonify({
                        'error': error_msg,
                        'supported_formats': supported_formats
                    }), 400
                else:
                    raise
            
            # do i need to creat a numpy tensor or can torch take in pil iamge 

            # TODO get rid of totensor and pass hwc 0,255 since we 
            # gotta extract patches with craft
            # transform = transforms.Compose([
            #     transforms.Resize((512, 512)),
            #     #transforms.PILToTensor() 
            #     #transforms.ToTensor()
            # ])
            # convert pil image to numpy

            # TODO can i pass a numpy array to the model since craft extract patches wants numpy array
            # and then convert to torch tensor afterwards? would save a data roundtrip
            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
            # image tensor BHWC 0, 255 still
            response_data = {}
        
            # TODO use built in visualize_craft_detections 
            # Generate visualization if in research mode
            if research_mode and isinstance(model, CRAFTFontClassifier):
                logger.info("Generating visualization for research mode")   
                try:
                    # TODO add option to return image
                    # breakpoint()
                    # TODO want unnormalized BHWC tensor
                    #print(f"original image shape: {original_image.size}, type {type(original_image)}")
                    # want tensor 0, 255 bHWC
                    visual_image = model.visualize_craft_detections(
                        images=image_tensor,  # Original images
                        label_mapping=label_mapping,
                        targets=None,
                        save_path=None
                    )
                    print(f"finished vis craft detections, image type: {type(visual_image)}")

                    # Convert visualization to base64
                    buffer = bio.BytesIO()
                    visual_image.save(buffer, format='PNG')
                    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    response_data['visualization'] = encoded_img
                    logger.info("Visualization generated successfully") 
                except Exception as vis_error:
                    logger.error(f"Visualization error: {str(vis_error)}")
                    response_data['visualization_error'] = str(vis_error)
            
            
            with torch.no_grad():
                if isinstance(model, CRAFTFontClassifier):
                    # Character model - get embedding and logits
                    # TODO make sure model is expecting HWC 0, 255 input
                    outputs = model(image_tensor)
                    embedding = outputs['font_embedding']
                    logits = outputs['logits']
                else:
                    # Original model approach
                    embedding = model.get_embedding(image_tensor)
                    logits = model.classifier(embedding)
                
                # Get predictions using both methods
                # Embedding similarity approach
                top_k_indices, top_k_similarities = get_top_k_similar_fonts(embedding, k=5)
                
                # Classifier approach
                top_k_cls_indices, top_k_probs = get_top_k_predictions(logits, k=5)
                
                # Format results
                embedding_results = [
                    {
                        'font': label_mapping.get(idx, f'Unknown Font ({idx})'),
                        'similarity': float(score)
                    }
                    for idx, score in zip(top_k_indices, top_k_similarities)
                ]
                
                classifier_results = [
                    {
                        'font': label_mapping.get(idx, f'Unknown Font ({idx})'),
                        'probability': float(prob)
                    }
                    for idx, prob in zip(top_k_cls_indices, top_k_probs)
                ]

                prediction_results = {
                    'embedding_similarity': embedding_results,
                    'classifier_predictions': classifier_results,
                    'research_mode': research_mode
                }
                update_metadata(metadata_path, prediction_results)
                
                return jsonify({
                    'embedding_similarity': embedding_results,
                    'classifier_predictions': classifier_results,
                    **response_data 
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
    """Initialize and run the Flask application when run directly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--data_dir", required=True, help="Directory containing embeddings and label mapping")
    parser.add_argument("--embeddings_path", required=True, help="Path to class_embeddings.npy")
    parser.add_argument("--label_mapping_file", default="label_mapping.npy", help="Label mapping file name")
    parser.add_argument("--uploads_dir", default="uploads", help="Directory to save uploaded images")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    app = create_app(args.model_path, args.data_dir, args.embeddings_path, 
                     args.label_mapping_file, uploads_dir=args.uploads_dir)
    app.run(host='0.0.0.0', port=args.port, debug=True)

if __name__ == '__main__':
    main()