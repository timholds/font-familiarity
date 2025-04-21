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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        state_dict = state['model_state_dict']
        
        # Determine number of classes from classifier
        classifier_key = 'font_classifier.font_classifier.weight'
        if classifier_key in state_dict:
            num_fonts = state_dict[classifier_key].shape[0]
            logger.info(f"Model has {num_fonts} font classes")
        else:
            # Try to find classifier key
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
            if classifier_keys:
                classifier_key = classifier_keys[0]
                num_fonts = state_dict[classifier_key].shape[0]
                logger.info(f"Found alternative classifier at {classifier_key} with {num_fonts} classes")
            else:
                raise ValueError("Could not determine number of font classes from model")
        
        # Determine embedding dimension - look at projection layer in aggregator
        embedding_dim = 512  # Default fallback
        for key in state_dict.keys():
            if 'projection' in key and 'weight' in key:
                embedding_dim = state_dict[key].shape[0]
                logger.info(f"Found embedding dimension: {embedding_dim}")
                break
        
        # Initialize model
        # TODO pull this from state_dict
        model = CRAFTFontClassifier(
            num_fonts=num_fonts,
            device=device,
            patch_size=32, 
            embedding_dim=embedding_dim,
            craft_fp16=False,  # Conservative setting for production
            use_precomputed_craft=False
        )
        
        # Load the weights
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
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

def create_app(model_path=None, data_dir=None, embeddings_path=None, label_mapping_file=None, use_char_model=True):
    """Factory function to create and configure Flask app instance."""
    app = Flask(__name__)

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
            
            # Convert uploaded image bytes to tensor 
            # Todo this needs to match the dataloader
            original_image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Original image mode: {original_image.mode}, size: {original_image.size}")
            image = original_image.convert('RGB')
            logger.info(f"Converted image mode: {image.mode}, size: {image.size}")
            
            # do i need to creat a numpy tensor or can torch take in pil iamge 

            # TODO get rid of totensor and pass hwc 0,255 since we 
            # gotta extract patches with craft
            # transform = transforms.Compose([
            #     transforms.Resize((512, 512)),
            #     #transforms.PILToTensor() 
            #     #transforms.ToTensor()
            # ])
            # convert pil image to numpy
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
                    print(f"finsiedh vis craft detections, image type: {type(visual_image)}")

                    # from PIL import ImageDraw, ImageFont

                    # visual_image = original_image.copy()
                    # draw = ImageDraw.Draw(visual_image)
                    
                    # # Try to get polygons from CRAFT
                    # logger.info("Getting polygons from CRAFT")
                    # polygons = model.craft.get_polygons(original_image)
                    # logger.info(f"Found {len(polygons)} polygons")
                    
                    # # Draw polygons on the image
                    # for poly in polygons:
                    #     # Convert to tuple format for PIL
                    #     poly_tuple = [tuple(p) for p in poly]
                    #     draw.polygon(poly_tuple, outline=(255, 0, 0), width=2)
                    
                    # # Add text showing number of detections
                    # try:
                    #     font = ImageFont.truetype("arial.ttf", 20)
                    # except:
                    #     font = ImageFont.load_default()
                    
                    # draw.text((10, 10), f"CRAFT Detections: {len(polygons)} characters", 
                    #         fill=(0, 0, 0), font=font)
                        
                        
                    # Convert visualization to base64
                    import io as bio
                    import base64
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
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    app = create_app(args.model_path, args.data_dir, args.embeddings_path, args.label_mapping_file)
    app.run(host='0.0.0.0', port=args.port, debug=True)

if __name__ == '__main__':
    main()