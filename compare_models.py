import argparse
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import logging
import traceback
import json
from ml.char_model import CRAFTFontClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def load_model(model_path, embeddings_path):
    """Load character-based model and its embeddings."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model state
        state = torch.load(model_path, map_location=device)
        state_dict = state['model_state_dict']
        
        # Determine number of classes from classifier
        classifier_key = 'font_classifier.font_classifier.weight'
        if classifier_key in state_dict:
            num_fonts = state_dict[classifier_key].shape[0]
        else:
            # Try to find classifier key
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
            if classifier_keys:
                classifier_key = classifier_keys[0]
                num_fonts = state_dict[classifier_key].shape[0]
            else:
                raise ValueError("Could not determine number of font classes from model")
        
        # Determine embedding dimension
        embedding_dim = 512  # Default fallback
        for key in state_dict.keys():
            if 'projection' in key and 'weight' in key:
                embedding_dim = state_dict[key].shape[0]
                break
        
        # Initialize model
        model = CRAFTFontClassifier(
            num_fonts=num_fonts,
            device=device,
            patch_size=32, 
            embedding_dim=embedding_dim,
            craft_fp16=False,
            use_precomputed_craft=False
        )
        
        # Load weights and set to eval mode
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Load embeddings
        class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
        
        return model, class_embeddings, device
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def process_image(image_path, device):
    """Load and process image for model inference."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        # Convert to numpy array
        image_np = np.array(image)
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def compare_models(image_path, model_paths, embeddings_paths, label_mapping, k=5, output_file=None):
    """Compare predictions from two models on the same image."""
    # Load label mapping
    label_map = np.load(label_mapping, allow_pickle=True).item()
    # Invert mapping from index -> font name
    labels = {v: k for k, v in label_map.items()}
    
    # Load models
    model1, embeddings1, device = load_model(model_paths[0], embeddings_paths[0])
    model2, embeddings2, device = load_model(model_paths[1], embeddings_paths[1])
    
    # Process image
    image_tensor = process_image(image_path, device)
    
    # Get predictions from models
    with torch.no_grad():
        # Model 1
        outputs1 = model1(image_tensor)
        embedding1 = outputs1['font_embedding']
        logits1 = outputs1['logits']
        
        # Model 2
        outputs2 = model2(image_tensor)
        embedding2 = outputs2['font_embedding']
        logits2 = outputs2['logits']
        
        # Get similarities and predictions
        indices1_emb, scores1_emb = get_top_k_similar_fonts(embedding1, embeddings1, k)
        indices2_emb, scores2_emb = get_top_k_similar_fonts(embedding2, embeddings2, k)
        
        indices1_cls, scores1_cls = get_top_k_predictions(logits1, k)
        indices2_cls, scores2_cls = get_top_k_predictions(logits2, k)
    
    # Format results
    results = {
        'image_path': image_path,
        'model1': {
            'name': os.path.basename(model_paths[0]),
            'embedding_similarity': [
                {'font': labels.get(int(idx), f'Unknown Font ({idx})'), 'score': float(score)}
                for idx, score in zip(indices1_emb, scores1_emb)
            ],
            'classifier_predictions': [
                {'font': labels.get(int(idx), f'Unknown Font ({idx})'), 'score': float(score)}
                for idx, score in zip(indices1_cls, scores1_cls)
            ],
        },
        'model2': {
            'name': os.path.basename(model_paths[1]),
            'embedding_similarity': [
                {'font': labels.get(int(idx), f'Unknown Font ({idx})'), 'score': float(score)}
                for idx, score in zip(indices2_emb, scores2_emb)
            ],
            'classifier_predictions': [
                {'font': labels.get(int(idx), f'Unknown Font ({idx})'), 'score': float(score)}
                for idx, score in zip(indices2_cls, scores2_cls)
            ],
        }
    }
    
    # Print results
    print("\nClassifier Predictions:")
    print(f"{'Model 1: ' + results['model1']['name']:<40} {'Model 2: ' + results['model2']['name']:<40}")
    print("-" * 80)
    for i in range(k):
        font1 = results['model1']['classifier_predictions'][i]['font']
        score1 = results['model1']['classifier_predictions'][i]['score']
        font2 = results['model2']['classifier_predictions'][i]['font']
        score2 = results['model2']['classifier_predictions'][i]['score']
        print(f"{font1:<30} {score1:.4f}    {font2:<30} {score2:.4f}")
    
    print("\nEmbedding Similarity:")
    print(f"{'Model 1: ' + results['model1']['name']:<40} {'Model 2: ' + results['model2']['name']:<40}")
    print("-" * 80)
    for i in range(k):
        font1 = results['model1']['embedding_similarity'][i]['font']
        score1 = results['model1']['embedding_similarity'][i]['score']
        font2 = results['model2']['embedding_similarity'][i]['font']
        score2 = results['model2']['embedding_similarity'][i]['score']
        print(f"{font1:<30} {score1:.4f}    {font2:<30} {score2:.4f}")
    
    # Save results if output file is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results

def main():
    """Parse command line arguments and run model comparison."""
    parser = argparse.ArgumentParser(description="Compare character-based font prediction models")
    
    # Required arguments
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--label_mapping", required=True, help="Path to shared label mapping file")
    
    # Model paths
    parser.add_argument("--model1_path", required=True, help="Path to first model .pt file")
    parser.add_argument("--embeddings1_path", required=True, help="Path to first model embeddings")
    
    parser.add_argument("--model2_path", required=True, help="Path to second model .pt file")
    parser.add_argument("--embeddings2_path", required=True, help="Path to second model embeddings")
    
    # Optional arguments
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--output_file", help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Run comparison
    compare_models(
        args.image_path, 
        [args.model1_path, args.model2_path],
        [args.embeddings1_path, args.embeddings2_path],
        args.label_mapping,
        args.top_k, 
        args.output_file
    )

if __name__ == "__main__":
    main()