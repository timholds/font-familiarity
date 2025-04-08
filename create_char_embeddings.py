import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
from typing import Tuple

from ml.char_model import CRAFTFontClassifier
from ml.dataset import get_char_dataloaders
from ml.utils import get_embedding_path

def load_char_model(model_path: str) -> Tuple[CRAFTFontClassifier, torch.device]:
    """
    Load the trained character-based font classifier model.
    
    Args:
        model_path: Path to the saved model checkpoint
    Returns:
        model: Loaded model
        device: Torch device (cuda or cpu)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved state
    print(f"Loading model from {model_path}")
    state = torch.load(model_path, map_location=device)
    state_dict = state['model_state_dict']
    
    # Get classifier shape to determine number of classes
    classifier_key = 'font_classifier.font_classifier.weight'
    if classifier_key in state_dict:
        num_fonts = state_dict[classifier_key].shape[0]
        print(f"Model has {num_fonts} font classes")
    else:
        # Debug info if key not found
        print("Available keys containing 'classifier':")
        for key in [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]:
            print(f"- {key}: {state_dict[key].shape}")
        raise ValueError(f"Could not find classifier weights at {classifier_key}")
    
    # Determine embedding dimension
    embedding_keys = [k for k in state_dict.keys() if 'embedding' in k and 'weight' in k]
    if embedding_keys:
        embedding_key = embedding_keys[0]
        # For many embedding layers, output dim is first dimension of weight matrix
        embedding_dim = state_dict[embedding_key].shape[0]
        print(f"Model has embedding dimension {embedding_dim}")
    else:
        # Default value if not found
        embedding_dim = 512
        print(f"Could not determine embedding dimension, using default: {embedding_dim}")
    
    # Initialize model with correct parameters
    model = CRAFTFontClassifier(
        num_fonts=num_fonts,
        device=device,
        patch_size=32,       # Standard character patch size
        embedding_dim=embedding_dim,
        craft_fp16=False     # For wider compatibility
    )
    
    # Load the trained weights
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device

def compute_char_embeddings(
    model: CRAFTFontClassifier, 
    dataloader, 
    num_classes: int, 
    device: torch.device
) -> np.ndarray:
    """
    Compute average embeddings for each font class using the character-based model.
    
    Args:
        model: Trained CRAFTFontClassifier model
        dataloader: DataLoader with the correct format for character models
        num_classes: Number of font classes
        device: Torch device for computation
        
    Returns:
        class_embeddings: Array of shape [num_classes, embedding_dim]
    """
    # Determine embedding dimension from model
    embedding_dim = model.font_classifier.aggregator.projection.out_features
    
    print(f"\nComputing {embedding_dim}-dimensional embeddings for {num_classes} font classes...")
    
    # Initialize storage for embeddings and counts
    class_embeddings = torch.zeros(num_classes, embedding_dim).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Computing embeddings'):
            # Process batch format from the char_collate_fn
            images = batch['images'].to(device)
            targets = batch['labels'].to(device)
            annotations = batch.get('annotations', None)
            
            # Extract character patches
            patch_data = model.extract_patches_with_craft(images)
            patches = patch_data['patches']  
            attention_mask = patch_data['attention_mask']
            
            # Process through font classifier to get embeddings
            outputs = model.font_classifier(patches, attention_mask)
            font_embeddings = outputs['font_embedding']  # [batch_size, embedding_dim]
            
            # Accumulate embeddings by class
            for i, target in enumerate(targets):
                class_idx = target.item()
                class_embeddings[class_idx] += font_embeddings[i]
                class_counts[class_idx] += 1
    
    # Compute averages
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_embeddings[i] /= class_counts[i]
    
    # L2 normalize embeddings for cosine similarity computation
    class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)
    
    # Verify no classes were empty
    empty_classes = (class_counts == 0).sum().item()
    if empty_classes > 0:
        print(f"Warning: {empty_classes} classes had no samples!")
    
    final_embeddings = class_embeddings.cpu().numpy()
    print(f"Final embeddings shape: {final_embeddings.shape}")
    return final_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained character model .pt file")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--embeddings_file", help="Path to save embeddings (optional)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (smaller for char model)")
    args = parser.parse_args()
    
    # Load model
    print("Loading character-based model...")
    model, device = load_char_model(args.model_path)
    
    # Get dataloaders for character-based model
    print("Loading character dataset...")
    test_loader, _, num_classes = get_char_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Compute embeddings
    class_embeddings = compute_char_embeddings(model, test_loader, num_classes, device)
    
    # Save embeddings
    if args.embeddings_file:
        embeddings_path = args.embeddings_file
    else:
        # Default naming using model embedding dimension
        embedding_dim = model.font_classifier.aggregator.projection.out_features
        embeddings_path = os.path.join(
            args.data_dir, 
            f"char_embeddings_{embedding_dim}d.npy"
        )
    
    print(f"\nSaving embeddings to {embeddings_path}")
    np.save(embeddings_path, class_embeddings)
    print("Done!")

if __name__ == "__main__":
    main()