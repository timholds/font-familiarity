import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
from typing import Tuple

from ml.char_model import CRAFTFontClassifier
from ml.dataset import get_char_dataloaders
from ml.utils import get_embedding_path

def load_char_model(model_path: str, use_precomputed_craft: bool = False) -> Tuple[CRAFTFontClassifier, torch.device]:
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
        device=device,  # Pass device but also explicitly move model to device below
        patch_size=32,
        embedding_dim=embedding_dim,
        craft_fp16=False,
        use_precomputed_craft=use_precomputed_craft
    )
    
    # Load the trained weights
    model.load_state_dict(state_dict)
    
    # Explicitly move model to device AFTER loading state dict
    model = model.to(device)
    
    # Make sure all sub-components are on the correct device
    for module in model.modules():
        module.to(device)
    
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
    
    # Verify all model components are on the correct device
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Computing embeddings'):
            # Get targets, which should be present in all batches
            targets = batch['labels'].to(device)
            
            try:
                # Check if we're using precomputed patches
                if 'patches' in batch and 'attention_mask' in batch:
                    # Using precomputed patches - move to device
                    patches = batch['patches'].to(device)
                    attention_mask = batch['attention_mask'].to(device).bool()
                    
                    # Process through font classifier directly
                    with torch.cuda.amp.autocast(enabled=False):
                        outputs = model.font_classifier(patches, attention_mask)
                else:
                    # Original code path for processing images with CRAFT
                    if 'images' not in batch:
                        raise ValueError("Batch does not contain 'images' key")
                    
                    images = batch['images'].to(device)
                    
                    # Get patch data using CRAFT
                    patch_data = model.extract_patches_with_craft(images)
                    patches = patch_data['patches'].to(device)
                    attention_mask = patch_data['attention_mask'].to(device).bool()
                    
                    # Process through font classifier
                    with torch.cuda.amp.autocast(enabled=False):
                        outputs = model.font_classifier(patches, attention_mask)
                
                font_embeddings = outputs['font_embedding']  # [batch_size, embedding_dim]
                
                # Accumulate embeddings by class
                for i, target in enumerate(targets):
                    class_idx = target.item()
                    class_embeddings[class_idx] += font_embeddings[i]
                    class_counts[class_idx] += 1
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (smaller for char model)")
    parser.add_argument("--use_precomputed_craft", action="store_true", help="Use precomputed CRAFT boxes from data_dir")

    args = parser.parse_args()
    
    # Load model
    print("Loading character-based model...")
    model, device = load_char_model(args.model_path, use_precomputed_craft=args.use_precomputed_craft)

    
    # Get dataloaders for character-based model
    print("Loading character dataset...")
    test_loader, _, num_classes = get_char_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_precomputed_craft=args.use_precomputed_craft,
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
            f"class_embeddings_{embedding_dim}.npy"
        )
    
    print(f"\nSaving embeddings to {embeddings_path}")
    np.save(embeddings_path, class_embeddings)
    print("Done!")

if __name__ == "__main__":
    main()