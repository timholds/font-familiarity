import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
from typing import Tuple

from ml.char_model import CRAFTFontClassifier
from ml.dataset import get_char_dataloaders
from ml.utils import get_params_from_model_path, get_embedding_path

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
    hparams = get_params_from_model_path(model_path)

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
    
    # Initialize model with correct parameters
    model = CRAFTFontClassifier(
        num_fonts=num_fonts,
        device=device,  # Pass device but also explicitly move model to device below
        patch_size=hparams["patch_size"],
        embedding_dim=hparams["embedding_dim"],
        initial_channels=hparams["initial_channels"],
        n_attn_heads=hparams["n_attn_heads"],
        craft_fp16=False,
        use_precomputed_craft=use_precomputed_craft,
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
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Computing embeddings'):
            # EXACT MIRROR OF evaluate() FUNCTION FROM train.py
            
            # Move labels to device (should always be present)
            targets = batch_data['labels'].to(device)
            
            # Check if we're using precomputed patches or images
            if 'patches' in batch_data and 'attention_mask' in batch_data:
                # Using precomputed patches - move to device
                batch_data['patches'] = batch_data['patches'].to(device)
                batch_data['attention_mask'] = batch_data['attention_mask'].to(device)
                # DO NOT convert to bool explicitly here - let model handle it
            elif 'images' in batch_data:
                # Using images - move to device
                batch_data['images'] = batch_data['images'].to(device)
            else:
                raise ValueError("Batch data must contain either 'patches' or 'images'")

            # Forward pass with the batch data - EXACTLY as in evaluate()
            outputs = model(batch_data)
            
            # Extract font embeddings (instead of logits as in evaluate())
            if isinstance(outputs, dict) and 'font_embedding' in outputs:
                font_embeddings = outputs['font_embedding']
                print(f"Font embeddings shape: {font_embeddings.shape}")

            else:
                raise ValueError("Model output doesn't contain 'font_embedding'")
            
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
    
    # Check for empty classes
    empty_classes = (class_counts == 0).sum().item()
    if empty_classes > 0:
        print(f"Warning: {empty_classes} classes had no samples!")
    
    return class_embeddings.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained character model .pt file")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
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
        num_workers=os.cpu_count(),
    )
    
    # Compute embeddings
    class_embeddings = compute_char_embeddings(model, test_loader, num_classes, device)
    # TODO move logic into this file for calling the forward pass of the model
    
    
    embeddings_path = get_embedding_path(args.data_dir, args.model_path)
    
    print(f"\nSaving embeddings to {embeddings_path}")
    np.save(embeddings_path, class_embeddings)
    print("Done!")

if __name__ == "__main__":
    main()