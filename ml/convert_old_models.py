#!/usr/bin/env python3
"""
Script to convert old model format (with model_state_dict) to new format (with full model object).
This allows backwards compatibility with models trained before the architecture changes.

Usage:
    python convert_old_models.py --input_model old_model.pt --output_model new_model.pt
    python convert_old_models.py --input_dir models/ --output_dir converted_models/
"""

import argparse
import os
import torch
import shutil
from pathlib import Path
from char_model import CRAFTFontClassifier
from utils import get_params_from_model_path


def convert_single_model(input_path, output_path, device='cuda'):
    """Convert a single model from old format to new format."""
    print(f"Converting {input_path} -> {output_path}")
    
    # Load the old model checkpoint
    checkpoint = torch.load(input_path, map_location=device)
    
    # Check if it's already in new format
    if 'model' in checkpoint and hasattr(checkpoint['model'], 'eval'):
        print(f"  Model {input_path} is already in new format, copying...")
        shutil.copy2(input_path, output_path)
        return
    
    # Check if it has the old format
    if 'model_state_dict' not in checkpoint:
        print(f"  ERROR: Model {input_path} doesn't have expected 'model_state_dict' key")
        return
    
    print(f"  Loading old format model state_dict...")
    state_dict = checkpoint['model_state_dict']
    
    # Get model architecture parameters from filename
    try:
        hparams = get_params_from_model_path(input_path)
        print(f"  Parsed hyperparameters: {hparams}")
    except Exception as e:
        print(f"  ERROR: Could not parse hyperparameters from filename: {e}")
        return
    
    # Determine number of classes from the state dict
    classifier_key = 'font_classifier.font_classifier.weight'
    if classifier_key in state_dict:
        num_fonts = state_dict[classifier_key].shape[0]
        print(f"  Found {num_fonts} font classes")
    else:
        # Try to find classifier key
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
        if classifier_keys:
            classifier_key = classifier_keys[0]
            num_fonts = state_dict[classifier_key].shape[0]
            print(f"  Found alternative classifier at {classifier_key} with {num_fonts} classes")
        else:
            print(f"  ERROR: Could not determine number of font classes from model")
            return
    
    # Reconstruct the model architecture
    print(f"  Reconstructing model architecture...")
    try:
        model = CRAFTFontClassifier(
            num_fonts=num_fonts,
            device=device,
            patch_size=hparams["patch_size"],
            embedding_dim=hparams["embedding_dim"],
            initial_channels=hparams["initial_channels"],
            n_attn_heads=hparams["n_attn_heads"],
            craft_fp16=False,
            use_precomputed_craft=False,
            pad_x=hparams["pad_x"],
            pad_y=hparams["pad_y"],
        )
        
        # Load the weights
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"  Model reconstructed successfully")
        
    except Exception as e:
        print(f"  ERROR: Failed to reconstruct model: {e}")
        return
    
    # Create new checkpoint with full model object
    new_checkpoint = {
        'epoch': checkpoint.get('epoch', 0),
        'model': model,  # Full model object instead of state_dict
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'train_metrics': checkpoint.get('train_metrics'),
        'test_metrics': checkpoint.get('test_metrics'),
        'num_classes': checkpoint.get('num_classes', num_fonts)
    }
    
    # Save the converted model
    print(f"  Saving converted model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_checkpoint, output_path)
    print(f"  ✅ Successfully converted {input_path}")


def convert_directory(input_dir, output_dir, device='cuda'):
    """Convert all .pt files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"ERROR: Input directory {input_dir} does not exist")
        return
    
    # Find all .pt files
    pt_files = list(input_path.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return
    
    print(f"Found {len(pt_files)} .pt files to convert")
    
    # Convert each file
    for pt_file in pt_files:
        output_file = output_path / pt_file.name
        convert_single_model(str(pt_file), str(output_file), device)


def main():
    parser = argparse.ArgumentParser(description="Convert old model format to new format")
    parser.add_argument("--input_model", help="Path to input model file")
    parser.add_argument("--output_model", help="Path to output model file")
    parser.add_argument("--input_dir", help="Directory containing input model files")
    parser.add_argument("--output_dir", help="Directory to save converted model files")
    parser.add_argument("--device", default="cuda", help="Device to use for model loading")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.input_model and args.output_model:
        # Convert single model
        convert_single_model(args.input_model, args.output_model, device)
    elif args.input_dir and args.output_dir:
        # Convert directory
        convert_directory(args.input_dir, args.output_dir, device)
    else:
        print("ERROR: Provide either --input_model and --output_model, or --input_dir and --output_dir")
        parser.print_help()


if __name__ == "__main__":
    main()