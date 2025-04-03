# Create a model name
import os
from pathlib import Path

def get_model_path(base_dir, prefix, batch_size, embedding_dim, initial_channels):
    model_name = f"{prefix}_BS{batch_size}-ED{embedding_dim}-IC{initial_channels}.pt"
    model_path = Path(os.path.join(base_dir, model_name))
    return model_path


def get_embedding_path(base_dir, embedding_file=None, embedding_dim=None):
    if embedding_file is not None:
        return Path(os.path.join(base_dir, embedding_file))
    else:
        # TODO make sure this still works when embedding_dim is None
        embedding_file = f"class_embeddings_{embedding_dim}.npy"
        return Path(os.path.join(base_dir, embedding_file))

def check_char_model_batch_independence(model, batch_size=4, max_chars=10, 
                                       target_index=2, device='cuda'):
    """
    Tests whether the CharacterBasedFontClassifier properly maintains batch 
    independence by verifying gradients only flow to the selected example.
    
    Args:
        model: The CharacterBasedFontClassifier model to test
        batch_size: Number of samples in the batch
        max_chars: Maximum number of character patches per sample
        target_index: Which batch sample to compute loss from
        device: Device to run test on
        
    Returns:
        True if test passes (no batch leakage)
    """
    import torch
    
    # Put model in eval mode to disable dropout, batch norm, etc.
    model.eval()
    
    # Create a synthetic batch of character patches
    # Shape: [batch_size, max_chars, 1, patch_size, patch_size]
    patch_size = model.char_encoder.input_size
    char_patches = torch.rand(batch_size, max_chars, 1, 
                             patch_size, patch_size, 
                             requires_grad=True, device=device)
    
    # Create attention mask (all ones = all patches are valid)
    attention_mask = torch.ones(batch_size, max_chars, device=device)
    
    # Forward pass
    outputs = model(char_patches, attention_mask)
    logits = outputs['logits']
    
    # Set loss to depend only on target example
    loss = logits[target_index].sum()
    
    # Backpropagate
    loss.backward()
    
    # Check that only target_index has non-zero gradients
    passed = True
    for i in range(batch_size):
        if i == target_index:
            # Target should have non-zero gradients
            if not (char_patches.grad[i] != 0).any():
                print(f"❌ FAIL: Target sample {i} has no gradients")
                passed = False
        else:
            # Non-targets should have zero gradients
            if not (char_patches.grad[i] == 0).all():
                print(f"❌ FAIL: Sample {i} has gradient leakage from target")
                passed = False
    
    if passed:
        print("✅ SUCCESS: No batch leakage detected!")
    
    return passed