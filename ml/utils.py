# Create a model name
import os
from pathlib import Path

def get_model_path(base_dir, prefix, batch_size, embedding_dim, initial_channels,
                   patch_size=32, n_attn_heads=4, pad_x=.05, pad_y=.15):
    
    model_name = (
        f"{prefix}-BS{batch_size}-ED{embedding_dim}-IC{initial_channels}-"
        f"PS{patch_size}-NH{n_attn_heads}-PX{pad_x}-PY{pad_y}.pt"
    )
    model_path = Path(os.path.join(base_dir, model_name))
    return model_path


# def get_embedding_path(base_dir, embedding_file=None, embedding_dim=None):
#     if embedding_file is not None:
#         return Path(os.path.join(base_dir, embedding_file))
#     else:
#         # TODO make sure this still works when embedding_dim is None
#         embedding_file = f"class_embeddings_{embedding_dim}.npy"
#         return Path(os.path.join(base_dir, embedding_file))

def get_embedding_path(base_dir, model_path):
    """
    Get the path to the class embeddings file based on the model path.
    Args:
        base_dir: Base directory where the embeddings are stored
        model_path: Path to the model file
    Returns:
        Path to the class embeddings file inside the base_dir
    
    """
    hparams = get_params_from_model_path(model_path)
    embedding_file = (
        f"class_embeddings-BS{hparams['batch_size']}-ED{hparams['embedding_dim']}-"
        f"IC{hparams['initial_channels']}-PS{hparams['patch_size']}-"
        f"NH{hparams['n_attn_heads']}.npy"
    )
    embedding_path = Path(os.path.join(base_dir, embedding_file))
    # Check if the embedding file already exists
    if embedding_path.exists():
        print(f"Embedding file already exists: {embedding_path}")
        return embedding_path
    else:
        return embedding_path
    


def get_params_from_model_path(model_path):
    """ Get the hyperparameters out of a model filename
    Args:
        model_path: Path to the model file
        ex: fontCNN_BS64-ED1024-IC16-PS64-NH16.pt
    Returns dict with keys:
        batch_size: Batch size used for training
        embedding_dim: Embedding dimension used for training
        initial_channels: Initial channels used for training
        patch_size: Patch size used for training
        nheads: Number of self-attention heads 
        LR: Learning rate used for training
        WD: Weight decay used for training
    """

    # Default values in case they are not in the filename
    params = {
        "batch_size": 64,
        "embedding_dim": 1024,
        "initial_channels": 16,
        "patch_size": 32,
        "n_attn_heads": 4,
        "learning_rate": 0.00005,
        "weight_decay": 0.001,
        "pad_x": .05,
        "pad_y": .15,
    }

    model_name = os.path.basename(model_path)
    model_name = os.path.splitext(model_name)[0]  # Remove the .pt extension

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("BS"):
            params["batch_size"] = int(part[2:])
        elif part.startswith("ED"):
            params["embedding_dim"] = int(part[2:])
        elif part.startswith("IC"):
            params["initial_channels"] = int(part[2:])
        elif part.startswith("PS"):
            params["patch_size"] = int(part[2:])
        elif part.startswith("NH"):
            params["n_attn_heads"] = int(part[2:])
        elif part.startswith("LR"):
            params["learning_rate"] = float(part[2:])
        elif part.startswith("WD"):
            params["weight_decay"] = float(part[2:])
        elif part.startswith("PX"):
            params["pad_x"] = float(part[2:])
        elif part.startswith("PY"):
            params["pad_y"] = float(part[2:])

    return params



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