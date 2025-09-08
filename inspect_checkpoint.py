import torch
import sys

def inspect_checkpoint(path):
    print(f"\n=== Inspecting checkpoint: {path} ===\n")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # Check top-level keys
    print("Top-level keys:", list(checkpoint.keys()))
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n=== Model State Dict Analysis ===")
        
        # Look for embedding layer info
        embedding_layers = [k for k in state_dict.keys() if 'embedding_layer' in k]
        print(f"\nEmbedding layer keys ({len(embedding_layers)} total):")
        for key in sorted(embedding_layers)[:10]:  # Show first 10
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
        
        # Look for classifier info
        classifier_layers = [k for k in state_dict.keys() if 'classifier' in k]
        print(f"\nClassifier layer keys ({len(classifier_layers)} total):")
        for key in sorted(classifier_layers)[:10]:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
            
        # Look for attention info
        attn_layers = [k for k in state_dict.keys() if 'attn' in k or 'attention' in k]
        print(f"\nAttention layer keys ({len(attn_layers)} total):")
        for key in sorted(attn_layers)[:10]:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
    
    # Check for metadata
    if 'num_classes' in checkpoint:
        print(f"\nnum_classes in checkpoint: {checkpoint['num_classes']}")
    if 'embedding_dim' in checkpoint:
        print(f"embedding_dim in checkpoint: {checkpoint['embedding_dim']}")
    if 'model_config' in checkpoint:
        print(f"\nModel config: {checkpoint['model_config']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        # Inspect both models
        inspect_checkpoint("v4model/fontCNN-BS64-ED1024-IC16-PS64-NH16.pt")
        print("\n" + "="*60)
        inspect_checkpoint("test-data/font-dataset-npz/fontCNN_BS64-ED256-IC16.pt")
