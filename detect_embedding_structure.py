import torch

def analyze_embedding_structure(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Count embedding layers
    embedding_layers = [k for k in state_dict.keys() if 'embedding_layer' in k and 'weight' in k]
    
    if len(embedding_layers) == 1:
        # Single linear layer (v4 style)
        weight_shape = state_dict['font_classifier.char_encoder.embedding_layer.0.weight'].shape
        print(f"Single embedding layer: {weight_shape[1]} -> {weight_shape[0]}")
        return "single", weight_shape[0]  # embedding_dim
    else:
        # Multi-layer (older style)
        print(f"Multi-layer embedding: {len(embedding_layers)} linear layers")
        return "multi", None

# Check both models
print("V4 model:")
analyze_embedding_structure("v4model/fontCNN-BS64-ED1024-IC16-PS64-NH16.pt")

print("\nTest model:")  
analyze_embedding_structure("test-data/font-dataset-npz/fontCNN_BS64-ED256-IC16.pt")
