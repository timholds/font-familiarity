import torch

# Load v4 checkpoint
checkpoint = torch.load("v4model/fontCNN-BS64-ED1024-IC16-PS64-NH16.pt", map_location='cpu')

print("=== V4 Model Architecture Analysis ===\n")

# Look at the embedding layer structure
embedding_layers = {}
for key in checkpoint['model_state_dict'].keys():
    if 'embedding_layer' in key:
        embedding_layers[key] = checkpoint['model_state_dict'][key].shape

print("Embedding layer architecture in checkpoint:")
for key in sorted(embedding_layers.keys()):
    print(f"  {key}: {embedding_layers[key]}")

# The structure shows:
# embedding_layer.0 is a Linear layer: [1024, 2048] 
# There's no embedding_layer.2 in the checkpoint!
# But the model being created expects embedding_layer.2

print("\n=== Full layer structure ===")
# Let's see what layers exist
for key in sorted(checkpoint['model_state_dict'].keys())[:20]:
    print(f"  {key}: {checkpoint['model_state_dict'][key].shape}")
