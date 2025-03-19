import torch
from torch import nn

import torch
import torch.nn as nn
import torchvision.transforms as transforms


# Todo update this to take in character patches
# Todo aggregation of the patches in 
class CharSimpleCNN(nn.Module):
    """
    Basic CNN architecture for font classification.
    Input: (batch_size, 1, 32, 32)
    """
    def __init__(self, num_classes: int, 
                 in_channels: int = 1,
                 input_size: int = 32,
                 embedding_dim: int = 256,
                 initial_channels: int = 16):

        super().__init__() 
        self.input_size = input_size   
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization for grayscale images
        ])
        # Build feature layers with single conv per block
        layers = []
        curr_channels = in_channels
        next_channels = initial_channels
        for _ in range(4):  # 4 downsample blocks
            layers.extend([
                nn.Conv2d(curr_channels, next_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(next_channels, next_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            curr_channels = next_channels
            next_channels *= 2  # Double channels after each block
            
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
       
        # Calculate flattened dim dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_size, self.input_size)
            dummy_output = self.features(dummy_input)
            self.flatten_dim = self.flatten(dummy_output).shape[1]
        

        # Calculate dimensions by hand:
        # resolution /(2^num_downsamples)
        # After resizing to 64x64 and going through the CNN layers:
        # Ex: input 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # Final channels = 256
        # Final feature map 256 * 4 * 4 = 4096
        # TODO clean this up 
        #self.flatten_dim = 128 * 4 * 4 

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
    
        
        self.classifier = nn.Linear(embedding_dim, num_classes)


    def get_embedding(self, x):
        x = self.transform(x)
        x = self.features(x)
        x = self.flatten(x)
        embeddings = self.embedding_layer(x)
        return embeddings

    def forward(self, x):
        embeddings = self.get_embedding(x)
        x = self.classifier(embeddings)
        return x