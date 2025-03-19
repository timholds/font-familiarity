import os
import json
import cv2
import torch
from torch.utils.data import Dataset
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
    
class SelfAttentionAggregator(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Character embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Boolean mask [batch_size, seq_len] where True values are masked
        """
        # Convert mask to proper format for nn.MultiheadAttention
        if attention_mask is not None:
            # 1 means keep, 0 means mask out
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
            
        # Self-attention
        attn_output, attn_weights = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        
        # Normalize
        attn_output = self.norm(attn_output + x)  # Add residual connection
        
        # Aggregate sequence - take mean of unmasked elements
        if attention_mask is not None:
            # Expand attention mask for broadcasting
            expanded_mask = attention_mask.unsqueeze(-1)
            # Sum embeddings where mask is 1, divide by count of 1s
            masked_sum = (attn_output * expanded_mask).sum(dim=1)
            mask_sum = expanded_mask.sum(dim=1) + 1e-8  # Avoid division by zero
            aggregated = masked_sum / mask_sum
        else:
            # Simple mean if no mask
            aggregated = attn_output.mean(dim=1)
            
        # Final projection
        aggregated = self.projection(aggregated)
        
        return aggregated, attn_weights
    
class CharacterBasedFontClassifier(nn.Module):
    def __init__(self, num_fonts, char_size=32, embedding_dim=256):
        super().__init__()
        
        # Character feature extractor - reuse existing CNN architecture
        self.char_encoder = CharSimpleCNN(
            num_classes=embedding_dim,  # Use as feature extractor
            input_size=char_size,
            embedding_dim=embedding_dim
        )
        
        # Replace classifier with identity to get embeddings only
        self.char_encoder.classifier = nn.Identity()
        
        # Self-attention aggregation
        self.aggregator = SelfAttentionAggregator(embedding_dim)
        
        # Final font classifier
        self.font_classifier = nn.Linear(embedding_dim, num_fonts)
        
    def forward(self, char_patches, attention_mask=None):
        """
        Args:
            char_patches: Character images [batch_size, max_chars, 1, H, W]
            attention_mask: Mask for padding [batch_size, max_chars]
        """
        batch_size, max_chars = char_patches.shape[:2]
        
        # Reshape to process all characters at once
        flat_patches = char_patches.view(-1, 1, char_patches.shape[3], char_patches.shape[4])
        
        # Get character embeddings
        char_embeddings = self.char_encoder.get_embedding(flat_patches)
        
        # Reshape back to [batch_size, max_chars, embedding_dim]
        char_embeddings = char_embeddings.view(batch_size, max_chars, -1)
        
        # Aggregate character embeddings with attention
        font_embedding, attention_weights = self.aggregator(char_embeddings, attention_mask)
        
        # Classify font
        logits = self.font_classifier(font_embedding)
        
        return {
            'logits': logits,
            'font_embedding': font_embedding,
            'attention_weights': attention_weights,
            'char_embeddings': char_embeddings
        }