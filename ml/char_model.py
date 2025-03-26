import os
import json
import cv2
import torch
from PIL import Image

from torch.utils.data import Dataset
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF


from CRAFT import CRAFTModel, draw_polygons
import numpy as np
from PIL import Image
from tqdm import tqdm


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
        print(f"Input shape to get_embedding: {x.shape}")

        x = x / 255.0

        # Always force resize using F.interpolate (skip the transforms.Resize step)
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            print(f"Resizing from {x.shape[-2]}x{x.shape[-1]} to {self.input_size}x{self.input_size}")
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        # Apply only normalization from transform
        # x = F.normalize(x, mean=[0.5], std=[0.5])
        x = TF.normalize(x, mean=[0.5], std=[0.5])


        x = self.features(x)
        print(f"After features, shape: {x.shape}")
        x = self.flatten(x)
    
        print(f"After flatten, shape: {x.shape}, expected flatten_dim: {self.flatten_dim}")
        embeddings = self.embedding_layer(x)
        print(f"After embedding_layer, shape: {embeddings.shape}")

        
        return embeddings
    
    def forward(self, x):
        x = x / 255.0 
        embeddings = self.get_embedding(x)
        print(f"ebmedding shape {embeddings.shape}")
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

        print(f"Self-attention input shape: {x.shape}, embedding_dim: {self.multihead_attn.embed_dim}")
        if x.shape[-1] != self.multihead_attn.embed_dim:
            raise ValueError(f"Expected embedding dim {self.multihead_attn.embed_dim}, got {x.shape[-1]}")
            
            
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
    def __init__(self, num_fonts, patch_size=32, embedding_dim=256):
        super().__init__()
        
        # Character feature extractor - reuse existing CNN architecture
        self.char_encoder = CharSimpleCNN(
            num_classes=embedding_dim,  # Use as feature extractor
            input_size=patch_size,
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
        print(f"CharacterBasedFontClassifier input patches shape: {char_patches.shape}")

        if len(char_patches.shape) > 5:
            print(f"Fixing unexpected shape: {char_patches.shape}")
            # Reshape to [batch_size, max_chars, channels, H, W]
            batch_size = char_patches.shape[0]
            max_chars = char_patches.shape[1]
            # Get the channels from the last dimension
            channels = char_patches.shape[-1]
            print(f"batch_size: {batch_size}, max_chars: {max_chars}, channels: {channels}")
            char_patches = char_patches.reshape(batch_size, max_chars, channels, 
                                                char_patches.shape[3], char_patches.shape[4])
            print(f"Reshaped to: {char_patches.shape}")
        
        batch_size, max_chars = char_patches.shape[:2]
        # Reshape to process all characters at once - flatten batch and max_chars dimensions
        flat_patches = char_patches.reshape(-1, char_patches.shape[2], 
                                            char_patches.shape[3], char_patches.shape[4])
        print(f"Flattened patches shape: {flat_patches.shape}")


        # Get character embeddings
        char_embeddings = self.char_encoder.get_embedding(flat_patches)
        print(f"Raw embeddings shape: {char_embeddings.shape}")

        # expected_embedding_dim = self.aggregator.multihead_attn.embed_dim
        # if char_embeddings.shape[1] != expected_embedding_dim:
        #     print(f"WARNING: Embedding dimension mismatch. Got {char_embeddings.shape[1]}, expected {expected_embedding_dim}")
        #     # Use a linear projection to fix the dimension
        #     projection = nn.Linear(char_embeddings.shape[1], expected_embedding_dim).to(char_embeddings.device)
        #     char_embeddings = projection(char_embeddings)


        # Reshape back to [batch_size, max_chars, embedding_dim]
        char_embeddings = char_embeddings.view(batch_size, max_chars, -1)
        print(f"Reshaped embeddings shape: {char_embeddings.shape}")
        # Aggregate character embeddings with attention
        font_embedding, attention_weights = self.aggregator(char_embeddings, attention_mask)
        print(f"Font embedding shape: {font_embedding.shape}, attention_weights shape: {attention_weights.shape}")
        # Classify font
        logits = self.font_classifier(font_embedding)
        print(f"Logits shape: {logits.shape}")
        
        return {
            'logits': logits,
            'font_embedding': font_embedding,
            'attention_weights': attention_weights,
            'char_embeddings': char_embeddings
        }
    
class CRAFTFontClassifier(nn.Module):
    """
    Combined model that uses CRAFT for text detection and CharacterBasedFontClassifier for font identification.
    During training, uses provided annotations for character extraction.
    During inference, uses CRAFT to extract character patches.
    """
    def __init__(self, num_fonts, craft_weights_dir='weights/', device='cuda', 
                 patch_size=32, embedding_dim=256, craft_fp16=True):
        super().__init__()
        # Initialize CRAFT model for text detection
        self.craft = CRAFTModel(
            cache_dir=craft_weights_dir,
            device=device,
            use_refiner=True,
            fp16=craft_fp16
        )
        
        # Initialize the font classifier
        self.font_classifier = CharacterBasedFontClassifier(
            num_fonts=num_fonts,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )
        
        self.device = device
        self.patch_size = patch_size
        
    def visualize_char_preds(self, patches, attention_mask, predictions=None, targets=None, save_path=None):
        """
        Visualize character patches (for debugging)
        
        Args:
            patches: Character patches [batch_size, max_patches, C, H, W]
            attention_mask: Mask indicating valid patches [batch_size, max_patches]
            predictions: Optional model predictions [batch_size]
            targets: Optional ground truth labels [batch_size]
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        import os
        
        batch_size = min(4, patches.size(0))  # Visualize up to 4 samples
        
        for b in range(batch_size):
            # Create figure
            fig = plt.figure(figsize=(12, 6))
            
            # Set title with prediction info if available
            title = "Character Patches"
            if predictions is not None and targets is not None:
                pred_label = predictions[b].item()
                true_label = targets[b].item()
                title += f" - Pred: {pred_label} | True: {true_label}"
                
            plt.suptitle(title)
            
            # Count valid patches for this sample
            num_valid = int(attention_mask[b].sum().item())
            grid_size = int(np.ceil(np.sqrt(num_valid)))
            
            # Plot each valid patch
            valid_idx = 0
            for p in range(patches.size(1)):
                if attention_mask[b, p] == 0:
                    continue
                    
                # Get patch
                patch = patches[b, p, 0].cpu().numpy()  # First channel
                
                # Add subplot
                ax = fig.add_subplot(grid_size, grid_size, valid_idx + 1)
                ax.imshow(patch, cmap='gray')
                ax.axis('off')
                
                valid_idx += 1
                
                if valid_idx >= grid_size * grid_size:
                    break
            
            plt.tight_layout()
            
            # Save figure
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(f"{save_path}_sample_{b}.png")
                plt.close()
            else:
                plt.show()

    def visualize_craft_detections(self, images, save_path=None):
        """
        Visualize CRAFT character detections on original images

        Args:
            images: Tensor of shape [batch_size, channels, height, width]
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import os

        batch_size = min(4, images.size(0))  # Visualize up to 4 samples

        for b in range(batch_size):
            print(f"visualize_craft_detections input Image {b} shape: {images[b].shape}, min: {images[b].min()}, max: {images[b].max()}")
            # Convert image to numpy and prepare for visualization
            # check if the image is in CHW format
            if len(images[b].shape) == 3 and images[b].shape[0] in [1, 3]:
                img = images[b].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            else:
                img = images[b].cpu().numpy() # HWC
            #img = (img * 255).astype(np.uint8)
            print(f"visualize_craft_detections Image {b} shape: {img.shape}, min: {img.min()}, max: {img.max()}")
            
            # # Handle grayscale/RGB
            # if len(img.shape) == 2:
            #     rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # elif img.shape[-1] == 1:
            #     rgb_img = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)
            # else:
            rgb_img = img.astype(np.uint8)
                
            # Convert to PIL for CRAFT
            from PIL import Image
            print(f"visualize_craft_detections rgb_img {b} shape: {rgb_img.shape}, min: {rgb_img.min()}, max: {rgb_img.max()}")
            pil_img = Image.fromarray(rgb_img)

            print('!!!!!!!!!!!')
            print(f"Image {b} shape: {rgb_img.shape}, min: {rgb_img.min()}, max: {rgb_img.max()}")
            print(f"PIL image size: {pil_img.size}, mode: {pil_img.mode}")

            
            # Get polygons from CRAFT
            try:
                polygons = self.craft.get_polygons(pil_img)
            except Exception as e:
                print(f"CRAFT detection error: {e}")
                polygons = []
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_img)
            
            # Draw polygons
            for poly in polygons:
                # Convert to numpy array for matplotlib
                poly_array = np.array(poly)
                # Create polygon patch
                patch = mpatches.Polygon(poly_array, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(patch)
            
            ax.set_title(f"CRAFT Detections: {len(polygons)} characters")
            ax.axis('off')
            
            # Save or show
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(f"{save_path}_craft_sample_{b}.png")
                plt.close()
            else:
                plt.show()

    def extract_patches_from_annotations(self, images, targets, annotations):
        """
        Extract character patches using ground truth annotations during training
        
        Args:
            images: Tensor of shape [batch_size, channels, height, width]
            targets: Font class targets
            annotations: List of character annotations for each image
            
        Returns:
            Dictionary with patches, attention_mask and targets
        """
        batch_size = images.size(0)
        all_patches = []
        attention_masks = []
        
        for i in range(batch_size):
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            img = (img * 255).astype(np.uint8)
            
            # Get annotations for this image
            char_anns = annotations[i]
            height, width = img.shape[:2]
            
            # Extract patches for this image
            img_patches = []
            for ann in char_anns:
                # Extract coordinates from annotation (YOLO format: class_id, x_center, y_center, w, h)
                class_id, x_center, y_center, w, h = ann
                
                # Convert to pixel coordinates
                x_center = float(x_center) * width
                y_center = float(y_center) * height
                w = float(w) * width
                h = float(h) * height
                
                # Calculate corner coordinates
                x1 = max(0, int(x_center - w/2))
                y1 = max(0, int(y_center - h/2))
                x2 = min(width, int(x_center + w/2))
                y2 = min(height, int(y_center + h/2))
                
                # Extract patch
                if x2-x1 > 2 and y2-y1 > 2:  # Ensure minimum size
                    patch = img[y1:y2, x1:x2].copy()
                    
                    # Resize and normalize
                    patch = self._normalize_patch(patch)
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Add channel dimension
                    print(f"Patch shape: {patch.shape}, tensor shape: {patch_tensor.shape}")
                    img_patches.append(patch_tensor)
            
            # If no valid patches, create a default patch from the whole image
            if not img_patches:
                img_resized = cv2.resize(img, (self.patch_size, self.patch_size))
                img_norm = img_resized.astype(np.float32) / 255.0
                patch_tensor = torch.from_numpy(img_norm).float().unsqueeze(0)
                img_patches = [patch_tensor]
            
            # Stack patches for this image and create attention mask
            img_patches_tensor = torch.stack(img_patches)
            all_patches.append(img_patches_tensor)
            attention_masks.append(torch.ones(len(img_patches)))
        
        # Pad to same number of patches in batch
        max_patches = max(p.size(0) for p in all_patches)
        padded_patches = []
        padded_masks = []
        
        for patches, mask in zip(all_patches, attention_masks):
            if patches.size(0) < max_patches:
                padding = torch.zeros(
                    (max_patches - patches.size(0), 1, self.patch_size, self.patch_size), 
                    dtype=patches.dtype, device=patches.device
                )
                padded = torch.cat([patches, padding], dim=0)
                
                # Extend mask
                pad_mask = torch.cat([
                    mask, 
                    torch.zeros(max_patches - mask.size(0), device=mask.device)
                ])
            else:
                padded = patches
                pad_mask = mask
                
            padded_patches.append(padded)
            padded_masks.append(pad_mask)
        
        # Stack into batch tensors
        patches_batch = torch.stack(padded_patches).to(self.device)
        attention_batch = torch.stack(padded_masks).to(self.device)
        targets_batch = targets.to(self.device)
        
        return {
            'patches': patches_batch,
            'attention_mask': attention_batch,
            'labels': targets_batch
        }
    
    def extract_patches_with_craft(self, images):
        batch_size = images.size(0)
        all_patches = []
        attention_masks = []
        print(f"Starting CRAFT extraction with images shape: {images.shape}")

        for i in range(batch_size):
            # Step 1: Convert tensor to numpy with proper format handling
            img_tensor = images[i].cpu()
            print(f"Image {i} shape: {img_tensor.shape}, min: {img_tensor.min()}, max: {img_tensor.max()}")
            
            img_np = img_tensor.permute(1, 2, 0).numpy()  # Convert CHW to HWC

            # Ensure correct RGB format for CRAFT
            if img_np.shape[2] == 1:
                img_np = cv2.cvtColor(img_np.squeeze(2), cv2.COLOR_GRAY2RGB)
            
            # Ensure uint8 range
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Convert to PIL for CRAFT
            pil_img = Image.fromarray(img_np)
            print(f"Processing image {i}, PIL image size: {pil_img.size}, mode: {pil_img.mode}")

            # print(f"Processing image {i}, shape: {img_np.shape} into pil image: {pil_img.size}")
        
            # Get polygons from CRAFT
            try:
                print("getting polygons")
                polygons = self.craft.get_polygons(pil_img)
                print(f"Found {len(polygons)} characters in image {i}")
            except Exception as e:
                print(f"CRAFT error: {e}")
                polygons = []
            
            # Extract character patches
            img_patches = []
            for polygon in polygons:
                # Convert polygon to bounding box
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                # Ensure integer coordinates and minimum size
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img_np.shape[1], int(x2)), min(img_np.shape[0], int(y2))
                
                # Skip very small regions
                if x2-x1 < 3 or y2-y1 < 3:
                    continue
                
                # Extract patch
                patch = img_np[y1:y2, x1:x2].copy()
                print(f"In patch extraction with craft, Character patch shape: {patch.shape}")
                
                # Convert to grayscale
                if len(patch.shape) == 3 and patch.shape[2] == 3:
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                
                # Resize and normalize (returns a 2D array)
                normalized_patch = self._normalize_patch(patch)
                
                # Convert to tensor with channel dimension [1, H, W] - PyTorch format
                patch_tensor = torch.from_numpy(normalized_patch).float().unsqueeze(0)
                img_patches.append(patch_tensor)
                        
            # Step 6: If no valid patches, create a default patch from the whole image
            if not img_patches:
                print(f"No valid patches for image {i}, using whole image")
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
                img_resized = cv2.resize(img_gray, (self.patch_size, self.patch_size))
                normalized = img_resized.astype(np.float32) / 255.0
                patch_tensor = torch.from_numpy(normalized).float().unsqueeze(0)
                img_patches = [patch_tensor]
            
            # Stack patches for this image [num_patches, 1, H, W]
            img_patches_tensor = torch.stack(img_patches)
            all_patches.append(img_patches_tensor)
            attention_masks.append(torch.ones(len(img_patches)))
                
        max_patches = max(p.size(0) for p in all_patches)
        print(f"Maximum patches in batch: {max_patches}")
        if max_patches > 100:
            max_patches = min(max_patches, 100)
            print(f"Limiting to {max_patches} patches")

        # Pad to same number of patches
        padded_patches = []
        padded_masks = []

        for patches, mask in zip(all_patches, attention_masks):
            # Limit number of patches if needed
            if patches.size(0) > max_patches:
                patches = patches[:max_patches]
                mask = mask[:max_patches]
            
            # Pad if needed
            if patches.size(0) < max_patches:
                padding = torch.zeros(
                    (max_patches - patches.size(0), 1, self.patch_size, self.patch_size),
                    dtype=patches.dtype, device=patches.device
                )
                padded = torch.cat([patches, padding], dim=0)
                
                # Extend mask
                pad_mask = torch.cat([
                    mask,
                    torch.zeros(max_patches - mask.size(0), device=mask.device)
                ])
            else:
                padded = patches
                pad_mask = mask
            
            padded_patches.append(padded)
            padded_masks.append(pad_mask)

        # Stack into batch, final shape [batch_size, max_patches, 1, H, W]
        patches_batch = torch.stack(padded_patches).to(self.device)
        attention_batch = torch.stack(padded_masks).to(self.device)

        print(f"Final patches shape: {patches_batch.shape}, mask shape: {attention_batch.shape}")
        print(f"Final patches elements: {patches_batch.numel()}")

        return {
            'patches': patches_batch,
            'attention_mask': attention_batch
        }
    
    def _normalize_patch(self, patch):
        """Normalize a character patch to standard size with preserved aspect ratio."""
        if patch.size == 0:
            return np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        # Ensure grayscale for consistency with model expectations
        if len(patch.shape) == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        elif len(patch.shape) == 3 and patch.shape[2] == 1:
            patch = patch.squeeze(-1)  # Remove single channel dimension
        
        # Calculate resize dimensions preserving aspect ratio
        h, w = patch.shape[:2]
        if h > w:
            new_h = self.patch_size
            new_w = int(w * (self.patch_size / h))
        else:
            new_w = self.patch_size
            new_h = int(h * (self.patch_size / w))

        # Ensure minimum dimensions
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        try:
            # Resize the patch
            resized = cv2.resize(patch, (new_w, new_h))
            
            # Create a blank canvas (always grayscale) and center the resized patch
            normalized = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            pad_h = (self.patch_size - new_h) // 2
            pad_w = (self.patch_size - new_w) // 2
            normalized[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            
            print(f"Patch shape in normalize_patch after grayscale conversion: {patch.shape}")
            return normalized / 255.0  # Normalize to [0,1]
        except Exception as e:
            print(f"Error normalizing patch: {e}, patch shape: {patch.shape}")
            # Always return grayscale
            return np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
    
    def forward(self, images, targets=None, annotations=None):
        """
        Forward pass that handles both training and inference modes

        Args:
            images: Tensor of shape [batch_size, channels, height, width]
            targets: Optional font class targets (for training)
            annotations: Optional character annotations (for training)
            
        Returns:
            Dictionary with model outputs
        """
        print(f"Input images shape: {images.shape}, targets: {targets.shape}")

        if len(images.shape) == 4 and images.shape[3] in [1, 3]:  # HWC format
            print(f"Converting images from HWC to CHW format")
            # Permute dimensions: [B, H, W, C] -> [B, C, H, W]
            images = images.permute(0, 3, 1, 2)
            print(f"After transposing: {images.shape}")

        # Check if input is a dictionary (from dataloader)
        if isinstance(images, dict):
            # Extract components from dictionary
            batch_data = images
            images = batch_data['images']
            targets = batch_data['labels'] if 'labels' in batch_data else None
            annotations = batch_data['annotations'] if 'annotations' in batch_data else None

        # Now proceed with normal processing
        # if self.training and annotations is not None:
        #     print("?????????????")
        #     # Training mode: use provided annotations to extract patches
        #     batch_data = self.extract_patches_from_annotations(images, targets, annotations)
        # else:
        #     print("starting craft !!!!!!!!!!!!!!11")
            # Inference mode: use CRAFT to extract patches
        batch_data = self.extract_patches_with_craft(images)
        if targets is not None:
            batch_data['labels'] = targets.to(self.device)

        print(f"Batch data shape: {batch_data['patches'].shape}, attention_mask: {batch_data['attention_mask'].shape}")
        # Process patches with font classifier
        output = self.font_classifier(
            batch_data['patches'], 
            batch_data['attention_mask']
        )

        # Add labels to output if available
        if 'labels' in batch_data:
            output['labels'] = batch_data['labels']
            
        return output