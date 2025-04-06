import os
import json
import cv2
import torch
from PIL import Image

from torch.utils.data import Dataset
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF


from CRAFT import CRAFTModel
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
            #transforms.Resize((self.input_size, self.input_size)),
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
            #nn.Dropout(0.25)
        )
    
        
        self.classifier = nn.Linear(embedding_dim, num_classes)


    def get_embedding(self, x):
        # print(f"Input shape to get_embedding: {x.shape}")

        # Always force resize using F.interpolate
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            # print(f"Resizing from {x.shape[-2]}x{x.shape[-1]} to {self.input_size}x{self.input_size}")
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        # Apply only normalization from transform
        x = self.transform(x)


        x = self.features(x)
        # print(f"After features, shape: {x.shape}")
        x = self.flatten(x)
    
        # print(f"After flatten, shape: {x.shape}, expected flatten_dim: {self.flatten_dim}")
        embeddings = self.embedding_layer(x)
        # print(f"After embedding_layer, shape: {embeddings.shape}")

        
        return embeddings
    
    def forward(self, x):
        embeddings = self.get_embedding(x)
        # print(f"ebmedding shape {embeddings.shape}")
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

        # print(f"Self-attention input shape: {x.shape}, embedding_dim: {self.multihead_attn.embed_dim}")
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
        # print(f"CharacterBasedFontClassifier input patches shape: {char_patches.shape}")

        if len(char_patches.shape) > 5:
            # print(f"Fixing unexpected shape: {char_patches.shape}")
            # Reshape to [batch_size, max_chars, channels, H, W]
            batch_size = char_patches.shape[0]
            max_chars = char_patches.shape[1]
            # Get the channels from the last dimension
            channels = char_patches.shape[-1]
            # print(f"batch_size: {batch_size}, max_chars: {max_chars}, channels: {channels}")
            char_patches = char_patches.reshape(batch_size, max_chars, channels, 
                                                char_patches.shape[3], char_patches.shape[4])
            # print(f"Reshaped to: {char_patches.shape}")
        
        batch_size, max_chars = char_patches.shape[:2]
        # Reshape to process all characters at once - flatten batch and max_chars dimensions
        flat_patches = char_patches.reshape(-1, char_patches.shape[2], 
                                            char_patches.shape[3], char_patches.shape[4])
        # print(f"Flattened patches shape: {flat_patches.shape}")


        # Get character embeddings
        char_embeddings = self.char_encoder.get_embedding(flat_patches)
        # print(f"Raw embeddings shape: {char_embeddings.shape}")

        # expected_embedding_dim = self.aggregator.multihead_attn.embed_dim
        # if char_embeddings.shape[1] != expected_embedding_dim:
        #     print(f"WARNING: Embedding dimension mismatch. Got {char_embeddings.shape[1]}, expected {expected_embedding_dim}")
        #     # Use a linear projection to fix the dimension
        #     projection = nn.Linear(char_embeddings.shape[1], expected_embedding_dim).to(char_embeddings.device)
        #     char_embeddings = projection(char_embeddings)


        # Reshape back to [batch_size, max_chars, embedding_dim]
        char_embeddings = char_embeddings.view(batch_size, max_chars, -1)
        # print(f"Reshaped embeddings shape: {char_embeddings.shape}")
        # Aggregate character embeddings with attention
        font_embedding, attention_weights = self.aggregator(char_embeddings, attention_mask)
        # print(f"Font embedding shape: {font_embedding.shape}, attention_weights shape: {attention_weights.shape}")
        # Classify font
        logits = self.font_classifier(font_embedding)
        # print(f"Logits shape: {logits.shape}")
        
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
                 patch_size=32, embedding_dim=256, craft_fp16=False):
        super().__init__()
        # Initialize CRAFT model for text detection
        self.craft = CRAFTModel(
            cache_dir=craft_weights_dir,
            device=device,
            use_refiner=True,
            fp16=craft_fp16, 
            link_threshold=1.9,
            text_threshold=.5,
            low_text=.5,
        )
        
        # Initialize the font classifier
        self.font_classifier = CharacterBasedFontClassifier(
            num_fonts=num_fonts,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )
        
        self.device = device
        self.patch_size = patch_size
        

    def extract_patches_batch(self, batch_images: torch.Tensor, batch_polys: list):
        """Fully GPU-based patch extraction"""
        # Convert polygons to ROI tensors
        rois = []
        for b_idx, polys in enumerate(batch_polys):
            if not polys:  # Fallback to full image
                rois.append([b_idx, 0, 0, 1, 1])  # Will be scaled later
                continue
                
            for poly in polys:
                x_coords = poly[:, 0]
                y_coords = poly[:, 1]
                x1 = x_coords.min()
                y1 = y_coords.min()
                x2 = x_coords.max()
                y2 = y_coords.max()
                rois.append([b_idx, y1, x1, y2, x2])

        # Handle empty case
        if not rois: 
            return torch.zeros(), torch.zeros()

        # Convert to tensor and normalize coordinates
        roi_tensor = torch.tensor(rois, dtype=torch.float32, device=self.device)
        roi_tensor[:, 1:] /= torch.tensor([
            batch_images.shape[2],  # Height
            batch_images.shape[3],  # Width
            batch_images.shape[2],
            batch_images.shape[3]
        ], device=self.device)

        # Batch extract using ROI align
        patches = torchvision.ops.roi_align(
            batch_images,
            roi_tensor,
            output_size=(self.patch_size, self.patch_size)
        )

        # Create attention mask
        valid_counts = [len(p) if p else 1 for p in batch_polys]
        max_patches = max(valid_counts)
        mask = torch.zeros(
            (batch_images.size(0), max_patches),
            device=self.device
        )
        for i, count in enumerate(valid_counts):
            mask[i, :count] = 1

        return patches, mask
    
    def get_batch_polygons(self, batch_images: torch.Tensor, ratios_w: torch.Tensor, ratios_h: torch.Tensor):
        """Batch process pre-normalized images on GPU"""
        # Forward pass
        with torch.no_grad():
            y, _ = self.net(batch_images)
            if self.refiner:
                y, _ = self.refiner(y, None)

        # Batch post-processing
        text_scores = y[..., 0]  # [B, H, W]
        link_scores = y[..., 1] if not self.refiner else y[..., 0]
        
        # Threshold maps on GPU
        text_mask = (text_scores > self.text_threshold)
        link_mask = (link_scores > self.link_threshold)
        combined_mask = text_mask & link_mask

        # Find connected components using PyTorch's label
        batch_labels = [
            torch.ops.torchvision.label_connected_components(mask.float())
            for mask in combined_mask
        ]

        # Extract polygon coordinates for each component
        batch_polys = []
        for b_idx in range(batch_images.size(0)):
            polys = []
            for label in torch.unique(batch_labels[b_idx]):
                if label == 0: continue
                # Get component coordinates (GPU tensor)
                y_coords, x_coords = torch.where(batch_labels[b_idx] == label)
                if len(x_coords) < 4: continue
                
                # Find convex hull (custom kernel or approximation)
                poly_points = self._convex_hull(x_coords, y_coords)
                
                # Scale coordinates using precomputed ratios
                scaled_poly = poly_points * torch.tensor([
                    [ratios_w[b_idx], ratios_h[b_idx]]
                ], device=self.device)
                
                polys.append(scaled_poly)
            batch_polys.append(polys)

        return batch_polys

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

    def visualize_craft_detections(self, images, targets, label_mapping, save_path=None):
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
        label_mapping = {v: k for k, v in label_mapping.items()}


        for b in range(batch_size):
            # Convert image to numpy and prepare for visualization
            # check if the image is in CHW format
            if len(images[b].shape) == 3 and images[b].shape[0] in [1, 3]:
                img = images[b].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            else:
                img = images[b].cpu().numpy() # HWC
            
            rgb_img = img.astype(np.uint8)
                
            # Convert to PIL for CRAFT
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(rgb_img)
            
            # Get polygons from CRAFT
            try:
                polygons = self.craft.get_polygons(pil_img)
            except Exception as e:
                print(f"CRAFT detection error: {e}")
                polygons = []

            draw = ImageDraw.Draw(pil_img)

            # Draw polygons
            for poly in polygons:
                # Convert to tuple format for PIL
                poly_tuple = [tuple(p) for p in poly]
                draw.polygon(poly_tuple, outline=(255, 0, 0), width=2)

            # Add text at the top if needed
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # Adjust font and size as needed
            except:
                font = ImageFont.load_default()
                draw.text((10, 10), f"CRAFT Detections: {len(polygons)} characters", 
                        fill=(0, 0, 0), font=font)

            # Save with exact dimensions
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image_label = targets[b].item()
                pil_img.save(f"{save_path}_{label_mapping[image_label]}_craft_sample_{b}.png")
            else:
                pil_img.show()  # Display directly with PIL
           
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
                    # print(f"Patch shape: {patch.shape}, tensor shape: {patch_tensor.shape}")
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
    
    def add_padding_to_polygons(polygons, padding_x=5, padding_y=8):
        padded_polygons = []

        for polygon in polygons:
            # Convert to numpy array if it's not already
            polygon = np.array(polygon)

            # Find min and max coordinates
            min_x = np.min(polygon[:, 0])
            max_x = np.max(polygon[:, 0])
            min_y = np.min(polygon[:, 1])
            max_y = np.max(polygon[:, 1])

            # Calculate width and height
            width = max_x - min_x
            height = max_y - min_y

            # Create expanded rectangle
            expanded_rect = np.array([
                [min_x - padding_x, min_y - padding_y],
                [max_x + padding_x, min_y - padding_y],
                [max_x + padding_x, max_y + padding_y],
                [min_x - padding_x, max_y + padding_y]
            ])

            padded_polygons.append(expanded_rect)

        return padded_polygons
    
    def extract_patches_with_craft_old(self, images):
        batch_size = images.size(0)
        all_patches = []
        attention_masks = []

        for i in range(batch_size):
            # if len(images[i].shape) == 3 and images[i].shape[0] in [1, 3]:
            #     img_np = images[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            # else:
            #     img_np = images[i].cpu().numpy() # HWC
        

            # Step 1: Convert tensor to numpy with proper format handling
            # img_tensor = images[i].cpu().numpy().astype(np.uint8)
            # img_np = img_tensor.permute(1, 2, 0).numpy()  # Convert CHW to HWC

            # Ensure correct RGB format for CRAFT
            # if img_np.shape[2] == 1:
            #     img_np = cv2.cvtColor(img_np.squeeze(2), cv2.COLOR_GRAY2RGB)
            
            # Ensure uint8 range
            # if img_np.max() <= 1.0:
            #     print(f"!!!!! Converting image to uint8 from float range [0, 1]")
            #     img_np = (img_np * 255).astype(np.uint8)
            # else:
            #     img_np = img_np.astype(np.uint8)
            
            # # Convert to PIL for CRAFT
            # pil_img = Image.fromarray(img_np)

        
            # Get polygons from CRAFT
            try:
                polygons = self.craft.get_polygons(images[i])
            except Exception as e:
                print(f"CRAFT error: {e}")
                polygons = []
            
            # Extract character patches
            img_patches = []
            for polygon in polygons:
                # polygon = self.add_padding_to_polygons(polygon)

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
        if max_patches > 100:
            max_patches = min(max_patches, 100)

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


        return {
            'patches': patches_batch,
            'attention_mask': attention_batch
        }
    
    def extract_patches_with_craft(self, images, ratio_w=None, ratio_h=None):
        batch_size = images.size(0)
        all_patches = []
        attention_masks = []

        for i in range(batch_size):
            if ratio_w is not None:
                if isinstance(ratio_w, torch.Tensor) and len(ratio_w.shape) > 0:
                    # It's a batch tensor, extract the specific item
                    ratio_w = ratio_w[i].item()
                else:
                    # It's a scalar value (same for all images)
                    ratio_w = ratio_w if not isinstance(ratio_w, torch.Tensor) else ratio_w.item()
            else:
                ratio_w = None
                
            # Handle ratio_h properly for all possible cases  
            if ratio_h is not None:
                if isinstance(ratio_h, torch.Tensor) and len(ratio_h.shape) > 0:
                    # It's a batch tensor, extract the specific item
                    ratio_h = ratio_h[i].item()
                else:
                    # It's a scalar value (same for all images)
                    ratio_h = ratio_h if not isinstance(ratio_h, torch.Tensor) else ratio_h.item()
            else:
                ratio_h = None
            # Get polygons from CRAFT
            try:
                # images is BCHW
                # polygons = self.craft.get_polygons(images[i], ratio_w, ratio_h)
                batch_polys = self.craft.get_batch_polygons(images, ratio_w, ratio_h)
            except Exception as e:
                print(f"CRAFT error: {e}")
                polygons = []
            
            # Extract character patches
            img_patches = []
            for polygon in polygons:
                # polygon = self.add_padding_to_polygons(polygon)

                # Convert polygon to bounding box
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                # Ensure integer coordinates and minimum size
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(images[i].shape[1], int(x2)), min(images[i].shape[0], int(y2))
                
                # Skip very small regions
                if x2-x1 < 3 or y2-y1 < 3:
                    continue
                
                # Extract patch
                patch = images[i][:, y1:y2, x1:x2]  # CHW format
                
                # Convert to grayscale
                # breakpoint()
                if len(patch.shape) == 3 and patch.shape[2] == 3:
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                
                # Resize and normalize (returns a 2D array)
                normalized_patch = self._normalize_patch(patch)
                
                # Convert to tensor with channel dimension [1, H, W] - PyTorch format
                patch_tensor = torch.from_numpy(normalized_patch).float().unsqueeze(0)
                img_patches.append(patch_tensor)

                     
            # Step 6: If no valid patches, create a default patch from the whole image
            if not img_patches:
                img_gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY) if len(images[i].shape) == 3 else images[i]
                img_resized = cv2.resize(img_gray, (self.patch_size, self.patch_size))
                normalized = img_resized.astype(np.float32) / 255.0
                patch_tensor = torch.from_numpy(normalized).float().unsqueeze(0)
                img_patches = [patch_tensor]
            
            # Stack patches for this image [num_patches, 1, H, W]
            img_patches_tensor = torch.stack(img_patches)
            all_patches.append(img_patches_tensor)
            attention_masks.append(torch.ones(len(img_patches)))
                
        max_patches = max(p.size(0) for p in all_patches)
        if max_patches > 100:
            max_patches = min(max_patches, 100)

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
            
            return normalized / 255.0  # Normalize to [0,1]
        except Exception as e:
            print(f"Error normalizing patch: {e}, patch shape: {patch.shape}")
            # Always return grayscale
            return np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
    
    # def forward(self, images, targets=None, annotations=None):

    def forward(self, batch):
        """
        Forward pass that handles both training and inference modes

        Args:
            images: Tensor of shape [batch_size, channels, height, width]
            targets: Optional font class targets (for training)
            annotations: Optional character annotations (for training)
            
        Returns:
            Dictionary with model outputs
        """

        # 1. Get preprocessed images and ratios from DataLoader
        images = batch['images'].to(self.device)  # [B, C, H, W]
        ratios_w = batch['ratio_w'].to(self.device)
        ratios_h = batch['ratio_h'].to(self.device)

        # 2. Batch polygon detection (entirely on GPU)
        batch_polys = self.craft.get_batch_polygons(images, ratios_w, ratios_h)

        # 3. Batch patch extraction (no CPU sync)
        patches, mask = self.extract_patches_batch(images, batch_polys)
        
        # 4. Process through classifier
        return self.font_classifier(patches, mask)

        
