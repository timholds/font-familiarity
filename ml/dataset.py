import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import cv2
import tqdm

def load_npz_mmap(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NPZ file using memory mapping."""
    with np.load(file_path, mmap_mode='r') as data:
        # Load data using memory mapping
        images = data['images']
        # Load labels and adjust indexing (convert from 1-based to 0-based)
        # im sorry this is so fucked but the old data is 1 indexed. i fixed it to be 0 indexed in the new data
        labels = data['labels'] 
        if (labels == 0).any():
            print("Labels file 0 indexed.")
        else: # labels are 1 indexed
            labels -= 1
        
        assert (labels >= 0).all(), f"Negative label indices found after converting to 0 index.\
              Expecting riginal to be >= 1"
        return images, labels

class FontDataset(Dataset):
    """
    Dataset for loading font images from NPZ files.
    """
    def __init__(self, root_dir: str, train: bool = True):
        self.root_dir = root_dir
        mode = 'train' if train else 'test'
        h5_file = os.path.join(root_dir, f'{mode}.h5')
        npz_file = os.path.join(root_dir, f'{mode}.npz')
        
        # Choose correct loader based on file existence
        if os.path.exists(h5_file):
            print(f"Loading H5 dataset from {h5_file}")
            self.data, self.targets, self.h5_file = load_h5_dataset(h5_file)
            self.using_h5 = True
        elif os.path.exists(npz_file):
            print(f"Loading NPZ dataset from {npz_file}")
            self.data, self.targets = load_npz_mmap(npz_file)
            self.using_h5 = False
            self.h5_file = None
        else:
            raise FileNotFoundError(f"No dataset file found at {h5_file} or {npz_file}")
        
        # Load label mapping
        label_map_path = os.path.join(root_dir, 'label_mapping.npy')
        self.label_mapping = np.load(label_map_path, allow_pickle=True).item()
        self.num_classes = len(self.label_mapping)
        print(f"Number of classes: {self.num_classes}")
        print(f"Label mapping loaded from {label_map_path}")
      
    def __del__(self):
        # Close H5 file if it's open
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

    def _validate_targets(self):
        """Validate that all targets are within the correct range."""
        min_target = self.targets.min()
        max_target = self.targets.max()
        unique_targets = np.unique(self.targets)
        
        if min_target < 0 or max_target >= self.num_classes:
            raise ValueError(
                f"Invalid target values found!\n"
                f"Number of classes: {self.num_classes}\n"
                f"Target range: [{min_target}, {max_target}]\n"
                f"Unique targets: {unique_targets}\n"
                f"Label mapping size: {len(self.label_mapping)}"
            )
        
        print(f"Target validation passed:\n"
              f"- Number of classes: {self.num_classes}\n"
              f"- Target range: [{min_target}, {max_target}]\n"
              f"- Number of unique targets: {len(unique_targets)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        target = self.targets[idx]

        if not (0 <= target < self.num_classes):
            raise ValueError(f"Invalid target {target} at index {idx}")
            
        return img, target

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and test DataLoaders.
    """
    train_dataset = FontDataset(data_dir, train=True)
    test_dataset = FontDataset(data_dir, train=False)

    assert train_dataset.num_classes == test_dataset.num_classes, (
        f"Mismatch between train ({train_dataset.num_classes}) and "
        f"test ({test_dataset.num_classes}) class counts"
    )
    
    # Validate label mappings are identical
    assert train_dataset.label_mapping == test_dataset.label_mapping, (
        "Train and test datasets have different label mappings"
    )    
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Label mapping size: {len(train_dataset.label_mapping)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.num_classes

def load_h5_dataset(file_path):
    """Load dataset from H5 file format"""
    import h5py
    
    # Ensure proper extension
    h5_filename = file_path if file_path.endswith('.h5') else f"{file_path}.h5"
    
    try:
        # Open the H5 file
        f = h5py.File(h5_filename, 'r')
        
        # Access the datasets
        images = f['images']
        labels = f['labels'][:]  # Load labels fully into memory as they're small
        
        # Convert labels if needed (keeping your 1-indexed vs 0-indexed logic)
        if not (labels == 0).any():  # labels are 1-indexed
            labels -= 1
            
        assert (labels >= 0).all(), "Negative label indices found after converting to 0 index"
        
        return images, labels, f  # Return file handle to keep it open
    except Exception as e:
        print(f"Error loading H5 dataset: {e}")
        # Try loading as npz for backward compatibility
        try:
            return load_npz_mmap(file_path.replace('.h5', '.npz')), None
        except:
            raise RuntimeError(f"Failed to load dataset from {file_path}")

def load_char_npz_mmap(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NPZ file using memory mapping."""
    # TODO are we sticking with grayscale images?
    with np.load(file_path, allow_pickle=True, mmap_mode='r') as data:

        # Load data using memory mapping
        images = data['images']
        # Load labels and adjust indexing (convert from 1-based to 0-based)
        # im sorry this is so fucked but the old data is 1 indexed. i fixed it to be 0 indexed in the new data
        labels = data['labels'] 
        if (labels == 0).any():
            print("Labels file 0 indexed.")
        else: # labels are 1 indexed
            labels -= 1
        
        assert (labels >= 0).all(), f"Negative label indices found after converting to 0 index.\
              Expecting riginal to be >= 1"
        return images, labels

def preprocess_with_craft(data_dir, craft_model, batch_size=32, num_workers=4, train=True):
    """
    Create a dataloader with character patches extracted by CRAFT
    
    Args:
        data_dir: Directory containing font dataset
        craft_model: Initialized CRAFT model
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        
    Returns:
        DataLoader with pre-extracted character patches
    """
    from PIL import Image
    
    # Use your existing dataset class
    train_dataset = CharacterFontDataset(data_dir, train=train, use_annotations=False)
    
    # Process each image to extract character patches
    processed_data = []
    
    print("Preprocessing dataset with CRAFT character detection...")
    for idx in tqdm(range(len(train_dataset))):
        img, target = train_dataset[idx]
        
        # Convert tensor to image format for CRAFT
        img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
        img_np = (img_np * 255).astype(np.uint8)
        
        # Handle grayscale vs RGB
        if img_np.shape[-1] == 1:
            img_np = np.squeeze(img_np)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(img_np)
        
        # Get character polygons from CRAFT
        polygons = craft_model.get_polygons(pil_img)
        
        # Extract patches
        char_patches = []
        for polygon in polygons:
            # Convert polygon to bounding box
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            
            x1, y1 = max(0, min(x_coords)), max(0, min(y_coords))
            x2, y2 = min(img_np.shape[1], max(x_coords)), min(img_np.shape[0], max(y_coords))
            
            # Skip very small regions
            if x2-x1 < 3 or y2-y1 < 3:
                continue
                
            # Extract patch
            patch = img_np[y1:y2, x1:x2].copy()
            
            # Convert to grayscale if needed
            if len(patch.shape) == 3 and patch.shape[2] == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                
            # Normalize and resize to standard size
            patch = cv2.resize(patch, (32, 32))
            patch = patch.astype(np.float32) / 255.0
            
            # Convert to tensor format
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Add channel dim
            char_patches.append(patch_tensor)
        
        # If no patches found, use whole image
        if not char_patches:
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np
            
            patch = cv2.resize(img_gray, (32, 32))
            patch = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
            char_patches = [patch_tensor]
        
        # Stack patches and store with font label
        patches_tensor = torch.stack(char_patches)
        processed_data.append({
            'patches': patches_tensor,
            'label': target,
            'num_patches': len(char_patches)
        })
    
    # Create dataloader with custom collate function
    return DataLoader(
        processed_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=char_collate_fn  # Reuse your existing collate function
    )

class CharacterFontDataset(Dataset):
    """Dataset for font classification using character patches."""
    
    def __init__(self, root_dir: str, train: bool = True, char_size: int = 32,
                  max_chars: int = 50, use_annotations=False, 
                  use_precomputed_craft=False):
        self.root_dir = root_dir
        self.char_size = char_size
        self.max_chars = max_chars
        self.use_annotations = use_annotations

        self.use_precomputed_craft = use_precomputed_craft
        
        # Load font data using original approach
        mode = 'train' if train else 'test'
        #data_file = os.path.join(root_dir, f'{mode}.h5')
        h5_file = os.path.join(root_dir, f'{mode}.h5')
        npz_file = os.path.join(root_dir, f'{mode}.npz')
        
        # Choose correct loader based on file existence
        if os.path.exists(h5_file):
            print(f"Loading H5 dataset from {h5_file}")
            self.data, self.targets, self.h5_file = load_h5_dataset(h5_file)
            self.using_h5 = True
        elif os.path.exists(npz_file):
            print(f"Loading NPZ dataset from {npz_file}")
            self.data, self.targets = load_char_npz_mmap(npz_file)
            self.using_h5 = False
            self.h5_file = None
        else:
            raise FileNotFoundError(f"No dataset file found at {h5_file} or {npz_file}")
        
        #self.data, self.targets = load_char_npz_mmap(data_file)
        
        # Load label mapping
        label_map_path = os.path.join(root_dir, 'label_mapping.npy')
        self.label_mapping = np.load(label_map_path, allow_pickle=True).item()
        self.num_classes = len(self.label_mapping)
        
        # Create reverse mapping to get font name from index
        self.idx_to_font = {idx: font for font, idx in self.label_mapping.items()}
        
        # Load character class mapping
        # Load character class mapping if it exists
        self.char_mapping = {}
        classes_path = os.path.join(root_dir, 'classes.txt')
        if os.path.exists(classes_path):
            self.char_mapping = self._load_char_mapping(classes_path)

        print(f"Initialized CharacterFontDataset with {len(self.data)} samples, {self.num_classes} fonts")
        if self.use_annotations:
            print(f"Using character annotations. Found {len(self.char_mapping)} character mappings.")
            print(f"Initialized CharacterFontDataset with {len(self.data)} samples, {self.num_classes} fonts")
        
        self.precomputed_boxes = None
        if self.use_precomputed_craft:
            craft_file = os.path.join(self.root_dir, f'{mode}_craft_boxes.npz')
            if os.path.exists(craft_file):
                try:
                    loaded_data = np.load(craft_file, allow_pickle=True)
                    self.precomputed_boxes = loaded_data['boxes']
                    print(f"Loaded precomputed CRAFT boxes for {len(self.precomputed_boxes)} images")
                except Exception as e:
                    print(f"Error loading precomputed CRAFT boxes: {e}")
                    self.precomputed_boxes = None
            else:
                print(f"Warning: Precomputed CRAFT file not found at {craft_file}")
    

    def _load_char_mapping(self, mapping_file: str) -> dict:
        """Load mapping from class_id to character."""
        mapping = {}
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        class_id, char = parts
                        mapping[int(class_id)] = char
            print(f"Loaded {len(mapping)} character mappings from {mapping_file}")
        except Exception as e:
            print(f"Warning: Could not load character mapping: {e}")
        return mapping
        
    def _extract_char_patches(self, image: np.ndarray, font_idx: int, img_idx: int) -> list:
        """Extract character patches using YOLO annotations."""
        # Construct annotation path based on font name and image index
        font_name = self.idx_to_font.get(font_idx, f"unknown_font_{font_idx}")
        sample_id = f"sample_{img_idx:04d}"
        
        # Try to find annotations (YOLO format preferred, fallback to JSON)
        yolo_path = os.path.join(self.root_dir, font_name, "annotations", f"{sample_id}.txt")
        json_path = os.path.join(self.root_dir, font_name, "annotations", f"{sample_id}.json")
        
        patches = []
        height, width = image.shape[:2]
        
        # Process YOLO format annotations
        if os.path.exists(yolo_path):
            try:
                with open(yolo_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * width
                            y_center = float(parts[2]) * height
                            w = float(parts[3]) * width
                            h = float(parts[4]) * height
                            
                            # Convert to pixel coordinates
                            x1 = max(0, int(x_center - w/2))
                            y1 = max(0, int(y_center - h/2))
                            x2 = min(width, int(x_center + w/2))
                            y2 = min(height, int(y_center + h/2))
                            
                            # Extract and preprocess patch
                            if x2-x1 > 2 and y2-y1 > 2:  # Ensure minimum size
                                patch = image[y1:y2, x1:x2].copy()
                                if len(patch.shape) == 3 and patch.shape[2] == 3:
                                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                                normalized_patch = self._normalize_patch(patch)
                                
                                char = self.char_mapping.get(class_id, '?')
                                patches.append({
                                    'patch': normalized_patch,
                                    'class_id': class_id,
                                    'char': char
                                })
            except Exception as e:
                print(f"Error processing YOLO annotations for {yolo_path}: {e}")
        
        # Process JSON format annotations if YOLO not available
        elif os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                for char_info in data.get('characters', []):
                    x = char_info.get('x', 0)
                    y = char_info.get('y', 0)
                    w = char_info.get('width', 0)
                    h = char_info.get('height', 0)
                    char = char_info.get('char', '?')
                    
                    # Extract and preprocess patch
                    if w > 2 and h > 2:  # Ensure minimum size
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(width, x + w), min(height, y + h)
                        
                        patch = image[y1:y2, x1:x2].copy()
                        normalized_patch = self._normalize_patch(patch)
                        
                        # Find class ID for this character if possible
                        class_id = next((k for k, v in self.char_mapping.items() if v == char), -1)
                        patches.append({
                            'patch': normalized_patch,
                            'class_id': class_id,
                            'char': char
                        })
            except Exception as e:
                print(f"Error processing JSON annotations for {json_path}: {e}")
        
        # Limit to max_chars
        return patches[:self.max_chars]
    
    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Normalize a character patch to standard size with preserved aspect ratio."""
        if patch.size == 0:
            return np.zeros((self.char_size, self.char_size), dtype=np.float32)
            
        # Calculate resize dimensions preserving aspect ratio
        h, w = patch.shape
        if h > w:
            new_h = self.char_size
            new_w = int(w * (self.char_size / h))
        else:
            new_w = self.char_size
            new_h = int(h * (self.char_size / w))
        
        # Ensure minimum dimensions
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        try:
            # Resize the patch
            import cv2
            resized = cv2.resize(patch, (new_w, new_h))
            
            # Create a blank canvas and center the resized patch
            normalized = np.zeros((self.char_size, self.char_size), dtype=np.float32)
            pad_h = (self.char_size - new_h) // 2
            pad_w = (self.char_size - new_w) // 2
            normalized[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            
            return normalized 
        except Exception as e:
            print(f"Error normalizing patch: {e}")
            return np.zeros((self.char_size, self.char_size), dtype=np.float32)
    
    def _extract_patches_from_boxes(self, image: np.ndarray, boxes: list) -> tuple:
        """Extract character patches from image using precomputed bounding boxes."""
        patches = []

        # Debug information
        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Fix dimension ordering if needed - handle (height, channels, width) format
        if len(image.shape) == 3 and image.shape[1] == 3 and image.shape[0] > 10 and image.shape[2] > 10:
            print(f"Fixing swapped dimensions from {image.shape}")
            image = np.transpose(image, (0, 2, 1))  # Convert to (height, width, channels)
            print(f"Transposed to {image.shape}")

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                print(f"Converting from float [0,1] to uint8")
                image = (image * 255).astype(np.uint8)

        # Ensure image has proper dimensions
        height, width = image.shape[:2]
        print(f"Image dimensions: height={height}, width={width}")

        # Handle grayscale or other formats
        if len(image.shape) == 2:
            # Convert grayscale to 3-channel for consistent processing
            image = np.stack([image, image, image], axis=2)
            print(f"Converted grayscale to 3-channel image: {image.shape}")
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel image - expand to 3 channels
            image = np.concatenate([image, image, image], axis=2)
            print(f"Expanded 1-channel to 3-channel image: {image.shape}")

        # Process each box
        for box in boxes:
            try:
                # Handle different box formats
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                else:
                    print(f"Skipping unsupported box format: {box}")
                    continue
                    
                # Ensure valid coordinates
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(width, int(x2)), min(height, int(y2))
                
                # Skip very small regions
                if x2-x1 < 3 or y2-y1 < 3:
                    continue
                
                # Extract patch
                patch = image[y1:y2, x1:x2].copy()
                
                # Safely convert to grayscale
                try:
                    if len(patch.shape) == 3 and patch.shape[2] == 3:
                        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                    elif len(patch.shape) == 3 and patch.shape[2] == 1:
                        patch = patch.squeeze(-1)
                except Exception as e:
                    print(f"Error in grayscale conversion: {e}, shape={patch.shape}")
                    # Fallback to manual grayscale conversion
                    if len(patch.shape) == 3:
                        patch = np.mean(patch, axis=2).astype(np.uint8)
                
                # Normalize patch
                normalized_patch = self._normalize_patch(patch)
                
                # Convert to tensor
                patch_tensor = torch.from_numpy(normalized_patch).float().unsqueeze(0)
                patches.append(patch_tensor)
                
            except Exception as e:
                print(f"Error processing box {box}: {e}")
                continue

        # If no valid patches, create a default patch from the whole image
        if not patches:
            print(f"No valid patches, creating fallback from whole image")
            try:
                # Try to create grayscale version of the whole image
                if len(image.shape) == 3:
                    try:
                        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    except Exception as e:
                        print(f"Fallback grayscale error: {e}")
                        # Manual grayscale conversion
                        gray_img = np.mean(image, axis=2).astype(np.uint8)
                else:
                    gray_img = image
                    
                # Resize to standard size
                img_resized = cv2.resize(gray_img, (self.char_size, self.char_size))
                normalized = img_resized.astype(np.float32) / 255.0
                patch_tensor = torch.from_numpy(normalized).float().unsqueeze(0)
                patches = [patch_tensor]
            except Exception as e:
                print(f"Error creating fallback: {e}")
                # Last resort - create an empty patch
                empty_patch = np.zeros((self.char_size, self.char_size), dtype=np.float32)
                patch_tensor = torch.from_numpy(empty_patch).float().unsqueeze(0)
                patches = [patch_tensor]

        # Stack patches and create mask
        patches = patches[:self.max_chars]
        stacked_patches = torch.stack(patches)
        attention_mask = torch.ones(len(patches))

        return stacked_patches, attention_mask

    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, idx: int):
        img = self.data[idx].astype(np.float32)  # HWC
        target = int(self.targets[idx])
        target = torch.tensor(target, dtype=torch.long)
        
        # Check if we should use precomputed CRAFT boxes
        if self.use_precomputed_craft and self.precomputed_boxes is not None:
            # Extract boxes for this image
            boxes = self.precomputed_boxes[idx]
            
            # Extract patches using precomputed boxes
            patches, attention_mask = self._extract_patches_from_boxes(img, boxes)
            
            # Return patches directly
            return {
                'patches': patches,
                'attention_mask': attention_mask,
                'labels': target
            }
        
        else:
            # Convert the full image to tensor
            img_tensor = torch.from_numpy(img).float()

            # Add channel dimension if needed
            if img_tensor.dim() == 2:  # If grayscale without channel
                img_tensor = img_tensor.unsqueeze(0)
            
            # Skip annotation processing if not using annotations
            if not self.use_annotations:
                return img_tensor, target, []

            # Get font name for annotation lookup
            font_idx = target
            font_name = self.idx_to_font.get(font_idx, f"unknown_font_{font_idx}")
            sample_id = f"sample_{idx:04d}"
            yolo_path = os.path.join(self.root_dir, font_name, "annotations", f"{sample_id}.txt")
            
            # Process YOLO format annotations if available
            annotations = []
            if os.path.exists(yolo_path):
                try:
                    with open(yolo_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Format: class_id, x_center, y_center, w, h
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                annotations.append([class_id, x_center, y_center, w, h])
                except Exception as e:
                    print(f"Error processing YOLO annotations for {yolo_path}: {e}")

            return img_tensor, target, annotations

def char_collate_fn(batch):
    """
    Custom collate function for batches with images and annotations.
    
    Args:
        batch: List of (image, target, annotations) tuples from dataset
        
    Returns:
        Dictionary with batched data
    """
    # Separate images, labels, and annotations
    if 'patches' in batch[0]:
        # Extract items from each batch element
        patches = [item['patches'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        targets = [item['labels'] for item in batch]
        
        # Get max number of patches
        max_patches = max(p.size(0) for p in patches)
        
        # Pad patches and attention masks
        padded_patches = []
        padded_masks = []
        
        for patch_set, mask in zip(patches, attention_masks):
            if patch_set.size(0) < max_patches:
                # Create padding
                padding = torch.zeros(
                    (max_patches - patch_set.size(0), 1, patch_set.size(2), patch_set.size(3)),
                    dtype=patch_set.dtype
                )
                padded = torch.cat([patch_set, padding], dim=0)
                
                # Extend mask
                pad_mask = torch.cat([
                    mask,
                    torch.zeros(max_patches - mask.size(0))
                ])
            else:
                padded = patch_set
                pad_mask = mask
            
            padded_patches.append(padded)
            padded_masks.append(pad_mask)
        
        # Stack into batch tensors
        patches_batch = torch.stack(padded_patches)
        attention_batch = torch.stack(padded_masks)
        targets_batch = torch.stack(targets)
        
        return {
            'patches': patches_batch,
            'attention_mask': attention_batch,
            'labels': targets_batch
        }
    else:
        images, targets, annotations_list = zip(*batch)

        # Stack images and convert targets to tensor
        images_batch = torch.stack(images)
        targets_batch = torch.tensor(targets)

        return {
            'images': images_batch,          # [batch_size, channels, H, W]
            'labels': targets_batch,         # [batch_size]
            'annotations': annotations_list  # List of annotation lists
        }


def get_char_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4, 
    use_annotations: bool = False,
    use_precomputed_craft: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and test DataLoaders.
    """
    train_dataset = CharacterFontDataset(
        data_dir, train=True, 
        use_annotations=use_annotations,
        use_precomputed_craft=use_precomputed_craft,
    )
    
    test_dataset = CharacterFontDataset(
        data_dir, train=False, 
        use_annotations=use_annotations,
        use_precomputed_craft=use_precomputed_craft,
    )

    assert train_dataset.num_classes == test_dataset.num_classes, (
        f"Mismatch between train ({train_dataset.num_classes}) and "
        f"test ({test_dataset.num_classes}) class counts"
    )
    
    # Validate label mappings are identical
    assert train_dataset.label_mapping == test_dataset.label_mapping, (
        "Train and test datasets have different label mappings"
    )    
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Label mapping size: {len(train_dataset.label_mapping)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=char_collate_fn 
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=char_collate_fn 
    )
    
    return train_loader, test_loader, train_dataset.num_classes


