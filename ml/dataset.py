import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import cv2
import tqdm
import h5py
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
try:
    from .font_utils import FontNameNormalizer, get_normalizer
except ImportError:
    # Handle case when run from project root
    from font_utils import FontNameNormalizer, get_normalizer

CATEGORY_LIST = ["DISPLAY", "SANS_SERIF", "SERIF", "HANDWRITING", "MONOSPACE"]
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORY_LIST)}


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
    def __init__(self, root_dir: str, train: bool = True, return_category: bool = False):
        self.root_dir = root_dir
        self.return_category = return_category
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

        # Load font name to category mapping using FontNameNormalizer
        if self.return_category:
            normalizer = get_normalizer()
            
            # Build label idx to category idx mapping
            self.labelidx_to_catidx = {}
            for font_name, idx in self.label_mapping.items():
                # Get category using the normalizer (handles all format conversions)
                cat = normalizer.get_category(font_name)
                
                # Debug first few lookups only
                if idx < 3:
                    print(f"DEBUG lookup: '{font_name}' -> category: {cat}")
                
                if cat is not None:
                    self.labelidx_to_catidx[idx] = CATEGORY_TO_IDX[cat]
                else:
                    self.labelidx_to_catidx[idx] = -1  # Unknown
                    if self.return_category:  # Only warn when categories are needed
                        print(f"WARNING: No category for font '{font_name}'")
                    
            # Debug output
            matched = sum(1 for v in self.labelidx_to_catidx.values() if v != -1)
            print(f"DEBUG: Matched {matched}/{len(self.labelidx_to_catidx)} fonts to categories")
            if matched > 0:
                print(f"DEBUG: Sample mappings: {list(self.labelidx_to_catidx.items())[:5]}")
                print(f"DEBUG: Categories found: {set(self.labelidx_to_catidx.values())}")

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

    def __getitem__(self, idx: int):
        img = self.data[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Handle both grayscale and color images
        if len(img.shape) == 2:  # Grayscale image (H, W)
            img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension -> (1, H, W)
        elif len(img.shape) == 3:  # Color image (H, W, C)
            img = torch.from_numpy(img).permute(2, 0, 1)  # Convert HWC to CHW
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
            
        target = self.targets[idx]
        if not (0 <= target < self.num_classes):
            raise ValueError(f"Invalid target {target} at index {idx}")
        if self.return_category:
            cat_idx = self.labelidx_to_catidx.get(target, -1)
            return img, target, torch.tensor(cat_idx, dtype=torch.long)
        # Always return three items for consistency
        return img, target, torch.tensor(-1, dtype=torch.long)

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    return_category: bool = False
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Creates train and test DataLoaders.
    """
    train_dataset = FontDataset(data_dir, train=True, return_category=return_category)
    test_dataset = FontDataset(data_dir, train=False, return_category=return_category)

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
    
def polygon_to_box(polygon):
    """
    Convert polygon [[x1,y1], [x2,y2], ...] to bounding box [x1, y1, x2, y2]
    
    Args:
        polygon: List of [x, y] coordinate pairs
    
    Returns:
        Bounding box as [x1, y1, x2, y2]
    """
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def add_padding_to_box(box, padding_x=0.1, padding_y=0.2, asym=False, jitter_std=0.0):
    """
    Add padding to a bounding box with optional jittering for data augmentation.

    Args:
        box: A bounding box [x1, y1, x2, y2]
        padding_x: Base horizontal padding as a fraction of width.
        padding_y: Base vertical padding as a fraction of height.
        asym: If True, add more padding to the left side to correct for bias.
        jitter_std: Standard deviation for jittering the padding values.

    Returns:
        Padded bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box

    # Calculate width and height
    width = x2 - x1
    height = y2 - y1

    jittered_padding_x = float(padding_x) + random.gauss(0, jitter_std)
    jittered_padding_y = float(padding_y) + random.gauss(0, jitter_std)

    # Calculate padding
    pad_x = int(jittered_padding_x * width)
    pad_y = int(jittered_padding_y * height)

    # Apply padding
    if not asym:
        padded_x1 = x1 - pad_x
        padded_x2 = x2 + pad_x
    else:
        padded_x1 = x1 - pad_x
        padded_x2 = x2 + int(pad_x // 3)  # Smaller padding on the right

    padded_y1 = y1 - pad_y
    padded_y2 = y2 + pad_y

    # Return padded bounding box
    return [padded_x1, padded_y1, padded_x2, padded_y2]


# Backward compatibility wrapper
def add_padding_to_polygons(data, padding_x=0.1, padding_y=0.2, asym=False, jitter_std=0.0):
    """
    Backward compatibility wrapper - converts polygon to box and adds padding.
    
    Args:
        data: A polygon [[x1, y1], [x2, y2], ...] or bounding box [x1, y1, x2, y2]
        padding_x: Base horizontal padding as a fraction of width.
        padding_y: Base vertical padding as a fraction of height.
        asym: If True, add more padding to the left side to correct for bias.
        jitter_std: Standard deviation for jittering the padding values.

    Returns:
        Padded bounding box [x1, y1, x2, y2]
    """
    # Convert polygon to box if needed
    if isinstance(data, list) and len(data) == 4:
        # Already a bounding box
        box = data
    elif isinstance(data, np.ndarray) and len(data) == 4:
        # Numpy array bounding box
        box = data.tolist()
    else:
        # Assume it's a polygon - convert to box
        box = polygon_to_box(data)
    
    return add_padding_to_box(box, padding_x, padding_y, asym, jitter_std)
    
class CharacterFontDataset(Dataset):
    """Dataset for font classification using character patches."""
    
    def __init__(self, root_dir: str, train: bool = True, char_size: int = 32,
                  max_chars: int = 100, use_precomputed_craft=False, pad_x=.05, pad_y=.15,
                  return_category: bool = False):
        self.root_dir = root_dir
        self.char_size = char_size
        self.max_chars = max_chars
        self.use_precomputed_craft = use_precomputed_craft
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.return_category = return_category
        
        self.mode = 'train' if train else 'test'
        h5_file = os.path.join(root_dir, f'{self.mode}.h5')
        
        if os.path.exists(h5_file):
            print(f"Loading H5 dataset from {h5_file}")
            self.data, self.targets, self.h5_file = load_h5_dataset(h5_file)
        else:
            raise FileNotFoundError(f"No dataset file found at {h5_file}")
        
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

        # Load font name to category mapping using FontNameNormalizer
        if self.return_category:
            normalizer = get_normalizer()
            
            # Build label idx to category idx mapping
            self.labelidx_to_catidx = {}
            for font_name, idx in self.label_mapping.items():
                # Get category using the normalizer (handles all format conversions)
                cat = normalizer.get_category(font_name)
                if cat is not None:
                    self.labelidx_to_catidx[idx] = CATEGORY_TO_IDX[cat]
                else:
                    self.labelidx_to_catidx[idx] = -1  # Unknown

        print(f"Initialized CharacterFontDataset with {len(self.data)} samples, {self.num_classes} fonts")

        self.precomputed_boxes = None
        self.craft_h5_file = None
        self.valid_indices = []

        if train:
            self.augmentations = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                # A.HueSaturationValue(p=0.5),                
                # A.CLAHE(p=0.5),
                A.RandomGamma(p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.RandomScale(scale_limit=0.1, p=0.5),
                # A.RandomCrop(height=self.char_size, width=self.char_size, p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                ToTensorV2()  # Convert to PyTorch tensor
            ])
        else:
            self.augmentations = None

        if self.use_precomputed_craft:
            craft_h5_file = os.path.join(self.root_dir, f'{self.mode}_craft_boxes.h5')
            if os.path.exists(craft_h5_file):
                try:
                    # Load CRAFT boxes
                    self.craft_h5_file = h5py.File(craft_h5_file, 'r')
                    self.boxes_group = self.craft_h5_file['boxes']
                    self.batch_size = self.boxes_group.attrs.get('preprocessing_batch_size', 32)  # Default to 32 if not found
                    self.precomputed_boxes = True  # Flag indicating boxes are available via HDF5
                    print(f"Opened precomputed CRAFT boxes from HDF5 file: {craft_h5_file}")
                except Exception as e:
                    print(f"Error opening precomputed CRAFT HDF5 file: {e}")
                    self.precomputed_boxes = None
                    if self.craft_h5_file is not None:
                        self.craft_h5_file.close()
                        self.craft_h5_file = None
            else:
                raise FileNotFoundError(f"Precomputed CRAFT file not found at {craft_h5_file}")
    

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
        
    def _extract_patches_from_boxes(self, image: np.ndarray, boxes: list, idx, mode, pad_x=.05, pad_y=.15) -> tuple:
        """Extract character patches from image using precomputed bounding boxes.
        Images HWC [0, 255]"""

        assert image.shape[2] == 3, f"Expecting HWC w/ RGB format (3 channels), got {image.shape}"
        patches = []

        
        if image.max() <= 1.0:
            print(f"Converting from float [0,1] to uint8")
            image = (image * 255).astype(np.uint8)

        if image.dtype != np.uint8:
            # print(f"Converting image to uint8")    
            image.astype(np.uint8)

        # Ensure image has proper dimensions
        height, width = image.shape[:2]
        #print(f"Image dimensions: height={height}, width={width}")
        
        # Process each box
        
        for box in boxes:
            # print(f"\n\n\nProcessing box: {box}\n\n\n")
            # Only add jittered padding for training mode
            if mode == 'train':
                box = add_padding_to_polygons(box, padding_x=pad_x, padding_y=pad_y, asym=True, jitter_std=0.05)
            else:
                box = add_padding_to_polygons(box, padding_x=pad_x, padding_y=pad_y, asym=True, jitter_std=0.0)
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
                if x2-x1 < 5 or y2-y1 < 5:
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
            print(f"WARNING: No valid patches for img {idx}, creating fallback from whole image")
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
 
    def __del__(self):
        # Close H5 file if it's open
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
        if hasattr(self, 'craft_h5_file') and self.craft_h5_file is not None:
            self.craft_h5_file.close()
        if hasattr(self, 'boxes_file') and self.boxes_file is not None:
            self.boxes_file.close()
    
    def __getitem__(self, idx: int):
        target = self.targets[idx]
        if not (0 <= target < self.num_classes):
            raise ValueError(f"Invalid target {target} at index {idx}")
        
        if self.use_precomputed_craft:
            # Get the image for CRAFT processing
            image = self.data[idx]  # Keep as uint8 for CRAFT
            
            # Get boxes for this index
            if self.craft_h5_file is not None:
                # Using HDF5 storage
                if str(idx) in self.boxes_group:
                    boxes = self.boxes_group[str(idx)][()]
                else:
                    boxes = []
            else:
                # Using in-memory storage
                boxes = self.precomputed_boxes.get(idx, [])
            
            # Extract patches from boxes
            patches, attention_mask = self._extract_patches_from_boxes(
                image, boxes, idx, self.mode, self.pad_x, self.pad_y
            )
            
            # Build the return dictionary
            result = {
                'patches': patches,
                'attention_mask': attention_mask,
                'labels': torch.tensor(target, dtype=torch.long)
            }
            
            # Add category label if using contrastive loss
            if self.return_category:
                cat_idx = self.labelidx_to_catidx.get(target, -1)
                result['category'] = torch.tensor(cat_idx, dtype=torch.long)
            
            return result
        else:
            # Original behavior - return raw image
            img = self.data[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
            img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
            
            if self.return_category:
                cat_idx = self.labelidx_to_catidx.get(target, -1)
                return img, target, torch.tensor(cat_idx, dtype=torch.long)
            # Always return three items for consistency
            return img, target, torch.tensor(-1, dtype=torch.long)

def char_collate_fn(batch):
    """
    Custom collate function for batches with images and patches.
    
    Args:
        batch: List of (image, target, patches) tuples from dataset
        
    Returns:
        Dictionary with batched data
    """
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
        
        result = {
            'patches': patches_batch,
            'attention_mask': attention_batch,
            'labels': targets_batch
        }
        
        # Add category labels if present
        if 'category' in batch[0]:
            categories = [item['category'] for item in batch]
            result['category'] = torch.stack(categories)
        
        return result
    elif 'images' in batch[0]:
        # Handle raw images
        images = torch.stack([item['images'] for item in batch])
        targets = torch.stack([item['labels'] for item in batch])
        return {
            'images': images,
            'labels': targets
        }
    else:
        raise ValueError("Data loader: Batch does not contain 'patches' or 'images' key. Check dataset output format.")
    
def get_char_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 16, 
    use_precomputed_craft: bool = False,
    save_problematic_images: bool = False,
    pad_x: float = 0.05,
    pad_y: float = 0.15,
    return_category: bool = False,
    max_datapts: int = None,
    max_chars: int = 100
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and test DataLoaders.
    """
    train_dataset = CharacterFontDataset(
        data_dir, train=True, 
        use_precomputed_craft=use_precomputed_craft,
        pad_x=pad_x, pad_y=pad_y,
        return_category=return_category,
        max_chars=max_chars
    )
    
    test_dataset = CharacterFontDataset(
        data_dir, train=False, 
        use_precomputed_craft=use_precomputed_craft,
        pad_x=pad_x, pad_y=pad_y,
        return_category=return_category,
        max_chars=max_chars
    )

    num_classes = train_dataset.num_classes
    label_mapping = train_dataset.label_mapping

    if use_precomputed_craft:
        # For training set
        train_valid_indices = []
        train_invalid_indices = []
        
        # Limit how many samples we check if max_datapts is specified
        train_limit = min(len(train_dataset), max_datapts) if max_datapts else len(train_dataset)
        
        for idx in range(train_limit):
            # Check if this sample has any VALID boxes (with minimum size)
            has_valid_boxes = False
            if train_dataset.craft_h5_file is not None:
                if str(idx) in train_dataset.boxes_group:
                    boxes = train_dataset.boxes_group[str(idx)][()]
                    if boxes.size > 0:
                        # Check each box for minimum size
                        for box in boxes:
                            if box.shape[0] >= 4:  # Ensure box has enough values
                                x1, y1, x2, y2 = box[:4]
                                if x2-x1 >= 5 and y2-y1 >= 5:
                                    has_valid_boxes = True
                                    break
            else:
                boxes = train_dataset.precomputed_boxes[idx]
                if len(boxes) > 0:
                    # Check each box for minimum size
                    for box in boxes:
                        if len(box) >= 4:  # Ensure box has enough values
                            x1, y1, x2, y2 = box[:4]
                            if x2-x1 >= 5 and y2-y1 >= 5:
                                has_valid_boxes = True
                                break
            
            # Append to appropriate list 
            if has_valid_boxes:
                train_valid_indices.append(idx)
            else:
                train_invalid_indices.append(idx)
        
        # Similar filtering for test set 
        test_valid_indices = []
        test_invalid_indices = []
        
        # Limit how many test samples we check if max_datapts is specified
        test_limit = min(len(test_dataset), max_datapts) if max_datapts else len(test_dataset)
        
        for idx in range(test_limit):
            has_valid_boxes = False
            if test_dataset.craft_h5_file is not None:
                if str(idx) in test_dataset.boxes_group:
                    boxes = test_dataset.boxes_group[str(idx)][()]
                    if boxes.size > 0:
                        # Check each box for minimum size
                        for box in boxes:
                            if box.shape[0] >= 4:
                                x1, y1, x2, y2 = box[:4]
                                if x2-x1 >= 5 and y2-y1 >= 5:
                                    has_valid_boxes = True
                                    break
            else:
                boxes = test_dataset.precomputed_boxes[idx]
                if len(boxes) > 0:
                    # Check each box for minimum size
                    for box in boxes:
                        if len(box) >= 4:
                            x1, y1, x2, y2 = box[:4]
                            if x2-x1 >= 5 and y2-y1 >= 5:
                                has_valid_boxes = True
                                break
            
            if has_valid_boxes:
                test_valid_indices.append(idx)
            else:
                test_invalid_indices.append(idx)
        
        # Print statistics
        print(f"\nFiltering statistics:")
        print(f"Train set: {len(train_valid_indices)}/{len(train_dataset)} valid samples "
              f"({len(train_valid_indices)/len(train_dataset)*100:.2f}%)")
        print(f"Test set: {len(test_valid_indices)}/{len(test_dataset)} valid samples "
              f"({len(test_valid_indices)/len(test_dataset)*100:.2f}%)")
        
        # Create filtered datasets using torch.utils.data.Subset
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_valid_indices)
        test_dataset = Subset(test_dataset, test_valid_indices)

        # Limit dataset size if max_datapts specified
        if max_datapts is not None:
            from torch.utils.data import Subset
            train_indices = list(range(min(max_datapts, len(train_dataset))))
            test_indices = list(range(min(max_datapts, len(test_dataset))))
            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
            print(f"Limited dataset to {len(train_dataset)} train and {len(test_dataset)} test samples")

        train_dataset.num_classes = num_classes
        train_dataset.label_mapping = label_mapping
        test_dataset.num_classes = num_classes
        test_dataset.label_mapping = label_mapping
        
        if save_problematic_images:
            # Save problematic sample information (optional)
            _save_problematic_samples(data_dir, train_dataset, train_invalid_indices, train=True)
            _save_problematic_samples(data_dir, test_dataset, test_invalid_indices, train=False)

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
        prefetch_factor=2,
        collate_fn=char_collate_fn 
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=char_collate_fn 
    )
    
    return train_loader, test_loader, train_dataset.num_classes

def _save_problematic_samples(data_dir, dataset, invalid_indices, train=True):
    """Save problematic samples for debugging."""
    import os
    from PIL import Image
    import json
    
    # Create debug directory
    debug_dir = os.path.join(data_dir, "debug-patchless")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create subdirectory for train/test
    split = "train" if train else "test"
    split_dir = os.path.join(debug_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Track problematic fonts
    problem_fonts = {}
    
    # Original dataset if using Subset
    orig_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    # Save each problematic image
    for i, idx in enumerate(invalid_indices):
        # Get the image and target
        if hasattr(dataset, 'dataset'):
            # If we're using a Subset, we need the original indices
            img = orig_dataset.data[idx].astype(np.uint8)
            target = int(orig_dataset.targets[idx])
            font_name = orig_dataset.idx_to_font.get(target, f"unknown_{target}")
        else:
            img = dataset.data[idx].astype(np.uint8)
            target = int(dataset.targets[idx])
            font_name = dataset.idx_to_font.get(target, f"unknown_{target}")
        
        # Save the image
        img_path = os.path.join(split_dir, f"{font_name}_{idx}.png")
        Image.fromarray(img).save(img_path)
        
        # Track problematic fonts
        problem_fonts[font_name] = problem_fonts.get(font_name, 0) + 1
    
    # Save font statistics
    stats_path = os.path.join(debug_dir, f"{split}_problem_fonts.json")
    with open(stats_path, 'w') as f:
        json.dump(problem_fonts, f, indent=2)
    
    # Print top problematic fonts
    print(f"\nTop problematic fonts ({split}):")
    for font, count in sorted(problem_fonts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {font}: {count} samples")


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import os
    
    parser = argparse.ArgumentParser(description="Debug CharacterFontDataset with precomputed CRAFT patches")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the font dataset directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--train", action="store_true", help="Use training set instead of test set")
    parser.add_argument("--output_dir", type=str, default="debug_output", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating dataset with precomputed CRAFT patches...")
    dataset = CharacterFontDataset(
        args.data_dir, 
        train=args.train, 
        use_precomputed_craft=True
    )
    
    if dataset.precomputed_boxes is None:
        print("ERROR: Failed to load precomputed CRAFT boxes")
        exit(1)
        
    print(f"Dataset size: {len(dataset)}")
    print(f"Precomputed boxes size: {len(dataset.precomputed_boxes)}")
    
    # Function to visualize patches
    def visualize_sample(idx):
        # Get sample from dataset
        sample = dataset[idx]
        
        if not isinstance(sample, dict) or 'patches' not in sample:
            print(f"Unexpected sample format: {type(sample)}")
            return
            
        # Get data from sample
        patches = sample['patches']
        attention_mask = sample['attention_mask']
        target = sample['labels'].item()
        font_name = dataset.idx_to_font.get(target, f"unknown_{target}")
        
        # Get original image and boxes
        image = dataset.data[idx].copy()
        boxes = dataset.precomputed_boxes[idx]
        
        viz_img = image.copy() # HWC [0, 255] image for drawing boxes
            
        # Create PIL image and draw boxes
        pil_img = Image.fromarray(viz_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw each box
        valid_boxes = 0
        for i, box in enumerate(boxes):
            try:
                if len(box) != 4:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                draw.text((x1, y1-10), str(i), fill=(255, 0, 0))
                valid_boxes += 1
            except Exception as e:
                print(f"Error drawing box {i}: {e}")
                
        # Create figure for visualization
        plt.figure(figsize=(16, 10))
        
        # Plot original image with boxes
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(pil_img))
        plt.title(f"Original with {valid_boxes} boxes\nFont: {font_name}")
        plt.axis('off')
        
        # Plot extracted patches
        num_valid = int(attention_mask.sum().item())
        plt.subplot(1, 2, 2)
        
        if num_valid > 0:
            # Create a grid of patches
            rows = cols = int(np.ceil(np.sqrt(min(num_valid, 25))))
            fig_patches = plt.figure(figsize=(10, 10))
            
            for i in range(min(num_valid, 25)):
                ax = fig_patches.add_subplot(rows, cols, i+1)
                patch = patches[i, 0].numpy()  # First channel
                ax.imshow(patch, cmap='gray')
                ax.set_title(f"Patch {i}")
                ax.axis('off')
                
            plt.tight_layout()
            patch_path = os.path.join(args.output_dir, f"sample_{idx}_patches.png")
            fig_patches.savefig(patch_path)
            plt.close(fig_patches)
            
            plt.text(0.5, 0.5, f"{num_valid} patches extracted\nSaved detailed view to {os.path.basename(patch_path)}", 
                    ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, "No valid patches extracted!", 
                    ha='center', va='center', fontsize=16, color='red')
            
        plt.axis('off')
        
        # Add overall title
        plt.suptitle(f"Sample {idx}: Font '{font_name}' - {valid_boxes} boxes, {num_valid} valid patches", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(args.output_dir, f"sample_{idx}_overview.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
        return num_valid, valid_boxes
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    if num_samples >= 5:
        # Sample randomly 
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        # Just use first few samples
        indices = list(range(num_samples))
    
    results = []
    for i, idx in enumerate(indices):
        print(f"\nProcessing sample {i+1}/{num_samples} (index {idx})...")
        num_patches, num_boxes = visualize_sample(idx)
        results.append((idx, num_patches, num_boxes))
    
    # Print summary
    print("\nSummary:")
    print("Index | Valid Patches | Valid Boxes")
    print("-" * 35)
    for idx, patches, boxes in results:
        print(f"{idx:5d} | {patches:13d} | {boxes:10d}")
    
    print(f"\nVisualizations saved to {args.output_dir}")