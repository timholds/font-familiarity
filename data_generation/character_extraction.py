import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import shutil

def extract_character_patches(dataset_dir, output_dir, padding_ratio=0.1, min_size=20):
    """
    Extract individual character patches from the annotated dataset
    
    Args:
        dataset_dir: Root directory of the annotated dataset
        output_dir: Directory to save extracted character patches
        padding_ratio: How much padding to add around characters (as a ratio of character size)
        min_size: Minimum size (width/height) for extracted patches
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_images = 0
    total_characters = 0
    chars_by_font = {}
    skipped_characters = 0
    
    # Process all font directories
    for font_dir in tqdm(list(Path(dataset_dir).glob('*')), desc="Processing fonts"):
        font_name = font_dir.name
        chars_by_font[font_name] = {}
        
        # Process each variation
        for variation_dir in font_dir.glob('*'):
            if not (variation_dir / 'annotations').exists():
                continue
                
            variation_name = variation_dir.name
            
            # Create output directory for this font/variation
            font_var_output = output_path / font_name / variation_name
            font_var_output.mkdir(parents=True, exist_ok=True)
            
            # Process all annotation files
            for anno_file in (variation_dir / 'annotations').glob('*.json'):
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                image_path = variation_dir / 'images' / data['image']
                if not image_path.exists():
                    print(f"Warning: Image {image_path} not found, skipping")
                    continue
                    
                try:
                    # Load image
                    img = Image.open(image_path)
                    total_images += 1
                    
                    # Process each character
                    for i, char_info in enumerate(data['characters']):
                        char = char_info['char']
                        
                        # Skip whitespace and non-printable characters
                        if char.isspace() or ord(char) < 32:
                            skipped_characters += 1
                            continue
                            
                        # Get bounding box with padding
                        x = char_info['x']
                        y = char_info['y']
                        w = char_info['width']
                        h = char_info['height']
                        
                        # Add padding
                        pad_x = w * padding_ratio
                        pad_y = h * padding_ratio
                        
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(data['image_width'], x + w + pad_x)
                        y2 = min(data['image_height'], y + h + pad_y)
                        
                        # Skip if box is too small
                        if x2 - x1 < min_size or y2 - y1 < min_size:
                            skipped_characters += 1
                            continue
                            
                        # Extract patch
                        patch = img.crop((x1, y1, x2, y2))
                        
                        # Create a clean filename - handle special characters
                        safe_char = f"ord{ord(char)}" if not char.isalnum() else char
                        
                        # Create character directory if needed
                        char_dir = font_var_output / safe_char
                        char_dir.mkdir(exist_ok=True)
                        
                        # Save patch
                        patch_filename = char_dir / f"{Path(data['image']).stem}_{i:03d}.png"
                        patch.save(patch_filename)
                        
                        # Update statistics
                        total_characters += 1
                        if char not in chars_by_font[font_name]:
                            chars_by_font[font_name][char] = 0
                        chars_by_font[font_name][char] += 1
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    # Print statistics
    print(f"\nExtracted {total_characters} character patches from {total_images} images")
    print(f"Skipped {skipped_characters} characters (whitespace or too small)")
    
    # Print some per-font statistics
    print("\nTop 5 fonts by character count:")
    font_counts = [(font, sum(counts.values())) for font, counts in chars_by_font.items()]
    font_counts.sort(key=lambda x: x[1], reverse=True)
    for font, count in font_counts[:5]:
        print(f"  {font}: {count} character patches")
    
    # Save statistics
    with open(output_path / "extraction_stats.json", 'w') as f:
        json.dump({
            "total_images": total_images,
            "total_characters": total_characters,
            "skipped_characters": skipped_characters,
            "chars_by_font": chars_by_font
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract character patches from annotated dataset")
    parser.add_argument("--dataset_dir", required=True, help="Path to annotated dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory for character patches")
    parser.add_argument("--padding", type=float, default=0.1, help="Padding ratio around characters")
    parser.add_argument("--min_size", type=int, default=20, help="Minimum character size")
    
    args = parser.parse_args()
    extract_character_patches(args.dataset_dir, args.output_dir, args.padding, args.min_size)aimport os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import random

def extract_character_patches(dataset_dir, output_dir, padding_ratio=0.1, min_size=20, 
                            max_chars_per_font=5000, balance_chars=True):
    """
    Extract individual character patches from the annotated dataset
    
    Args:
        dataset_dir: Root directory of the annotated dataset
        output_dir: Directory to save extracted character patches
        padding_ratio: How much padding to add around characters (as a ratio of character size)
        min_size: Minimum size (width/height) for extracted patches
        max_chars_per_font: Maximum number of samples per font to extract
        balance_chars: Whether to balance character classes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_images = 0
    total_characters = 0
    chars_by_font = {}
    skipped_characters = 0
    all_chars = set()
    
    # First pass - collect statistics on character distribution
    print("Scanning dataset for character distribution...")
    for font_dir in tqdm(list(Path(dataset_dir).glob('*')), desc="Analyzing fonts"):
        font_name = font_dir.name
        chars_by_font[font_name] = {}
        
        # Process all variations and annotations to gather statistics
        for variation_dir in font_dir.glob('*'):
            annotation_dir = variation_dir / 'annotations'
            if not annotation_dir.exists():
                continue
                
            for anno_file in annotation_dir.glob('*.json'):
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                # Collect stats on characters
                for char_info in data.get('characters', []):
                    char = char_info.get('char', '')
                    
                    # Skip whitespace and non-printable characters
                    if not char or char.isspace() or ord(char) < 32:
                        continue
                        
                    all_chars.add(char)
                    if char not in chars_by_font[font_name]:
                        chars_by_font[font_name][char] = 0
                    chars_by_font[font_name][char] += 1
    
    # Determine how many samples to extract per character for each font
    print(f"Found {len(all_chars)} unique characters across {len(chars_by_font)} fonts")
    
    # Calculate extraction limits
    char_limits = {}
    if balance_chars:
        # Determine a balanced limit per character across all fonts
        for font_name, char_counts in chars_by_font.items():
            total_chars = sum(char_counts.values())
            if total_chars > max_chars_per_font:
                # Need to limit extraction - compute per-character limit
                scale_factor = max_chars_per_font / total_chars
                char_limits[font_name] = {
                    char: max(1, int(count * scale_factor))
                    for char, count in char_counts.items()
                }
            else:
                # No need to limit
                char_limits[font_name] = {char: count for char, count in char_counts.items()}
    
    # Process all font directories to extract character patches
    for font_dir in tqdm(list(Path(dataset_dir).glob('*')), desc="Extracting characters"):
        font_name = font_dir.name
        current_char_counts = {char: 0 for char in all_chars}
        
        # Process each variation
        for variation_dir in font_dir.glob('*'):
            if not (variation_dir / 'annotations').exists():
                continue
                
            variation_name = variation_dir.name
            
            # Create output directory for this font/variation
            font_var_output = output_path / font_name / variation_name
            
            # Process all annotation files
            for anno_file in (variation_dir / 'annotations').glob('*.json'):
                with open(anno_file, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from {anno_file}")
                        continue
                
                # Skip if invalid data format
                if 'characters' not in data or not isinstance(data['characters'], list):
                    print(f"Invalid annotation format in {anno_file}")
                    continue
                
                image_path = variation_dir / 'images' / data['image']
                if not image_path.exists():
                    print(f"Warning: Image {image_path} not found, skipping")
                    continue
                    
                try:
                    # Load image
                    img = Image.open(image_path)
                    total_images += 1
                    
                    # Process each character
                    for i, char_info in enumerate(data['characters']):
                        # Skip if invalid character info
                        if not all(k in char_info for k in ['char', 'x', 'y', 'width', 'height']):
                            continue
                            
                        char = char_info['char']
                        
                        # Skip whitespace and non-printable characters
                        if char.isspace() or ord(char) < 32:
                            skipped_characters += 1
                            continue
                            
                        # Check if we've hit the limit for this character in this font
                        if (font_name in char_limits and 
                            char in char_limits[font_name] and 
                            current_char_counts[char] >= char_limits[font_name][char]):
                            skipped_characters += 1
                            continue
                            
                        # Get bounding box with padding
                        x = char_info['x']
                        y = char_info['y']
                        w = char_info['width']
                        h = char_info['height']
                        
                        # Skip exceptionally wide or tall characters (likely errors)
                        if w > data['image_width'] * 0.5 or h > data['image_height'] * 0.5:
                            skipped_characters += 1
                            continue
                        
                        # Add padding
                        pad_x = w * padding_ratio
                        pad_y = h * padding_ratio
                        
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(data['image_width'], x + w + pad_x)
                        y2 = min(data['image_height'], y + h + pad_y)
                        
                        # Skip if box is too small
                        if x2 - x1 < min_size or y2 - y1 < min_size:
                            skipped_characters += 1
                            continue
                            
                        # Create character directory if needed
                        char_dir = font_var_output / char
                        char_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Extract patch
                        patch = img.crop((x1, y1, x2, y2))
                        
                        # Save patch - include source file info in filename
                        patch_filename = char_dir / f"{Path(data['image']).stem}_{i:03d}.png"
                        patch.save(patch_filename)
                        
                        # Update statistics
                        total_characters += 1
                        current_char_counts[char] += 1
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    # Print statistics
    print(f"\nExtracted {total_characters} character patches from {total_images} images")
    print(f"Skipped {skipped_characters} characters (whitespace, too small, or over limit)")
    
    # Print per-font statistics
    print("\nTop 5 fonts by character count:")
    font_counts = [(font, sum(1 for var_dir in Path(output_dir).glob(f'{font}/*') 
                             for char_dir in var_dir.glob('*')
                             for _ in char_dir.glob('*.png')))
                  for font in chars_by_font.keys()]
    font_counts.sort(key=lambda x: x[1], reverse=True)
    for font, count in font_counts[:5]:
        print(f"  {font}: {count} character patches")
    
    # Print character statistics
    char_counts = {}
    for font_dir in Path(output_dir).glob('*'):
        for var_dir in font_dir.glob('*'):
            for char_dir in var_dir.glob('*'):
                char = char_dir.name
                if char not in char_counts:
                    char_counts[char] = 0
                char_counts[char] += sum(1 for _ in char_dir.glob('*.png'))
    
    print("\nTop 20 characters by count:")
    top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for char, count in top_chars:
        char_display = f"'{char}'" if len(char) == 1 else f"ord({ord(char)})"
        print(f"  {char_display}: {count} instances")
    
    # Save statistics
    with open(output_path / "extraction_stats.json", 'w') as f:
        json.dump({
            "total_images": total_images,
            "total_characters": total_characters,
            "skipped_characters": skipped_characters,
            "character_counts": {k: v for k, v in sorted(char_counts.items(), key=lambda x: x[1], reverse=True)}
        }, f, indent=2)
    
    return output_path

def convert_to_training_format(dataset_dir, output_dir, format='yolo',
                             train_ratio=0.8, include_font_info=True):
    """
    Convert extracted character patches to standard object detection training formats
    
    Args:
        dataset_dir: Directory with extracted character patches
        output_dir: Output directory for the training dataset
        format: Dataset format ('yolo', 'coco', or 'character_recognition')
        train_ratio: Ratio of data to use for training (remainder for validation)
        include_font_info: Whether to include font information in the labels
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == 'character_recognition':
        # For character recognition, organize patches by character class
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Maps for character and font IDs
        char_to_id = {}
        font_to_id = {}
        
        # First pass - collect all unique characters and fonts
        all_chars = set()
        all_fonts = set()
        
        for font_dir in Path(dataset_dir).glob('*'):
            font_name = font_dir.name
            all_fonts.add(font_name)
            
            for var_dir in font_dir.glob('*'):
                for char_dir in var_dir.glob('*'):
                    char = char_dir.name
                    all_chars.add(char)
        
        # Create ID mappings
        for i, char in enumerate(sorted(all_chars)):
            char_to_id[char] = i
        
        for i, font in enumerate(sorted(all_fonts)):
            font_to_id[font] = i
        
        # Save mappings
        with open(output_path / 'char_mapping.json', 'w') as f:
            json.dump(char_to_id, f, indent=2)
        
        with open(output_path / 'font_mapping.json', 'w') as f:
            json.dump(font_to_id, f, indent=2)
        
        # Second pass - organize patches
        for font_dir in tqdm(list(Path(dataset_dir).glob('*')), desc="Organizing characters"):
            font_name = font_dir.name
            font_id = font_to_id[font_name]
            
            for var_dir in font_dir.glob('*'):
                var_name = var_dir.name
                
                for char_dir in var_dir.glob('*'):
                    char = char_dir.name
                    char_id = char_to_id[char]
                    
                    # Create character class directory
                    train_char_dir = train_dir / str(char_id)
                    val_char_dir = val_dir / str(char_id)
                    train_char_dir.mkdir(exist_ok=True)
                    val_char_dir.mkdir(exist_ok=True)
                    
                    # Process all patches for this character
                    patches = list(char_dir.glob('*.png'))
                    random.shuffle(patches)
                    
                    # Split into train/val
                    train_count = int(len(patches) * train_ratio)
                    train_patches = patches[:train_count]
                    val_patches = patches[train_count:]
                    
                    # Copy patches
                    for i, patch in enumerate(train_patches):
                        # Include font info in filename if requested
                        if include_font_info:
                            new_name = f"{font_id}_{var_name}_{patch.stem}.png"
                        else:
                            new_name = f"{patch.stem}.png"
                        shutil.copy(patch, train_char_dir / new_name)
                    
                    for i, patch in enumerate(val_patches):
                        if include_font_info:
                            new_name = f"{font_id}_{var_name}_{patch.stem}.png"
                        else:
                            new_name = f"{patch.stem}.png"
                        shutil.copy(patch, val_char_dir / new_name)
    
    elif format == 'yolo':
        # Convert to YOLO format for character detection
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        train_images = train_dir / 'images'
        train_labels = train_dir / 'labels'
        val_images = val_dir / 'images'
        val_labels = val_dir / 'labels'
        
        train_images.mkdir(parents=True, exist_ok=True)
        train_labels.mkdir(parents=True, exist_ok=True)
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)
        
        # Collect all unique characters
        all_chars = set()
        for font_dir in Path(dataset_dir).glob('*'):
            for var_dir in font_dir.glob('*'):
                annotation_dir = var_dir / 'annotations'
                if not annotation_dir.exists():
                    continue
                
                for anno_file in annotation_dir.glob('*.json'):
                    with open(anno_file, 'r') as f:
                        try:
                            data = json.load(f)
                            for char_info in data.get('characters', []):
                                char = char_info.get('char', '')
                                if char and not char.isspace() and ord(char) >= 32:
                                    all_chars.add(char)
                        except:
                            continue
        
        # Create character mapping
        chars_list = sorted(all_chars)
        char_to_id = {char: i for i, char in enumerate(chars_list)}
        
        # Save mapping
        with open(output_path / 'classes.txt', 'w') as f:
            for char in chars_list:
                f.write(f"{char}\n")
        
        with open(output_path / 'class_mapping.json', 'w') as f:
            json.dump(char_to_id, f, indent=2)
        
        # Process annotations and images
        all_samples = []
        for font_dir in tqdm(list(Path(dataset_dir).glob('*')), desc="Collecting YOLO annotations"):
            font_name = font_dir.name
            
            for var_dir in font_dir.glob('*'):
                var_name = var_dir.name
                annotation_dir = var_dir / 'annotations'
                image_dir = var_dir / 'images'
                
                if not annotation_dir.exists() or not image_dir.exists():
                    continue
                
                for anno_file in annotation_dir.glob('*.json'):
                    image_name = anno_file.stem + '.jpg'
                    image_path = image_dir / image_name
                    
                    if not image_path.exists():
                        # Try PNG as well
                        image_path = image_dir / (anno_file.stem + '.png')
                        if not image_path.exists():
                            continue
                    
                    all_samples.append((font_name, var_name, anno_file, image_path))
        
        # Shuffle and split
        random.shuffle(all_samples)
        train_count = int(len(all_samples) * train_ratio)
        train_samples = all_samples[:train_count]
        val_samples = all_samples[train_count:]
        
        # Process training samples
        for font_name, var_name, anno_file, image_path in tqdm(train_samples, desc="Processing training samples"):
            try:
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                # Create YOLO annotation
                yolo_annotation = []
                
                img_width = data.get('image_width', 0)
                img_height = data.get('image_height', 0)
                
                if img_width <= 0 or img_height <= 0:
                    # Try to get dimensions from the image
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                
                if img_width <= 0 or img_height <= 0:
                    # Skip if we can't determine dimensions
                    continue
                
                for char_info in data.get('characters', []):
                    char = char_info.get('char', '')
                    
                    # Skip invalid characters
                    if not char or char.isspace() or ord(char) < 32 or char not in char_to_id:
                        continue
                    
                    # Get bounding box
                    x = char_info.get('x', 0)
                    y = char_info.get('y', 0)
                    w = char_info.get('width', 0)
                    h = char_info.get('height', 0)
                    
                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Skip if outside image bounds
                    if (x_center < 0 or x_center > 1 or 
                        y_center < 0 or y_center > 1 or
                        width <= 0 or width > 1 or
                        height <= 0 or height > 1):
                        continue
                    
                    # Add to annotation
                    class_id = char_to_id[char]
                    yolo_annotation.append(f"{class_id} {x_center} {y_center} {width} {height}")
                
                if not yolo_annotation:
                    # Skip if no valid annotations
                    continue
                
                # Save image and annotation
                dest_image = train_images / f"{font_name}_{var_name}_{image_path.stem}.jpg"
                
                # Convert to JPG if needed
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dest_image, quality=95)
                
                # Save annotation
                with open(train_labels / f"{font_name}_{var_name}_{image_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(yolo_annotation))
                    
            except Exception as e:
                print(f"Error processing {anno_file}: {e}")
        
        # Process validation samples - similar to training
        for font_name, var_name, anno_file, image_path in tqdm(val_samples, desc="Processing validation samples"):
            # (similar processing as above, just save to val_* directories)
            # This code is similar to the training sample processing
            try:
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                yolo_annotation = []
                
                img_width = data.get('image_width', 0)
                img_height = data.get('image_height', 0)
                
                if img_width <= 0 or img_height <= 0:
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                
                if img_width <= 0 or img_height <= 0:
                    continue
                
                for char_info in data.get('characters', []):
                    char = char_info.get('char', '')
                    
                    if not char or char.isspace() or ord(char) < 32 or char not in char_to_id:
                        continue
                    
                    x = char_info.get('x', 0)
                    y = char_info.get('y', 0)
                    w = char_info.get('width', 0)
                    h = char_info.get('height', 0)
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    if (x_center < 0 or x_center > 1 or 
                        y_center < 0 or y_center > 1 or
                        width <= 0 or width > 1 or
                        height <= 0 or height > 1):
                        continue
                    
                    class_id = char_to_id[char]
                    yolo_annotation.append(f"{class_id} {x_center} {y_center} {width} {height}")
                
                if not yolo_annotation:
                    continue
                
                dest_image = val_images / f"{font_name}_{var_name}_{image_path.stem}.jpg"
                
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dest_image, quality=95)
                
                with open(val_labels / f"{font_name}_{var_name}_{image_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(yolo_annotation))
                    
            except Exception as e:
                print(f"Error processing {anno_file}: {e}")
    
    elif format == 'coco':
        # COCO format implementation - similar approach but different output format
        print("COCO format conversion not yet implemented")
    
    else:
        print(f"Unknown format: {format}")
        return None
    
    print(f"\nDataset converted to {format} format in {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract character patches and convert to training formats")
    parser.add_argument("--dataset_dir", required=True, help="Path to annotated dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory for character patches")
    parser.add_argument("--padding", type=float, default=0.1, help="Padding ratio around characters")
    parser.add_argument("--min_size", type=int, default=20, help="Minimum character size")
    parser.add_argument("--max_chars_per_font", type=int, default=5000, help="Maximum characters per font")
    parser.add_argument("--balance_chars", action="store_true", help="Balance character classes")
    parser.add_argument("--format", choices=["none", "yolo", "coco", "character_recognition"], 
                      default="none", help="Convert to training format")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio")
    
    args = parser.parse_args()
    
    # Extract character patches
    patches_dir = extract_character_patches(
        args.dataset_dir, 
        args.output_dir, 
        args.padding, 
        args.min_size,
        args.max_chars_per_font,
        args.balance_chars
    )
    
    # Convert to training format if requested
    if args.format != "none":
        convert_dir = Path(args.output_dir) / "training_data"
        convert_to_training_format(
            patches_dir, 
            convert_dir, 
            args.format,
            args.train_ratio
        )