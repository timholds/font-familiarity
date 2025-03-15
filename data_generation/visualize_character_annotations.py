#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
import random
import matplotlib.pyplot as plt

def draw_annotations(image_path, annotation_path, json_path=None, classes_path=None, output_dir=None):
    """Draw character bounding boxes on an image and save or display the result."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error loading image: {image_path}")
        return False
    
    # Create a copy for drawing
    vis_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    # Load class mappings if available
    class_mapping = {}
    if classes_path and Path(classes_path).exists():
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_id = int(parts[0])
                    char = parts[1]
                    class_mapping[class_id] = char
    
    
    # Read YOLO annotations
    annotations = []
    if Path(annotation_path).exists():
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = int((x_center - width/2) * img_width)
                    y = int((y_center - height/2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    
                    annotation = {
                        'class_id': class_id,
                        'bbox': (x, y, w, h)
                    }
                    
                    # Add character info if available
                    if class_id in class_mapping:
                        annotation['char'] = class_mapping[class_id]
                        
                    annotations.append(annotation)
    
    print(f"Found {len(class_mapping)} character mappings")
    print(f"Annotation has {len(annotations)} bounding boxes")

    # Print missing mappings
    missing_labels = []
    for anno in annotations:
        if anno['class_id'] not in class_mapping:
            missing_labels.append(anno['class_id'])
    if missing_labels:
        print(f"Warning: Missing class mappings for IDs: {missing_labels}")
    # Draw bounding boxes
    for anno in annotations:
        x, y, w, h = anno['bbox']
        
        # Generate a consistent color based on class ID
        random.seed(anno['class_id'])
        color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 1)
        
        # Draw character label if available
        if 'char' in anno:
            char_text = anno['char']
            # Put the text at the top-left corner of the bounding box
            cv2.putText(
                vis_image, 
                char_text, 
                (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,  # Smaller font size
                color, 
                1
            )
    
    # Convert from BGR to RGB for matplotlib
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    # Save or display the visualization
    if output_dir:
        output_path = Path(output_dir) / f"{Path(image_path).stem}_viz.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"Saved visualization to {output_path}")
        return True
    else:
        # Display with matplotlib (better for notebooks and interactive sessions)
        plt.figure(figsize=(12, 10))
        plt.imshow(vis_image_rgb)
        plt.title(f"Characters detected: {len(annotations)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return True

def visualize_font_dataset(dataset_dir, font_name=None, max_images=5, output_dir=None):
    """Visualize character detections for a font dataset."""
    dataset_path = Path(dataset_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Get list of fonts to process
    font_dirs = []
    if font_name:
        # Process specific font
        font_dir = dataset_path / font_name.lower().replace(' ', '_')
        if font_dir.exists() and font_dir.is_dir():
            font_dirs.append(font_dir)
        else:
            print(f"Font directory not found: {font_dir}")
            return
    else:
        # Process all fonts
        font_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for font_dir in font_dirs:
        print(f"Processing font: {font_dir.name}")
        
        # Look for classes.txt
        classes_path = font_dir / "classes.txt"
        if not classes_path.exists():
            classes_path = font_dir / "annotations" / "classes.txt"
        
        if not classes_path.exists():
            print(f"Warning: No classes.txt found for {font_dir.name}")
            classes_path = None
        
        # Find all images
        images = list(font_dir.glob("*.jpg")) + list(font_dir.glob("*.png"))
        
        # Limit to max_images
        images = sorted(images)[:max_images]
        
        for img_path in images:
            print(f"  Visualizing {img_path.name}")
            
            # Find corresponding annotation file
            anno_path = font_dir / "annotations" / f"{img_path.stem}.txt"
            json_path = font_dir / "annotations" / f"{img_path.stem}.json"
            
            if not anno_path.exists():
                print(f"    No annotation file found for {img_path.name}")
                continue
            
            # Create visualization
            if output_dir:
                font_output_dir = Path(output_dir) / font_dir.name
                font_output_dir.mkdir(exist_ok=True, parents=True)
            else:
                font_output_dir = None
                
            draw_annotations(
                img_path, 
                anno_path, 
                json_path if json_path.exists() else None,
                classes_path,
                font_output_dir
            )

def main():
    parser = argparse.ArgumentParser(description="Visualize character detection annotations in a font dataset")
    parser.add_argument("--dataset_dir", help="Path to the font dataset directory")
    parser.add_argument("--font", help="Specific font to visualize (defaults to all fonts)")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of images to visualize per font")
    parser.add_argument("--output-dir", help="Directory to save visualizations (if not specified, displays on screen)")
    
    args = parser.parse_args()
    
    visualize_font_dataset(
        args.dataset_dir,
        args.font,
        args.max_images,
        args.output_dir
    )

if __name__ == "__main__":
    main()