import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import argparse
import shutil

def create_stratified_split(paths, labels, test_size=0.2, seed=42):
    """Create a stratified train/test split without using sklearn"""
    # Set random seed
    np.random.seed(seed)
    
    # Group indices by label
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    # Split each class proportionally
    for label in label_indices:
        indices = label_indices[label]
        np.random.shuffle(indices)
        
        # Calculate split point
        split = int(len(indices) * (1 - test_size))
        
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    
    # Return train and test paths/labels
    return (
        [paths[i] for i in train_indices],
        [paths[i] for i in test_indices],
        [labels[i] for i in train_indices],
        [labels[i] for i in test_indices]
    )

def save_dataset(image_paths, labels, filename):
    """Load images and save as single compressed file"""
    images = []
    
    for img_path in tqdm(image_paths):
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                images.append(img_array)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    np.savez_compressed(filename, 
                       images=np.array(images),
                       labels=np.array(labels))
    
    file_size = os.path.getsize(filename) / (1024 ** 3)  # Size in GB
    print(f"Saved {filename} with {len(images)} images ({file_size:.2f} GB)")

def save_dataset_in_batches(image_paths, labels, filename, batch_size=1000):
    """Load and save images in batches using HDF5 to avoid memory issues"""
    import h5py
    h5_filename = filename if filename.endswith('.h5') else f"{filename}.h5"

    total_images = len(image_paths)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division
    
    # Get the shape and dtype of the first image to initialize the dataset
    with Image.open(image_paths[0]) as sample_img:
        sample_array = np.array(sample_img)
        img_shape = sample_array.shape
        img_dtype = sample_array.dtype
    
    # Create HDF5 file and initialize datasets
    with h5py.File(filename, 'w') as f:
        # Create resizable datasets
        images_dataset = f.create_dataset(
            'images', 
            shape=(total_images, *img_shape),
            dtype=img_dtype,
            chunks=(1, *img_shape),  # Chunking for efficient access
            compression='gzip'       # Use compression to save space
        )
        
        labels_dataset = f.create_dataset(
            'labels',
            shape=(total_images,),
            dtype=np.int32,
            chunks=(min(1000, total_images),),
            compression='gzip'
        )
        
        # Process images in batches
        current_idx = 0
        for batch in range(num_batches):
            print(f"Processing batch {batch+1}/{num_batches}...")
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, total_images)
            batch_size_actual = end_idx - start_idx
            
            batch_images = []
            batch_indices = []
            
            # Load this batch of images
            for i, img_path in enumerate(tqdm(image_paths[start_idx:end_idx])):
                try:
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        if img_array.shape == img_shape:  # Ensure consistent shapes
                            batch_images.append(img_array)
                            batch_indices.append(start_idx + i)
                        else:
                            print(f"Skipping {img_path}: inconsistent shape {img_array.shape}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Write this batch to the HDF5 file
            if batch_images:
                batch_images_array = np.stack(batch_images)
                batch_labels = np.array([labels[i] for i in batch_indices])
                
                # Write to the correct positions in the datasets
                images_dataset[batch_indices] = batch_images_array
                labels_dataset[batch_indices] = batch_labels
                
                current_idx += len(batch_images)
                
            # Flush to disk after each batch
            f.flush()
    
    # Create a companion NPZ file with just the labels for quick loading
    filename = filename if filename.endswith('.h5') else f"{filename}.h5"

    
    file_size = os.path.getsize(filename) / (1024 ** 3)  # Size in GB
    print(f"Saved {filename} with {current_idx} images ({file_size:.2f} GB)")
    
    return filename


def process_dataset(input_image_dir, output_dir, test_size=0.2, include_annotations=False):
    """Create and save train/test datasets"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create label mapping, filtering out hidden files and non-directories
    font_dirs = [
        d for d in sorted(os.listdir(input_image_dir))
        if not d.startswith('.') and os.path.isdir(os.path.join(input_image_dir, d))
    ]
    label_mapping = {name: idx for idx, name in enumerate(font_dirs)}
    
    # Print label mapping info
    print("\nLabel Mapping Summary:")
    print(f"Number of fonts: {len(label_mapping)}")
    print("\nFirst 5 fonts:")
    for name, idx in sorted(label_mapping.items())[:5]:
        print(f"  {name}: {idx}")
    
    # Collect all paths and labels
    print("\nCollecting file paths...")
    image_paths = []
    labels = []
    
    for font_name in tqdm(font_dirs):  # Use filtered font_dirs here
        font_dir = os.path.join(input_image_dir, font_name)
        if not os.path.isdir(font_dir):
            continue
            
        label = label_mapping[font_name]
        for img_name in os.listdir(font_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(font_dir, img_name))
                labels.append(label)
    
    if include_annotations:
        print("\nCopying annotation files...")
        for font_name in tqdm(font_dirs):
            font_dir = os.path.join(input_image_dir, font_name)
            if not os.path.isdir(font_dir):
                continue

            # Check for and copy annotations directory
            ann_dir = os.path.join(font_dir, "annotations")
            if os.path.isdir(ann_dir):
                dst_dir = os.path.join(output_dir, font_name, "annotations")
                os.makedirs(dst_dir, exist_ok=True)
                
                for ann_file in os.listdir(ann_dir):
                    src_file = os.path.join(ann_dir, ann_file)
                    dst_file = os.path.join(dst_dir, ann_file)
                    shutil.copy2(src_file, dst_file)
            
            # Check for and copy classes.txt
            classes_file = os.path.join(font_dir, "classes.txt")
            if os.path.exists(classes_file):
                dst_file = os.path.join(output_dir, font_name, "classes.txt")
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(classes_file, dst_file)
                
            # Also copy a general classes.txt to the root if it exists
            if os.path.exists(classes_file) and not os.path.exists(os.path.join(output_dir, "classes.txt")):
                shutil.copy2(classes_file, os.path.join(output_dir, "classes.txt"))


    # Create train/test split
    print("\nCreating train/test split...")
    train_paths, test_paths, train_labels, test_labels = create_stratified_split(
        image_paths, labels, test_size=test_size
    )
    
    # Save label mapping
    mapping_path = os.path.join(output_dir, 'label_mapping.npy')
    np.save(mapping_path, label_mapping)
    print(f"\nSaved label mapping to {mapping_path}")
    
    # Process and save training data
    print("\nProcessing training data...")
    save_dataset_in_batches(train_paths, train_labels, os.path.join(output_dir, 'train.h5'))
    
    # Process and save test data
    print("\nProcessing test data...")
    save_dataset_in_batches(test_paths, test_labels, os.path.join(output_dir, 'test.h5'))
    
    print(f"\nComplete! Saved to {output_dir}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Number of classes: {len(label_mapping)}")

def main():
    parser = argparse.ArgumentParser(description="Prepare train/test datasets")
    parser.add_argument("--input_image_dir", default="data/font-images", help="Root directory of font images")
    parser.add_argument("--output_dir", default="font_dataset_npz", help="Output directory for processed datasets")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--detection_labels", action="store_true", help="Include character annotations in the output dataset")
    args = parser.parse_args()

    process_dataset(args.input_image_dir, args.output_dir, args.test_size, args.detection_labels)

if __name__ == "__main__":
    main()
    
