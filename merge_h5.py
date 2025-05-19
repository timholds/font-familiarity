#!/usr/bin/env python3
import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm

def merge_dataset_files(file1, file2, output_file):
    """
    Merge two H5 dataset files (train.h5 or test.h5)
    
    Args:
        file1: Path to the first dataset file
        file2: Path to the second dataset file
        output_file: Path to save the merged dataset
    
    Returns:
        int: Size of the first dataset (needed for offset)
    """
    with h5py.File(file1, 'r') as h5f1, \
         h5py.File(file2, 'r') as h5f2, \
         h5py.File(output_file, 'w') as h5out:
        
        # Get dataset shapes
        images_shape = h5f1['images'].shape
        labels_shape = h5f1['labels'].shape
        
        dataset2_images_shape = h5f2['images'].shape
        dataset2_labels_shape = h5f2['labels'].shape
        
        # Validate dataset structure compatibility
        if images_shape[1:] != dataset2_images_shape[1:]:
            raise ValueError(f"Image dimensions don't match: {images_shape[1:]} vs {dataset2_images_shape[1:]}")
        if labels_shape[1:] != dataset2_labels_shape[1:]:
            raise ValueError(f"Label dimensions don't match: {labels_shape[1:]} vs {dataset2_labels_shape[1:]}")
        
        dataset1_size = images_shape[0]
        total_images = dataset1_size + dataset2_images_shape[0]
        
        print(f"Dataset 1: {dataset1_size} images")
        print(f"Dataset 2: {dataset2_images_shape[0]} images")
        print(f"Total images after merge: {total_images}")
        
        # Create datasets in the output file with combined dimensions
        images_out = h5out.create_dataset(
            'images', 
            shape=(total_images, *images_shape[1:]),
            dtype=h5f1['images'].dtype,
            chunks=True,
            compression='gzip'
        )
        
        labels_out = h5out.create_dataset(
            'labels', 
            shape=(total_images, *labels_shape[1:]),
            dtype=h5f1['labels'].dtype,
            chunks=True,
            compression='gzip'
        )
        
        # Copy data from first dataset
        print("Copying dataset 1 images and labels...")
        for i in tqdm(range(dataset1_size)):
            images_out[i] = h5f1['images'][i]
            labels_out[i] = h5f1['labels'][i]
        
        # Append data from second dataset
        print("Copying dataset 2 images and labels...")
        for i in tqdm(range(dataset2_images_shape[0])):
            images_out[dataset1_size + i] = h5f2['images'][i]
            labels_out[dataset1_size + i] = h5f2['labels'][i]
            
        # Copy any additional dataset attributes if present
        for attr_name in h5f1.attrs:
            h5out.attrs[attr_name] = h5f1.attrs[attr_name]
            
        return dataset1_size

def merge_boxes_files(boxes1_file, boxes2_file, output_boxes_file, dataset1_size):
    """
    Merge two CRAFT boxes files with index remapping
    
    Args:
        boxes1_file: Path to the first boxes file
        boxes2_file: Path to the second boxes file
        output_boxes_file: Path to save the merged boxes
        dataset1_size: Size of the first dataset (for index offset)
    """
    with h5py.File(boxes1_file, 'r') as h5f1, \
         h5py.File(boxes2_file, 'r') as h5f2, \
         h5py.File(output_boxes_file, 'w') as h5out:
        
        # Create boxes group in output file
        boxes_group = h5out.create_group('boxes')
        
        # Count number of box entries in each dataset
        boxes1_count = len(h5f1['boxes'].keys())
        boxes2_count = len(h5f2['boxes'].keys())
        
        print(f"Dataset 1: {boxes1_count} box entries")
        print(f"Dataset 2: {boxes2_count} box entries")
        
        # Copy boxes from first dataset
        print("Copying boxes from dataset 1...")
        for idx in tqdm(h5f1['boxes'].keys()):
            boxes = h5f1['boxes'][idx][:]
            boxes_group.create_dataset(
                name=idx,
                data=boxes,
                compression="gzip"
            )
        
        # Copy boxes from second dataset with adjusted indices
        print("Copying boxes from dataset 2 with adjusted indices...")
        for idx in tqdm(h5f2['boxes'].keys()):
            boxes = h5f2['boxes'][idx][:]
            new_idx = str(dataset1_size + int(idx))
            boxes_group.create_dataset(
                name=new_idx,
                data=boxes,
                compression="gzip"
            )

def validate_merged_files(dataset_file, boxes_file):
    """
    Validate that the merged files have consistent data
    
    Args:
        dataset_file: Path to the merged dataset file
        boxes_file: Path to the merged boxes file
    """
    with h5py.File(dataset_file, 'r') as h5f_data, \
         h5py.File(boxes_file, 'r') as h5f_boxes:
        
        dataset_size = len(h5f_data['images'])
        boxes_count = len(h5f_boxes['boxes'].keys())
        
        # Check if we have at least some boxes entries
        if boxes_count == 0:
            print("WARNING: No boxes found in the merged file!")
            return False
            
        # Sample a few random indices to check for content
        import random
        sample_indices = random.sample(range(dataset_size), min(5, dataset_size))
        
        print(f"\nValidation results:")
        print(f"Total images in merged dataset: {dataset_size}")
        print(f"Total box entries in merged file: {boxes_count}")
        
        # Check if all images have corresponding boxes
        missing_boxes = 0
        for i in range(dataset_size):
            if str(i) not in h5f_boxes['boxes']:
                missing_boxes += 1
                
        if missing_boxes > 0:
            print(f"WARNING: {missing_boxes} images don't have corresponding box entries")
        else:
            print("All images have corresponding box entries ✓")
            
        # Sample check
        print("\nSample validation:")
        for idx in sample_indices:
            idx_str = str(idx)
            if idx_str in h5f_boxes['boxes']:
                boxes = h5f_boxes['boxes'][idx_str][:]
                print(f"  Image {idx}: {boxes.shape[0]} boxes found")
            else:
                print(f"  Image {idx}: No boxes found!")
                
        return missing_boxes == 0

def update_checkpoint(output_dir, mode, total_images):
    """
    Create a new checkpoint file with the latest index
    
    Args:
        output_dir: Directory to save the checkpoint
        mode: Dataset mode ('train' or 'test')
        total_images: Total number of images
    """
    checkpoint_file = os.path.join(output_dir, f'{mode}_craft_checkpoint.txt')
    with open(checkpoint_file, 'w') as f:
        f.write(str(total_images - 1))  # Last processed index
    print(f"Updated checkpoint file: {checkpoint_file}")

def merge_h5_datasets(dataset1_dir, dataset2_dir, output_dir, mode='train'):
    """
    Merge two H5 datasets and their corresponding CRAFT boxes
    
    Args:
        dataset1_dir: Directory containing the first dataset
        dataset2_dir: Directory containing the second dataset
        output_dir: Directory to save the merged dataset
        mode: Dataset mode ('train' or 'test')
    """
    print(f"\n{'=' * 50}")
    print(f"Merging {mode} datasets")
    print(f"{'=' * 50}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths for main dataset files
    dataset1_file = os.path.join(dataset1_dir, f'{mode}.h5')
    dataset2_file = os.path.join(dataset2_dir, f'{mode}.h5')
    output_file = os.path.join(output_dir, f'{mode}.h5')
    
    # Paths for CRAFT boxes files
    boxes1_file = os.path.join(dataset1_dir, f'{mode}_craft_boxes.h5')
    boxes2_file = os.path.join(dataset2_dir, f'{mode}_craft_boxes.h5')
    output_boxes_file = os.path.join(output_dir, f'{mode}_craft_boxes.h5')
    
    # Check if input files exist
    if not os.path.exists(dataset1_file):
        raise FileNotFoundError(f"Dataset 1 file not found: {dataset1_file}")
    if not os.path.exists(dataset2_file):
        raise FileNotFoundError(f"Dataset 2 file not found: {dataset2_file}")
    if not os.path.exists(boxes1_file):
        raise FileNotFoundError(f"Boxes 1 file not found: {boxes1_file}")
    if not os.path.exists(boxes2_file):
        raise FileNotFoundError(f"Boxes 2 file not found: {boxes2_file}")
    
    # 1. Merge main dataset files and get the size of dataset1
    print(f"\nStep 1: Merging main dataset files...")
    dataset1_size = merge_dataset_files(dataset1_file, dataset2_file, output_file)
    
    # 2. Merge CRAFT boxes files
    print(f"\nStep 2: Merging CRAFT boxes files...")
    merge_boxes_files(boxes1_file, boxes2_file, output_boxes_file, dataset1_size)
    
    # 3. Validate the merged files
    print(f"\nStep 3: Validating merged files...")
    with h5py.File(output_file, 'r') as h5f:
        total_images = len(h5f['images'])
    
    is_valid = validate_merged_files(output_file, output_boxes_file)
    
    # 4. Create a new checkpoint file
    update_checkpoint(output_dir, mode, total_images)
    
    print(f"\n{'=' * 50}")
    print(f"Merge completed for {mode} datasets")
    print(f"Total images: {total_images}")
    print(f"Validation {'succeeded' if is_valid else 'had warnings'}")
    print(f"{'=' * 50}\n")

def main():
    parser = argparse.ArgumentParser(description="Merge two H5 datasets and their CRAFT boxes")
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first dataset directory")
    parser.add_argument("--dataset2", type=str, required=True, help="Path to the second dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--modes", type=str, default="train,test", help="Dataset modes to merge (comma-separated)")
    args = parser.parse_args()
    
    # Process each mode
    modes = args.modes.split(',')
    for mode in modes:
        mode = mode.strip()
        try:
            merge_h5_datasets(args.dataset1, args.dataset2, args.output, mode)
        except Exception as e:
            print(f"Error merging {mode} datasets: {e}")
    
    print("All requested merges completed!")

if __name__ == "__main__":
    main()