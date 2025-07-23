#!/usr/bin/env python3
"""
Script to analyze dataset statistics, starting with patch distribution per image.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def analyze_patch_distribution(h5_file_path):
    """
    Analyze the distribution of patches (characters) per image in the dataset.
    
    Args:
        h5_file_path: Path to the craft_boxes.h5 file
        
    Returns:
        patch_counts: List of patch counts per image
    """
    patch_counts = []
    
    with h5py.File(h5_file_path, 'r') as f:
        boxes_group = f['boxes']
        
        # Get all indices (stored as string keys)
        indices = list(boxes_group.keys())
        print(f"Total images in dataset: {len(indices)}")
        
        # Count boxes for each image
        for idx in tqdm(indices, desc="Counting patches"):
            boxes = boxes_group[idx][()]
            if boxes.size > 0:
                # Each box has 4 coordinates, so divide by 4 to get number of boxes
                num_boxes = len(boxes) if boxes.ndim == 2 else 0
            else:
                num_boxes = 0
            patch_counts.append(num_boxes)
    
    return patch_counts


def plot_patch_distribution(patch_counts, output_path='patch_distribution.png'):
    """
    Create a histogram of patch counts and display statistics.
    
    Args:
        patch_counts: List of patch counts per image
        output_path: Path to save the plot
    """
    patch_counts = np.array(patch_counts)
    
    # Calculate statistics
    stats = {
        'min': np.min(patch_counts),
        'max': np.max(patch_counts),
        'mean': np.mean(patch_counts),
        'median': np.median(patch_counts),
        'std': np.std(patch_counts),
        '25th_percentile': np.percentile(patch_counts, 25),
        '75th_percentile': np.percentile(patch_counts, 75),
        '90th_percentile': np.percentile(patch_counts, 90),
        '95th_percentile': np.percentile(patch_counts, 95),
        '99th_percentile': np.percentile(patch_counts, 99)
    }
    
    # Print statistics
    print("\n=== Patch Count Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Count images with different patch ranges
    print(f"\nImages with 0 patches: {np.sum(patch_counts == 0)}")
    print(f"Images with 1-10 patches: {np.sum((patch_counts >= 1) & (patch_counts <= 10))}")
    print(f"Images with 11-50 patches: {np.sum((patch_counts >= 11) & (patch_counts <= 50))}")
    print(f"Images with 51-100 patches: {np.sum((patch_counts >= 51) & (patch_counts <= 100))}")
    print(f"Images with >100 patches: {np.sum(patch_counts > 100)}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Full histogram
    ax1.hist(patch_counts, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.1f}")
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.1f}")
    ax1.axvline(100, color='orange', linestyle='--', linewidth=2, label="max_chars=100")
    ax1.set_xlabel('Number of Patches per Image')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Distribution of Character Patches per Image (Full Range)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed histogram (0-150 patches)
    max_display = min(150, np.max(patch_counts))
    ax2.hist(patch_counts[patch_counts <= max_display], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.1f}")
    ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.1f}")
    ax2.axvline(100, color='orange', linestyle='--', linewidth=2, label="max_chars=100")
    ax2.set_xlabel('Number of Patches per Image')
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Distribution of Character Patches per Image (Zoomed: 0-150)')
    ax2.set_xlim(0, max_display)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text box with key statistics
    textstr = f'Total Images: {len(patch_counts)}\n'
    textstr += f'Mean: {stats["mean"]:.1f}\n'
    textstr += f'Median: {stats["median"]:.1f}\n'
    textstr += f'90th percentile: {stats["90th_percentile"]:.1f}\n'
    textstr += f'95th percentile: {stats["95th_percentile"]:.1f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.7, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset statistics')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/timholds/code/Font-Familiarity/data/dataset-384-3000spc-rect',
                       help='Path to dataset directory')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                       help='Which dataset split to analyze')
    parser.add_argument('--output', type=str, default='patch_distribution.png',
                       help='Output path for the plot')
    
    args = parser.parse_args()
    
    # Construct path to craft boxes file
    h5_file_path = os.path.join(args.data_dir, f'{args.mode}_craft_boxes.h5')
    
    if not os.path.exists(h5_file_path):
        print(f"Error: File not found: {h5_file_path}")
        return
    
    print(f"Analyzing patch distribution from: {h5_file_path}")
    
    # Analyze patch distribution
    patch_counts = analyze_patch_distribution(h5_file_path)
    
    # Plot and save results
    stats = plot_patch_distribution(patch_counts, args.output)
    
    # Suggest max_chars values based on percentiles
    print("\n=== Suggested max_chars values ===")
    print(f"To cover 90% of images: max_chars = {int(np.ceil(stats['90th_percentile']))}")
    print(f"To cover 95% of images: max_chars = {int(np.ceil(stats['95th_percentile']))}")
    print(f"To cover 99% of images: max_chars = {int(np.ceil(stats['99th_percentile']))}")
    print(f"Current max_chars = 100 covers {np.sum(np.array(patch_counts) <= 100) / len(patch_counts) * 100:.1f}% of images")


if __name__ == '__main__':
    main()