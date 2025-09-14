#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def load_embeddings(filepath):
    """Load embeddings from numpy file"""
    embeddings = np.load(filepath)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings

def calculate_cosine_similarities(embeddings):
    """Calculate pairwise cosine similarities between all embeddings
    Cosine similarity range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
    """
    # Normalize embeddings for cosine similarity
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Calculate cosine similarity matrix
    cosine_sim = np.dot(norm_embeddings, norm_embeddings.T)
    
    # Extract upper triangle (excluding diagonal with self-similarities of 1)
    upper_triangle_indices = np.triu_indices_from(cosine_sim, k=1)
    pairwise_similarities = cosine_sim[upper_triangle_indices]
    
    return pairwise_similarities, cosine_sim

def plot_multiple_histograms(all_similarities, labels, title="Pairwise Cosine Similarities Comparison", bins=50):
    """Create overlaid histograms for multiple embedding files using line plots with markers"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color and marker styles for up to 4 distributions
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    linestyles = ['-', '--', '-.', ':']
    
    # Statistics storage for legend
    stats_info = []
    
    for i, (similarities, label) in enumerate(zip(all_similarities, labels)):
        # Calculate histogram
        counts, bin_edges = np.histogram(similarities, bins=bins)
        # Use bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as line with markers
        ax.plot(bin_centers, counts, 
               color=colors[i], 
               marker=markers[i],
               linestyle=linestyles[i],
               linewidth=1.5,
               markersize=4,
               markevery=max(1, len(bin_centers) // 15),  # Show markers at intervals
               label=label,
               alpha=0.8)
        
        # Calculate statistics for info box
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Store statistics for display
        stats_info.append({
            'label': label,
            'pairs': len(similarities),
            'mean': mean_sim,
            'std': std_sim,
            'min': np.min(similarities),
            'max': np.max(similarities)
        })
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Create simplified statistics text box
    stats_lines = []
    for info in stats_info:
        stats_lines.append(f"{info['label']}:")
        stats_lines.append(f"  N={info['pairs']:,}")
        stats_lines.append(f"  μ={info['mean']:.3f}, σ={info['std']:.3f}")
        stats_lines.append(f"  Range: [{info['min']:.3f}, {info['max']:.3f}]")
        stats_lines.append("")  # Empty line between files
    
    stats_text = '\n'.join(stats_lines[:-1])  # Remove last empty line
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare cosine similarities of multiple embedding files')
    parser.add_argument('embedding_files', nargs='+', type=str, 
                       help='Paths to embedding numpy files (up to 4 files)')
    parser.add_argument('--labels', nargs='*', type=str, 
                       help='Custom labels for each embedding file (defaults to filenames)')
    parser.add_argument('--output', type=str, default='cosine_similarities_comparison.png',
                       help='Output filename for the histogram (default: cosine_similarities_comparison.png)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of histogram bins (default: 50)')
    parser.add_argument('--title', type=str, default='Pairwise Cosine Similarities Comparison',
                       help='Plot title')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot (only save)')
    
    args = parser.parse_args()
    
    # Validate number of files
    if len(args.embedding_files) > 4:
        print("Warning: Maximum of 4 embedding files supported. Using first 4 files.")
        args.embedding_files = args.embedding_files[:4]
    
    # Prepare labels
    if args.labels:
        if len(args.labels) != len(args.embedding_files):
            print("Warning: Number of labels doesn't match number of files. Using filenames.")
            labels = [Path(f).stem for f in args.embedding_files]
        else:
            labels = args.labels
    else:
        labels = [Path(f).stem for f in args.embedding_files]
    
    # Process each embedding file
    all_similarities = []
    
    for i, (filepath, label) in enumerate(zip(args.embedding_files, labels)):
        print(f"\n[{i+1}/{len(args.embedding_files)}] Processing: {filepath}")
        print(f"Label: {label}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            continue
            
        try:
            # Load embeddings
            embeddings = load_embeddings(filepath)
            
            # Calculate pairwise cosine similarities
            print("Calculating pairwise cosine similarities...")
            pairwise_similarities, similarity_matrix = calculate_cosine_similarities(embeddings)
            
            # Print statistics
            print(f"Number of pairwise similarities: {len(pairwise_similarities):,}")
            print(f"Statistics:")
            print(f"  Mean: {np.mean(pairwise_similarities):.3f}")
            print(f"  Median: {np.median(pairwise_similarities):.3f}")
            print(f"  Std: {np.std(pairwise_similarities):.3f}")
            print(f"  Min: {np.min(pairwise_similarities):.3f}")
            print(f"  Max: {np.max(pairwise_similarities):.3f}")
            
            all_similarities.append(pairwise_similarities)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    if not all_similarities:
        print("Error: No valid embedding files were processed.")
        return
    
    # Create comparison plot
    print(f"\nCreating comparison histogram with {len(all_similarities)} embedding files...")
    fig = plot_multiple_histograms(all_similarities, labels[:len(all_similarities)], 
                                   title=args.title, bins=args.bins)
    
    # Save figure
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {args.output}")
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()