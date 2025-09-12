#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_histogram(similarities, title="Pairwise Cosine Similarities", bins=50):
    """Create histogram of pairwise similarities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    counts, bins, patches = ax.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add statistics
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    std_sim = np.std(similarities)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    ax.axvline(median_sim, color='green', linestyle='--', linewidth=2, label=f'Median: {median_sim:.3f}')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Total pairs: {len(similarities):,}\nMean: {mean_sim:.3f}\nMedian: {median_sim:.3f}\nStd: {std_sim:.3f}\nMin: {np.min(similarities):.3f}\nMax: {np.max(similarities):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def main():
    # Load embeddings
    embedding_file = '/home/timholds/code/Font-Familiarity/v5-ED512-epoch10/class_embeddings-BS32-ED512-IC16-PS64-NH8-PX0.05-PY0.15-epoch10-C.3.npy'
    
    print(f"Loading embeddings from: {embedding_file}")
    embeddings = load_embeddings(embedding_file)
    
    # Calculate pairwise cosine similarities
    print("\nCalculating pairwise cosine similarities...")
    pairwise_similarities, similarity_matrix = calculate_cosine_similarities(embeddings)
    
    print(f"\nNumber of pairwise similarities: {len(pairwise_similarities):,}")
    print(f"Similarity statistics:")
    print(f"  Mean: {np.mean(pairwise_similarities):.3f}")
    print(f"  Median: {np.median(pairwise_similarities):.3f}")
    print(f"  Std: {np.std(pairwise_similarities):.3f}")
    print(f"  Min: {np.min(pairwise_similarities):.3f}")
    print(f"  Max: {np.max(pairwise_similarities):.3f}")
    
    # Create histogram
    fig = plot_histogram(pairwise_similarities, 
                        title="Pairwise Cosine Similarities - 512D Embeddings (700 Font Classes)")
    
    # Save figure
    output_file = 'cosine_similarities_histogram_512d.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()