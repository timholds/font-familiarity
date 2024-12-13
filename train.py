import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import FontEncoder, ContrastiveLoss
from typing import List, Dict, Tuple

@torch.compile
class FontSimilarityModel:
    def __init__(self, latent_dim: int = 128):
        self.encoder = FontEncoder(latent_dim)
        self.criterion = ContrastiveLoss()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.encoder.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            embeddings = self.encoder(images)
            loss = self.criterion(embeddings, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    

    def compute_centroids(self, dataloader: DataLoader) -> Dict[int, torch.Tensor]:
        self.encoder.eval()
        centroids = {}
        sample_counts = {}
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                embeddings = self.encoder(images)
                
                for emb, label in zip(embeddings, labels):
                    label = label.item()
                    if label not in centroids:
                        centroids[label] = emb
                        sample_counts[label] = 1
                    else:
                        centroids[label] += emb
                        sample_counts[label] += 1
        
        # Compute averages and normalize
        for label in centroids:
            centroids[label] = centroids[label] / sample_counts[label]
            centroids[label] = F.normalize(centroids[label], p=2, dim=0)
        
        return centroids
    
    def find_similar_fonts(self, query_idx: int, centroids: Dict[int, torch.Tensor], 
                          k: int = 5) -> List[Tuple[int, float]]:
        query_centroid = centroids[query_idx]
        similarities = []
        
        for idx, centroid in centroids.items():
            if idx != query_idx:
                sim = F.cosine_similarity(query_centroid.unsqueeze(0), 
                                        centroid.unsqueeze(0))
                similarities.append((idx, sim.item()))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
