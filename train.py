import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import FontEncoder, ContrastiveLoss

class FontSimilaritySystem:
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