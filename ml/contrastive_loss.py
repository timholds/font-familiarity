import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustCategoryContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, min_positives=2):
        super().__init__()
        self.temperature = temperature
        self.min_positives = min_positives

    def forward(self, embeddings, categories, font_labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Create masks
        categories = categories.unsqueeze(1)  # [B, 1]
        positive_mask = (categories == categories.T)  # [B, B] 
        positive_mask.fill_diagonal_(False)  # Exclude self
        
        # Only proceed if we have positive pairs
        if not positive_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute similarities
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # For each row, get positive similarities
        pos_sim = torch.where(positive_mask, sim_matrix, torch.tensor(-float('inf')).to(device))
        pos_sim = torch.logsumexp(pos_sim, dim=1)  # [B]
        
        # For each row, get all similarities (excluding self)
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        all_sim = torch.where(neg_mask, sim_matrix, torch.tensor(-float('inf')).to(device))
        all_sim = torch.logsumexp(all_sim, dim=1)  # [B]
        
        # Only compute loss for samples that have positive pairs
        has_positives = positive_mask.any(dim=1)
        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = (-pos_sim + all_sim)[has_positives]
        return loss.mean()
