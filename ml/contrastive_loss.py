import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCategoryLoss(nn.Module):
    """Simple category classification loss - much more efficient than contrastive loss"""
    def __init__(self, embedding_dim=1024, num_categories=5):
        super().__init__()
        self.category_classifier = nn.Linear(embedding_dim, num_categories)
    
    def forward(self, embeddings, categories, font_labels):
        # Simple cross-entropy loss on category prediction
        category_logits = self.category_classifier(embeddings)
        return F.cross_entropy(category_logits, categories)

class RobustCategoryContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.7, max_pairs=4):
        super().__init__()
        self.temperature = temperature
        self.max_pairs = max_pairs  # Limit total pairs processed

    def forward(self, embeddings, categories, font_labels):
        device = embeddings.device
        
        # Clear cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Find positive pairs more memory efficiently
        # Avoid creating B×B matrix for large batches
        batch_size = len(categories)
        pos_pairs = []
        
        # More memory-efficient pair finding
        for i in range(batch_size):
            # Find samples with same category as sample i
            same_cat_mask = (categories == categories[i])
            same_cat_indices = torch.where(same_cat_mask)[0]
            
            # Remove self-pair
            same_cat_indices = same_cat_indices[same_cat_indices != i]
            
            # Add pairs (i, j) where j has same category
            for j in same_cat_indices[:1]:  # Limit to 1 positive per anchor for memory
                pos_pairs.append((i, j.item()))
        
        if len(pos_pairs) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Convert to tensor format
        pos_pairs = pos_pairs[:self.max_pairs]  # Limit total pairs
        anchor_idxs = torch.tensor([p[0] for p in pos_pairs], device=device)
        pos_idxs = torch.tensor([p[1] for p in pos_pairs], device=device)
        num_pairs = len(pos_pairs)
        
        # Get anchor and positive embeddings
        anchors = embeddings[anchor_idxs]  # [num_pairs, embed_dim]
        positives = embeddings[pos_idxs]    # [num_pairs, embed_dim]
        
        # Process in smaller chunks to save memory
        losses = []
        chunk_size = min(4, num_pairs)  # Process 4 pairs at a time max
        
        for chunk_start in range(0, num_pairs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_pairs)
            chunk_losses = []
            
            for i in range(chunk_start, chunk_end):
                anchor = anchors[i:i+1]  # [1, embed_dim]
                positive = positives[i:i+1]  # [1, embed_dim]
                anchor_cat = categories[anchor_idxs[i]]
                
                # Get negatives (different category from anchor)
                neg_mask = (categories != anchor_cat)
                if not neg_mask.any():
                    continue
                    
                negatives = embeddings[neg_mask]  # [n_neg, embed_dim]
                
                # Limit negatives very aggressively for memory
                if len(negatives) > 4:
                    neg_perm = torch.randperm(len(negatives))[:4]
                    negatives = negatives[neg_perm]
                
                # Compute similarities
                pos_sim = torch.sum(anchor * positive) / self.temperature  # Scalar
                neg_sims = torch.matmul(anchor, negatives.T).squeeze() / self.temperature  # [n_neg]
                
                # Ensure neg_sims is always 1D, even with single negative
                if neg_sims.dim() == 0:  # Single negative case
                    neg_sims = neg_sims.unsqueeze(0)
                
                # InfoNCE loss
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + n_neg]
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0], device=device))
                chunk_losses.append(loss)
            
            # Add chunk losses and clean up
            if chunk_losses:
                losses.extend(chunk_losses)
                del chunk_losses  # Explicitly free memory
                
                # Clear cache between chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute final loss and clean up
        final_loss = torch.stack(losses).mean()
        del losses  # Free memory
        
        return final_loss

class OptimizedCategoryContrastiveLoss(nn.Module):
    """Optimized contrastive loss using vectorized operations and better memory management"""
    def __init__(self, temperature=0.7, chunk_size=8):
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size  # Process similarities in chunks to save memory
        
    def forward(self, embeddings, categories, font_labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings once
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # For small batches, use full matrix computation
        if batch_size <= 16:
            return self._forward_full(embeddings, categories, device)
        else:
            # For larger batches, use chunked computation
            return self._forward_chunked(embeddings, categories, device)
    
    def _forward_full(self, embeddings, categories, device):
        """Original full matrix computation for small batches"""
        batch_size = embeddings.shape[0]
        
        # Create category mask efficiently using broadcasting
        category_mask = categories.unsqueeze(1) == categories.unsqueeze(0)
        
        # Remove diagonal (self-similarity)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        category_mask = category_mask & ~diag_mask
        
        # Check if we have any positive pairs
        if not category_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute all pairwise similarities at once
        similarities = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # For numerical stability
        similarities = similarities - similarities.max(dim=1, keepdim=True)[0].detach()
        
        # Compute loss for each anchor that has at least one positive
        losses = []
        for i in range(batch_size):
            pos_mask_i = category_mask[i]
            neg_mask_i = ~category_mask[i] & ~diag_mask[i]
            
            # Skip if no positives or no negatives
            if not pos_mask_i.any() or not neg_mask_i.any():
                continue
                
            # Get positive and negative similarities
            pos_sims = similarities[i][pos_mask_i]
            neg_sims = similarities[i][neg_mask_i]
            
            # Take max positive similarity as the positive sample
            pos_sim = pos_sims.max()
            
            # Compute InfoNCE loss
            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return torch.stack(losses).mean()
    
    def _forward_chunked(self, embeddings, categories, device):
        """Memory-efficient chunked computation for large batches"""
        batch_size = embeddings.shape[0]
        losses = []
        
        # Process in chunks to avoid large similarity matrix
        for i in range(0, batch_size, self.chunk_size):
            chunk_end = min(i + self.chunk_size, batch_size)
            chunk_embeddings = embeddings[i:chunk_end]
            chunk_categories = categories[i:chunk_end]
            
            # For each anchor in chunk
            for j, (anchor_emb, anchor_cat) in enumerate(zip(chunk_embeddings, chunk_categories)):
                anchor_idx = i + j
                
                # Find positive indices (same category, not self)
                pos_mask = (categories == anchor_cat) & (torch.arange(batch_size, device=device) != anchor_idx)
                neg_mask = (categories != anchor_cat)
                
                if not pos_mask.any() or not neg_mask.any():
                    continue
                
                # Compute similarities only for this anchor
                anchor_emb_expanded = anchor_emb.unsqueeze(0)
                pos_embeddings = embeddings[pos_mask]
                neg_embeddings = embeddings[neg_mask]
                
                pos_sims = torch.matmul(anchor_emb_expanded, pos_embeddings.T).squeeze() / self.temperature
                neg_sims = torch.matmul(anchor_emb_expanded, neg_embeddings.T).squeeze() / self.temperature
                
                # Ensure sims are always 1D, even with single sample
                if pos_sims.dim() == 0:  # Single positive case
                    pos_sims = pos_sims.unsqueeze(0)
                if neg_sims.dim() == 0:  # Single negative case
                    neg_sims = neg_sims.unsqueeze(0)
                
                # Take max positive similarity
                pos_sim = pos_sims.max()
                
                # For numerical stability
                max_sim = max(pos_sim.item(), neg_sims.max().item())
                pos_sim = pos_sim - max_sim
                neg_sims = neg_sims - max_sim
                
                # Compute InfoNCE loss
                all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                loss = -pos_sim + torch.logsumexp(all_sims, dim=0)
                losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return torch.stack(losses).mean()

class CompiledContrastiveLoss(nn.Module):
    """Wrapper for torch.compile optimization"""
    def __init__(self, temperature=0.7, use_optimized=True):
        super().__init__()
        if use_optimized:
            self.loss_fn = OptimizedCategoryContrastiveLoss(temperature)
        else:
            self.loss_fn = RobustCategoryContrastiveLoss(temperature)
            
    def forward(self, embeddings, categories, font_labels):
        return self.loss_fn(embeddings, categories, font_labels)
