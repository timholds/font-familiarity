import torch

def compute_font_embedding(model, font_images):
    """
    Compute average latent representation for a font
    """
    with torch.no_grad():
        embeddings = model(font_images)
        return torch.mean(embeddings, dim=0)

def find_similar_fonts(query_embedding, font_embeddings, k=5):
    """
    Find k most similar fonts using cosine similarity
    """
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        font_embeddings
    )
    top_k = torch.topk(similarities, k)
    return top_k.indices


