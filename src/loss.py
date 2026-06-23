import torch
import torch.nn.functional as F

def info_nce_loss(a, b, temperature):
    # l2 normalize so dot product equals cosine similarity
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    
    # matrix mult gives cosine sim grid (batch_size x batch_size)
    logits = torch.matmul(a, b.T) / temperature
    
    # target is the diagonal (a1 should match b1, a2 matches b2)
    labels = torch.arange(a.shape[0], device=a.device)
    
    # symmetric loss (a->b and b->a)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
