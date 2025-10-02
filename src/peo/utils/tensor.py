import torch

def normalize(a, dim=-1, order=2):
    l2 = torch.linalg.norm(a, order, dim)
    return a / l2[:, None]