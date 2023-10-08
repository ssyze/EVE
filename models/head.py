import torch.nn as nn

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, projection_size)
    )