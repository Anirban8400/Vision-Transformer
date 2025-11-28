import torch
from torch import nn

class MultiLayerPreceptronBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape= embedding_dim)

        self.MLP = nn.Sequential(
            nn.Linear(in_features= embedding_dim,
                      out_features= mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features= mlp_size,
                      out_features= embedding_dim),
            nn.Dropout(p= dropout)
        )

    def forward(self, x):
        x = self.LayerNorm(x)
        x = self.MLP(x)
        return x