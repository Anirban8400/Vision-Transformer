import torch
from torch import nn
from VIT import MLP, MSA

class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 attn_dropout: float = 0,
                 mlp_dropout: float = 0.1):
        super().__init__()
        self.MSA_Block = MSA.MultiHeadSelfAttentionBlock(embedding_dim= embedding_dim,
                                               num_heads= num_heads,
                                               att_dropout= attn_dropout)
        self.MLP_Block = MLP.MultiLayerPreceptronBlock(embedding_dim= embedding_dim,
                                             mlp_size= mlp_size,
                                             dropout= mlp_dropout)
        
    def forward(self, x):
        x = self.MSA_Block(x) + x
        x = self.MLP_Block(x) + x
        return x