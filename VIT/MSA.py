import torch
from torch import nn

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim : int = 768,
                 num_heads : int = 12,
                 att_dropout : float = 0):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape= embedding_dim)

        self.MultiHeadAttention = nn.MultiheadAttention(embed_dim= embedding_dim,
                                                        num_heads= num_heads,
                                                        dropout= att_dropout,
                                                        batch_first= True)
        
    def forward(self, x):
        x = self.LayerNorm(x)
        attn_output, _ = self.MultiHeadAttention(query= x,
                                                 key= x,
                                                 value= x,
                                                 need_weights = False)
        return attn_output