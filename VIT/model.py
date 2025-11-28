import torch
from torch import nn
from VIT import path_emb, TransformerEncoder

class ViT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0,
                 mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()

        assert image_size % patch_size == 0, "patch size is divisible by image size"

        self.num_patches = int(image_size ** 2 / patch_size ** 2)
        
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                            requires_grad= True)
        
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad= True)
        
        self.patch_embedding = path_emb.PatchEmbedding(in_channels= in_channels,
                                              patch_size= patch_size,
                                              embedding_dim= embedding_dim)
        
        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        self.transformerencoder = nn.Sequential(* [TransformerEncoder.TransformerEncoder(embedding_dim= embedding_dim,
                                                     num_heads= num_heads,
                                                     mlp_size= mlp_size,
                                                     attn_dropout= attn_dropout,
                                                     mlp_dropout= mlp_dropout) for _ in range(num_transformer_layers)])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape= embedding_dim),
            nn.Linear(in_features= embedding_dim,
                    out_features= num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim = 1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformerencoder(x)

        x = self.classifier(x[:, 0])

        return x