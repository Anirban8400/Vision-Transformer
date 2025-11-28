import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """ Converting 2D image into 1D sequence of embedding vector.
    
    Args:
        in_channels : size of input images channel.
        path_size : size of desired patch.
        embedding_dim : required dimensions of embedding
        """
    
    def __init__(self,
                 in_channels : int,
                 patch_size : int,
                 embedding_dim : int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.patcher = nn.Conv2d(in_channels= in_channels,
                                 out_channels= embedding_dim,
                                 stride= patch_size,
                                 kernel_size= patch_size,
                                 padding= 0)
        self.flatten = nn.Flatten(start_dim= 2,
                                  end_dim= 3)
    
    def forward(self, x):
        image_res = x.shape[-1]
        assert (image_res % self.patch_size == 0), "patch size should be divisible with image resolution"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)