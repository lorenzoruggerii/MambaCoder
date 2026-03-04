import torch
from mamba_ssm import Mamba
from torch import nn

class MambaModel(nn.Module):

    def __init__(self, num_layers: int, embed_size: int, num_tokens: int):

        super().__init__()
        
        # Store parameters
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.num_tokens = num_tokens

        # Embedding layer
        self.embeddings = nn.Linear(num_tokens, embed_size, bias=False)

        # Mamba blocks
        self.layers = nn.ModuleList([
            Mamba(embed_size)
            for _ in range(num_layers)
        ])

        # Final RMSNorm
        self.norm_f = nn.RMSNorm(embed_size, eps=1e-5)

    def forward(self, x: torch.Tensor):

        # Apply embedding
        x = self.embeddings(x)

        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x) + x # residual connection

        # Apply final RMSNorm
        x = self.norm_f(x)

        return x
    

class MambaForCausalLMTorch(nn.Module):

    def __init__(self, num_layers: int, embed_size: int, num_tokens: int):

        super().__init__()

        # Initialize backbone
        self.backbone = MambaModel(num_layers, embed_size, num_tokens)

        # Use final head to predict tokens
        self.lm_head = nn.Linear(embed_size, num_tokens, bias=False)

    def forward(self, x: torch.Tensor):

        # Apply backbone
        x = self.backbone(x)

        # Get final predictions
        x = self.lm_head(x)

        return x



