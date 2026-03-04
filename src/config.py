from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn

class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)
    
@dataclass
class TranscoderConfig:
    """
    Config class for Transcoder
    """
    weight_path: str
    save_path: str = "models/MambaCoders/mambacoder_130m_pile_250k.pt"
    num_features: int = 12_288 # 16x of hidden_size
    device: str = 'cuda' if torch.cuda.is_available() else "cpu"
    topk_features: int = 32
    activation_fn: Callable = TopK(k=topk_features)
    lr: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 50
    max_length: int = 512 # Maximum prompt length
    batch_size: int = 4
    num_epochs: int = 2
    num_train_prompts: int = 250_000 # then increase
    tokenizer_path: str = "state-spaces/mamba-130m-hf"

