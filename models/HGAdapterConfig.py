from dataclasses import dataclass

import torch
from adapters import SeqBnConfig


@dataclass(eq=False)
class HGAdapterConfig(SeqBnConfig):
    num_heads: int = 8
    use_hyper: bool = True
    num_edge_types: int = 3
    torch_dtype: str = "float"
    dropout: float = 0.2
