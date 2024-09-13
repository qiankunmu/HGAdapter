from dataclasses import dataclass

import torch
from adapters import SeqBnConfig


@dataclass(eq=False)
class HyperStructAdapterConfig(SeqBnConfig):
    num_heads: int = 8
    use_hyper: bool = True
    num_edge_types: int = 3
    use_norm: bool = False
    torch_dtype: str = "float"
    dropout: float = 0
    is_newHGbd: bool = False
    use_adapter2: bool = False
