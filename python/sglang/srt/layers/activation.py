import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     d = x.shape[-1] // 2
    #     output_shape = x.shape[:-1] + (d,)
    #     out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    #     ops.silu_and_mul(out, x)
    #     return out
