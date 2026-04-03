"""Utility functions vendored from EfficientAT models/mn/utils.py.

Source: https://github.com/fschmid56/EfficientAT
License: Apache-2.0
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure channel count is divisible by 8 (from TF MobileNet repo)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    """Compute output size of a CNN layer."""
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(
    x: Tensor,
    dim: int,
    mode: str = "pool",
    pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
    combine_dim: int | None = None,
) -> Tensor:
    """Collapse a dimension by pooling or combining."""
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)
    raise ValueError(f"Unknown mode: {mode}")


class CollapseDim(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str = "pool",
        pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
        combine_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.pool_fn = pool_fn
        self.combine_dim = combine_dim

    def forward(self, x: Tensor) -> Tensor:
        return collapse_dim(x, dim=self.dim, mode=self.mode, pool_fn=self.pool_fn, combine_dim=self.combine_dim)


# Width multiplier map for model name lookup
NAME_TO_WIDTH: dict[str, float] = {
    "mn01": 0.1, "mn02": 0.2, "mn04": 0.4, "mn05": 0.5,
    "mn06": 0.6, "mn08": 0.8, "mn10": 1.0, "mn12": 1.2,
    "mn14": 1.4, "mn16": 1.6, "mn20": 2.0, "mn30": 3.0, "mn40": 4.0,
}
