"""Loss functions for drone audio classification training (TRN-10).

FocalLoss down-weights easy examples so the model focuses on hard
misclassifications. Wraps torchvision.ops.sigmoid_focal_loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Focal loss for binary classification.

    Wraps ``torchvision.ops.sigmoid_focal_loss`` with configurable
    *alpha* (class balance) and *gamma* (focus on hard examples).

    IMPORTANT: ``inputs`` must be raw logits (pre-sigmoid).  The focal
    loss function applies sigmoid internally.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits (pre-sigmoid), any shape.
            targets: Binary targets of the same shape as *inputs*.

        Returns:
            Scalar mean focal loss.
        """
        return sigmoid_focal_loss(
            inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction="mean"
        )


def build_loss_function(
    loss_type: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    bce_pos_weight: float = 1.0,
) -> nn.Module:
    """Factory: select loss function by config string.

    Args:
        loss_type: One of ``"focal"``, ``"bce"``, ``"bce_weighted"``.
        focal_alpha: Alpha for :class:`FocalLoss`.
        focal_gamma: Gamma for :class:`FocalLoss`.
        bce_pos_weight: Positive-class weight for weighted BCE.

    Returns:
        Configured loss module.

    Raises:
        ValueError: If *loss_type* is not recognized.
    """
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "bce_weighted":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]))
    raise ValueError(
        f"unknown loss_type {loss_type!r}; expected 'focal', 'bce', or 'bce_weighted'"
    )
