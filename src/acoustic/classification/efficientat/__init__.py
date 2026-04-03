"""EfficientAT MobileNetV3 mn10 model for drone detection.

Importing this package registers the 'efficientat_mn10' model type
in the ensemble model registry.
"""

from __future__ import annotations

import torch

from acoustic.classification.ensemble import register_model
from acoustic.classification.protocols import Classifier


def _load_efficientat_mn10(path: str) -> Classifier:
    """Load a fine-tuned EfficientAT mn10 checkpoint.

    Args:
        path: Path to the saved state_dict (.pt file).

    Returns:
        EfficientATClassifier instance satisfying the Classifier protocol.
    """
    from acoustic.classification.efficientat.classifier import EfficientATClassifier
    from acoustic.classification.efficientat.model import get_model

    model = get_model(
        num_classes=1,
        width_mult=1.0,
        head_type="mlp",
        input_dim_f=128,
        input_dim_t=100,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return EfficientATClassifier(model)


register_model("efficientat_mn10", _load_efficientat_mn10)
