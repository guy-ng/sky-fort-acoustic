"""EfficientATClassifier: wraps EfficientAT MN model to satisfy Classifier protocol.

The Classifier protocol requires: predict(features: torch.Tensor) -> float.
This wrapper handles mel preprocessing internally so the caller can pass raw waveforms.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EfficientATMelConfig
from .preprocess import AugmentMelSTFT


class EfficientATClassifier:
    """Wraps EfficientAT MN model to satisfy Classifier protocol.

    Accepts raw audio waveform tensors and returns a sigmoid probability.
    Mel-spectrogram preprocessing is handled internally using AugmentMelSTFT.
    """

    def __init__(
        self,
        model: nn.Module,
        mel_config: EfficientATMelConfig | None = None,
    ) -> None:
        self._model = model
        self._model.eval()
        cfg = mel_config or EfficientATMelConfig()
        self._mel = AugmentMelSTFT(
            n_mels=cfg.n_mels,
            sr=cfg.sample_rate,
            win_length=cfg.win_length,
            hopsize=cfg.hop_size,
            n_fft=cfg.n_fft,
        )
        self._mel.eval()
        self._target_sr = cfg.sample_rate

    def predict(self, features: torch.Tensor) -> float:
        """Run inference on raw audio waveform tensor.

        Args:
            features: Waveform tensor of shape (1, samples) or (samples,).

        Returns:
            Sigmoid probability as a Python float in [0, 1].
        """
        with torch.no_grad():
            x = features
            if x.dim() == 1:
                x = x.unsqueeze(0)
            mel = self._mel(x)
            # AugmentMelSTFT outputs (batch, n_mels, time); model needs (batch, 1, n_mels, time)
            mel = mel.unsqueeze(1)
            logits, _ = self._model(mel)
            return torch.sigmoid(logits).item()
