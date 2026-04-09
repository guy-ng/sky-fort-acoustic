"""EfficientATClassifier: wraps EfficientAT MN model to satisfy Classifier protocol.

The Classifier protocol requires: predict(features: torch.Tensor) -> float.
This wrapper handles mel preprocessing internally so the caller can pass raw waveforms.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import EfficientATMelConfig
from .preprocess import AugmentMelSTFT
from .window_contract import EFFICIENTAT_SEGMENT_SAMPLES, EFFICIENTAT_TARGET_SR

_logger = logging.getLogger(__name__)


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
        # Phase 22 Wave 2 (REQ-22-W3): one-shot flag for train/serve window
        # length mismatch. First bad input logs a WARN so the drift is visible
        # in operational logs; subsequent bad inputs on the same instance are
        # suppressed to avoid flooding at pipeline tick rate (~0.2 s).
        self._warned_mismatch = False

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

            # Phase 22 Wave 2 (REQ-22-W3): fail-soft length contract check.
            # The model is shape-agnostic (attention pooling + SE blocks on
            # channel dim only) so a wrong-length input will NOT crash — it
            # will silently regress. This WARN is the operator's only signal
            # that the v7 regression signature is reproducing in production.
            # We intentionally do NOT raise: killing a live detection pipeline
            # over a soft-drift would be worse than running out of domain.
            actual = int(x.shape[-1])
            if (
                actual != EFFICIENTAT_SEGMENT_SAMPLES
                and not self._warned_mismatch
            ):
                _logger.warning(
                    "EfficientAT input length %d != expected %d "
                    "(%.3fs vs %.3fs @ %d Hz). Model will run but is "
                    "out-of-domain — this is the v7 regression signature. "
                    "Check DetectionSession.window_seconds and "
                    "pipeline._cnn_segment_samples.",
                    actual,
                    EFFICIENTAT_SEGMENT_SAMPLES,
                    actual / float(EFFICIENTAT_TARGET_SR),
                    EFFICIENTAT_SEGMENT_SAMPLES / float(EFFICIENTAT_TARGET_SR),
                    EFFICIENTAT_TARGET_SR,
                )
                self._warned_mismatch = True

            mel = self._mel(x)
            # AugmentMelSTFT outputs (batch, n_mels, time); model needs (batch, 1, n_mels, time)
            mel = mel.unsqueeze(1)
            logits, _ = self._model(mel)
            return torch.sigmoid(logits).item()
