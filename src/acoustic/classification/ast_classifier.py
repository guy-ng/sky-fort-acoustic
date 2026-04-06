"""AST (Audio Spectrogram Transformer) classifier for drone detection.

Wraps a HuggingFace ASTForAudioClassification model to satisfy the
Classifier protocol. Downloads the model on first use and caches locally.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torchaudio

from acoustic.classification.protocols import Classifier

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "preszzz/drone-audio-detection-05-12"
AST_SAMPLE_RATE = 16000


class ASTClassifier:
    """Wraps a HuggingFace AST model to satisfy the Classifier protocol.

    Accepts raw audio waveform tensors and returns drone probability.
    Handles resampling to 16kHz and feature extraction internally.
    """

    def __init__(self, repo_id: str = DEFAULT_HF_REPO) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        logger.info("Loading AST model from %s", repo_id)
        self._model = AutoModelForAudioClassification.from_pretrained(repo_id)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
        self._model.eval()
        self._drone_label_id = None
        for id_, label in self._model.config.id2label.items():
            if "drone" in label.lower() and "not" not in label.lower():
                self._drone_label_id = int(id_)
                break
        if self._drone_label_id is None:
            self._drone_label_id = 1
        logger.info(
            "AST model loaded: %d labels, drone_id=%d, sr=%d",
            self._model.config.num_labels,
            self._drone_label_id,
            self._feature_extractor.sampling_rate,
        )

    def predict(self, features: torch.Tensor) -> float:
        """Run inference on raw audio waveform tensor.

        Args:
            features: Waveform tensor of shape (samples,) already at 16kHz.
                      The RawAudioPreprocessor (target_sr=16000) handles resampling.

        Returns:
            Drone probability as a Python float in [0, 1].
        """
        with torch.no_grad():
            audio = features.numpy() if isinstance(features, torch.Tensor) else features
            if audio.ndim > 1:
                audio = audio.squeeze()

            inputs = self._feature_extractor(
                audio,
                sampling_rate=AST_SAMPLE_RATE,
                return_tensors="pt",
            )
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0, self._drone_label_id].item()


def load_ast_classifier(path_or_repo: str = DEFAULT_HF_REPO) -> Classifier:
    """Load an AST classifier from a HuggingFace repo or local path."""
    return ASTClassifier(repo_id=path_or_repo)
