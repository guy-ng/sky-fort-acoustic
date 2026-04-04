"""Tests for ResearchCNN model and ResearchClassifier wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from acoustic.classification.research_cnn import ResearchCNN, ResearchClassifier
from acoustic.classification.protocols import Classifier


class TestResearchCNNForward:
    """Verify ResearchCNN forward pass shapes and output range."""

    def test_forward_shape(self):
        model = ResearchCNN()
        model.eval()
        out = model(torch.zeros(1, 1, 128, 64))
        assert out.shape == (1, 1)

    def test_batch_forward_shape(self):
        model = ResearchCNN()
        model.eval()
        out = model(torch.zeros(4, 1, 128, 64))
        assert out.shape == (4, 1)

    def test_output_range(self):
        model = ResearchCNN()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(8, 1, 128, 64))
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_eval_mode_deterministic(self):
        model = ResearchCNN()
        model.eval()
        x = torch.randn(2, 1, 128, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)


class TestResearchCNNArchitecture:
    """Verify architecture matches the research spec exactly."""

    def test_architecture_matches_spec(self):
        model = ResearchCNN()

        # Check features sequential
        conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
        assert len(conv_layers) == 3
        assert conv_layers[0].out_channels == 32
        assert conv_layers[1].out_channels == 64
        assert conv_layers[2].out_channels == 128

        bn_layers = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 3

        pool_layers = [m for m in model.features if isinstance(m, nn.MaxPool2d)]
        assert len(pool_layers) == 3

        # Check classifier sequential
        avg_pool = [m for m in model.classifier if isinstance(m, nn.AdaptiveAvgPool2d)]
        assert len(avg_pool) == 1

        linear_layers = [m for m in model.classifier if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2
        assert linear_layers[0].in_features == 128
        assert linear_layers[0].out_features == 128
        assert linear_layers[1].in_features == 128
        assert linear_layers[1].out_features == 1

        dropout_layers = [m for m in model.classifier if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 1
        assert dropout_layers[0].p == 0.3

        # Sigmoid moved to separate attribute for logits_mode support
        assert isinstance(model._sigmoid, nn.Sigmoid)


class TestResearchClassifier:
    """Verify ResearchClassifier wrapper satisfies Classifier protocol."""

    def test_satisfies_classifier_protocol(self):
        model = ResearchCNN()
        classifier = ResearchClassifier(model)
        assert isinstance(classifier, Classifier)

    def test_predict_returns_float(self):
        model = ResearchCNN()
        classifier = ResearchClassifier(model)
        result = classifier.predict(torch.zeros(1, 1, 128, 64))
        assert isinstance(result, float)

    def test_predict_range(self):
        model = ResearchCNN()
        classifier = ResearchClassifier(model)
        result = classifier.predict(torch.randn(1, 1, 128, 64))
        assert 0.0 <= result <= 1.0
