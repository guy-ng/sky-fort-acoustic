"""Unit tests for the evaluation harness (Evaluator class)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from acoustic.classification.research_cnn import ResearchCNN
from acoustic.evaluation.evaluator import Evaluator
from acoustic.evaluation.models import DistributionStats, EvaluationResult, FileResult


def _create_wav(path, samples: int = 16000, sr: int = 16000) -> None:
    """Write a synthetic WAV file with random noise."""
    audio = np.random.randn(samples).astype(np.float32) * 0.01
    sf.write(str(path), audio, sr)


def _create_test_data(tmp_path, drone_count: int = 3, bg_count: int = 3, samples: int = 16000):
    """Create a labeled directory structure with drone/ and 'no drone/' folders."""
    drone_dir = tmp_path / "drone"
    drone_dir.mkdir()
    for i in range(drone_count):
        _create_wav(drone_dir / f"drone_{i}.wav", samples=samples)

    # "no drone" with space in name
    bg_dir = tmp_path / "no drone"
    bg_dir.mkdir()
    for i in range(bg_count):
        _create_wav(bg_dir / f"bg_{i}.wav", samples=samples)

    # Save a model checkpoint
    model = ResearchCNN()
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)

    return str(model_path)


class TestEvaluatorConfusionMatrix:
    """Test 1: Evaluator.evaluate() produces correct confusion matrix."""

    def test_confusion_matrix_has_correct_totals(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=4, bg_count=3)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        assert isinstance(result, EvaluationResult)
        # Total should match file count
        assert result.tp + result.fp + result.tn + result.fn == 7
        assert result.total_files == 7

    def test_confusion_matrix_counts_are_nonnegative(self, tmp_path):
        model_path = _create_test_data(tmp_path)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        assert result.tp >= 0
        assert result.fp >= 0
        assert result.tn >= 0
        assert result.fn >= 0


class TestFileResults:
    """Test 2: FileResult objects contain correct fields."""

    def test_file_results_contain_expected_fields(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=2, bg_count=2)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        assert len(result.files) == 4
        for fr in result.files:
            assert isinstance(fr, FileResult)
            assert isinstance(fr.filename, str)
            assert fr.true_label in ("drone", "no_drone")
            assert fr.predicted_label in ("drone", "no_drone")
            assert 0.0 <= fr.p_agg <= 1.0
            assert isinstance(fr.correct, bool)

    def test_file_results_correct_flag_matches_labels(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=2, bg_count=2)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        for fr in result.files:
            expected = fr.true_label == fr.predicted_label
            assert fr.correct == expected


class TestDistributionStats:
    """Test 3: Distribution stats contain percentiles per class."""

    def test_distribution_stats_have_percentile_fields(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=5, bg_count=5)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        for stats in [
            result.drone_p_agg, result.drone_p_max, result.drone_p_mean,
            result.background_p_agg, result.background_p_max, result.background_p_mean,
        ]:
            assert isinstance(stats, DistributionStats)
            assert isinstance(stats.p25, float)
            assert isinstance(stats.p50, float)
            assert isinstance(stats.p75, float)
            assert isinstance(stats.p95, float)

    def test_percentiles_are_ordered(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=10, bg_count=10)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        for stats in [result.drone_p_agg, result.background_p_agg]:
            assert stats.p25 <= stats.p50 <= stats.p75 <= stats.p95


class TestMetricsComputation:
    """Test 4: Metrics are correctly computed from confusion matrix."""

    def test_accuracy_formula(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=5, bg_count=5)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        expected_acc = (result.tp + result.tn) / max(result.tp + result.tn + result.fp + result.fn, 1)
        assert abs(result.accuracy - expected_acc) < 1e-6

    def test_precision_recall_f1_formulas(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=5, bg_count=5)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        expected_precision = result.tp / max(result.tp + result.fp, 1)
        expected_recall = result.tp / max(result.tp + result.fn, 1)
        expected_f1 = 2 * expected_precision * expected_recall / max(expected_precision + expected_recall, 1e-8)

        assert abs(result.precision - expected_precision) < 1e-6
        assert abs(result.recall - expected_recall) < 1e-6
        assert abs(result.f1 - expected_f1) < 1e-6

    def test_total_correct_matches_files(self, tmp_path):
        model_path = _create_test_data(tmp_path, drone_count=3, bg_count=3)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        correct_count = sum(1 for fr in result.files if fr.correct)
        assert result.total_correct == correct_count


class TestDivisionByZeroEdge:
    """Test 5: All-same-class predictions return 0.0 for undefined metrics."""

    def test_all_drone_predictions(self, tmp_path):
        """When only drone files exist, tn=fn=0, metrics don't crash."""
        drone_dir = tmp_path / "drone"
        drone_dir.mkdir()
        for i in range(5):
            _create_wav(drone_dir / f"d_{i}.wav")

        model = ResearchCNN()
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        evaluator = Evaluator()
        result = evaluator.evaluate(str(model_path), str(tmp_path))

        # Should not raise, no NaN or Inf
        assert not np.isnan(result.accuracy)
        assert not np.isnan(result.precision)
        assert not np.isnan(result.recall)
        assert not np.isnan(result.f1)
        assert not np.isinf(result.f1)


class TestLabelMapSpaceHandling:
    """Test 6: 'no drone' folder name with space is handled."""

    def test_no_drone_folder_with_space(self, tmp_path):
        bg_dir = tmp_path / "no drone"
        bg_dir.mkdir()
        for i in range(3):
            _create_wav(bg_dir / f"bg_{i}.wav")

        model = ResearchCNN()
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        evaluator = Evaluator()
        result = evaluator.evaluate(str(model_path), str(tmp_path))

        assert result.total_files == 3
        for fr in result.files:
            assert fr.true_label == "no_drone"


class TestShortAudioFiles:
    """Test 7: Short audio files are zero-padded and produce valid results."""

    def test_short_files_produce_valid_results(self, tmp_path):
        # Files shorter than 8000 samples (0.5s at 16kHz)
        model_path = _create_test_data(tmp_path, drone_count=2, bg_count=2, samples=2000)
        evaluator = Evaluator()
        result = evaluator.evaluate(model_path, str(tmp_path))

        assert result.total_files == 4
        for fr in result.files:
            assert 0.0 <= fr.p_agg <= 1.0
