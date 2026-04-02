"""Evaluation harness: runs inference on labeled WAV folders and computes metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from acoustic.classification.aggregation import WeightedAggregator
from acoustic.classification.config import MelConfig
from acoustic.classification.preprocessing import mel_spectrogram_from_segment
from acoustic.classification.research_cnn import ResearchCNN
from acoustic.evaluation.models import DistributionStats, EvaluationResult, FileResult
from acoustic.training.dataset import collect_wav_files


class Evaluator:
    """Runs inference on labeled test folders and computes classification metrics.

    Uses the same preprocessing and aggregation pipeline as live inference
    to ensure no divergence between evaluation and production.
    """

    def __init__(
        self,
        mel_config: MelConfig | None = None,
        w_max: float = 0.5,
        w_mean: float = 0.5,
    ) -> None:
        self._config = mel_config or MelConfig()
        self._w_max = w_max
        self._w_mean = w_mean
        # Label map handles "no drone" folder name with space
        self._label_map: dict[str, int] = {
            "drone": 1,
            "no drone": 0,
            "background": 0,
            "other": 0,
        }

    def evaluate(self, model_path: str, data_dir: str) -> EvaluationResult:
        """Run evaluation on labeled WAV folders and return metrics.

        Args:
            model_path: Path to a .pt checkpoint (ResearchCNN state_dict).
            data_dir: Root directory containing label subdirectories.

        Returns:
            EvaluationResult with confusion matrix, metrics, distribution stats,
            and per-file detail.
        """
        # Load model
        model = ResearchCNN()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()

        # Collect files using shared training utility
        paths, labels = collect_wav_files(data_dir, self._label_map)

        # Build reverse label map for display (int -> str)
        int_to_label = {1: "drone", 0: "no_drone"}

        # Create aggregator
        aggregator = WeightedAggregator(w_max=self._w_max, w_mean=self._w_mean)

        segment_samples = self._config.segment_samples
        file_results: list[FileResult] = []

        with torch.no_grad():
            for path, label_int in zip(paths, labels):
                audio, sr = sf.read(str(path), dtype="float32")

                # Multi-channel -> mono
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Resample if needed
                if sr != self._config.sample_rate:
                    waveform = torch.from_numpy(audio).float()
                    resampler = torchaudio.transforms.Resample(sr, self._config.sample_rate)
                    audio = resampler(waveform).numpy()

                # Extract non-overlapping segments
                segment_probs: list[float] = []
                n_segments = max(1, len(audio) // segment_samples)

                for i in range(n_segments):
                    start = i * segment_samples
                    end = start + segment_samples

                    if end <= len(audio):
                        segment = audio[start:end]
                    else:
                        # Zero-pad short/final segment
                        segment = np.zeros(segment_samples, dtype=np.float32)
                        remaining = audio[start:]
                        segment[: len(remaining)] = remaining

                    # Preprocessing (same as live inference)
                    features = mel_spectrogram_from_segment(segment, self._config)

                    # Inference
                    prob = model(features).item()
                    segment_probs.append(prob)

                # Per-file statistics
                p_max_val = max(segment_probs) if segment_probs else 0.0
                p_mean_val = (
                    sum(segment_probs) / len(segment_probs) if segment_probs else 0.0
                )
                p_agg = aggregator.aggregate(segment_probs)

                # Prediction
                predicted_int = 1 if p_agg >= 0.5 else 0
                true_label = int_to_label[label_int]
                predicted_label = int_to_label[predicted_int]

                file_results.append(
                    FileResult(
                        filename=path.name,
                        true_label=true_label,
                        predicted_label=predicted_label,
                        p_agg=p_agg,
                        p_max=p_max_val,
                        p_mean=p_mean_val,
                        correct=(predicted_label == true_label),
                    )
                )

        # Compute metrics
        tp, fp, tn, fn = self._compute_confusion(file_results)
        accuracy, precision, recall, f1 = self._compute_metrics(tp, fp, tn, fn)

        # Compute distribution stats
        drone_stats = self._compute_distribution(
            [fr for fr in file_results if fr.true_label == "drone"]
        )
        bg_stats = self._compute_distribution(
            [fr for fr in file_results if fr.true_label == "no_drone"]
        )

        total_correct = sum(1 for fr in file_results if fr.correct)

        return EvaluationResult(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            drone_p_agg=drone_stats[0],
            drone_p_max=drone_stats[1],
            drone_p_mean=drone_stats[2],
            background_p_agg=bg_stats[0],
            background_p_max=bg_stats[1],
            background_p_mean=bg_stats[2],
            files=file_results,
            total_files=len(file_results),
            total_correct=total_correct,
        )

    @staticmethod
    def _compute_confusion(
        file_results: list[FileResult],
    ) -> tuple[int, int, int, int]:
        """Compute confusion matrix from file results."""
        tp = fp = tn = fn = 0
        for fr in file_results:
            if fr.true_label == "drone" and fr.predicted_label == "drone":
                tp += 1
            elif fr.true_label == "no_drone" and fr.predicted_label == "drone":
                fp += 1
            elif fr.true_label == "no_drone" and fr.predicted_label == "no_drone":
                tn += 1
            elif fr.true_label == "drone" and fr.predicted_label == "no_drone":
                fn += 1
        return tp, fp, tn, fn

    @staticmethod
    def _compute_metrics(
        tp: int, fp: int, tn: int, fn: int
    ) -> tuple[float, float, float, float]:
        """Compute accuracy, precision, recall, F1 with division-by-zero guards."""
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return accuracy, precision, recall, f1

    @staticmethod
    def _compute_distribution(
        file_results: list[FileResult],
    ) -> tuple[DistributionStats, DistributionStats, DistributionStats]:
        """Compute percentile distributions for p_agg, p_max, p_mean."""
        if not file_results:
            empty = DistributionStats(0.0, 0.0, 0.0, 0.0)
            return empty, empty, empty

        p_agg_vals = [fr.p_agg for fr in file_results]
        p_max_vals = [fr.p_max for fr in file_results]
        p_mean_vals = [fr.p_mean for fr in file_results]

        percentiles = [25, 50, 75, 95]

        def _stats(values: list[float]) -> DistributionStats:
            arr = np.array(values)
            p = np.percentile(arr, percentiles)
            return DistributionStats(
                p25=float(p[0]),
                p50=float(p[1]),
                p75=float(p[2]),
                p95=float(p[3]),
            )

        return _stats(p_agg_vals), _stats(p_max_vals), _stats(p_mean_vals)
