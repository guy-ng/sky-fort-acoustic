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
from acoustic.classification.protocols import Classifier
from acoustic.evaluation.models import (
    DistributionStats,
    EvaluationResult,
    FileResult,
    PerModelResult,
)
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

        Backward-compatible entry point that loads a ResearchCNN from a
        checkpoint path and delegates to evaluate_classifier.

        Args:
            model_path: Path to a .pt checkpoint (ResearchCNN state_dict).
            data_dir: Root directory containing label subdirectories.

        Returns:
            EvaluationResult with confusion matrix, metrics, distribution stats,
            and per-file detail.
        """
        from acoustic.classification.ensemble import load_model

        classifier = load_model("research_cnn", model_path)
        return self.evaluate_classifier(classifier, data_dir)

    def evaluate_classifier(
        self, classifier: Classifier, data_dir: str
    ) -> EvaluationResult:
        """Run evaluation using any Classifier protocol implementor.

        Args:
            classifier: Any object satisfying the Classifier protocol.
            data_dir: Root directory containing label subdirectories.

        Returns:
            EvaluationResult with confusion matrix, metrics, distribution stats,
            and per-file detail.
        """
        paths, labels = collect_wav_files(data_dir, self._label_map)
        int_to_label = {1: "drone", 0: "no_drone"}
        aggregator = WeightedAggregator(w_max=self._w_max, w_mean=self._w_mean)
        segment_samples = self._config.segment_samples
        file_results: list[FileResult] = []

        with torch.no_grad():
            for path, label_int in zip(paths, labels):
                segment_probs = self._process_file(path, segment_samples, classifier)

                p_max_val = max(segment_probs) if segment_probs else 0.0
                p_mean_val = (
                    sum(segment_probs) / len(segment_probs) if segment_probs else 0.0
                )
                p_agg = aggregator.aggregate(segment_probs)

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

        return self._build_result(file_results)

    def evaluate_ensemble(
        self,
        ensemble: object,
        model_entries: list,
        data_dir: str,
    ) -> EvaluationResult:
        """Evaluate an ensemble and each individual model in a single pass.

        Returns ensemble metrics as the main EvaluationResult, plus per-model
        metrics populated in per_model_results.

        Args:
            ensemble: An EnsembleClassifier instance.
            model_entries: List of ModelEntry objects from the ensemble config.
            data_dir: Root directory containing label subdirectories.

        Returns:
            EvaluationResult with ensemble metrics and per_model_results populated.
        """
        from acoustic.classification.ensemble import EnsembleClassifier, ModelEntry, load_model

        paths, labels = collect_wav_files(data_dir, self._label_map)
        int_to_label = {1: "drone", 0: "no_drone"}
        aggregator = WeightedAggregator(w_max=self._w_max, w_mean=self._w_mean)
        segment_samples = self._config.segment_samples

        # Load individual models for per-model evaluation
        individual_classifiers: list[Classifier] = []
        for entry in model_entries:
            individual_classifiers.append(load_model(entry.type, entry.path))

        # Track per-model file results
        per_model_file_results: list[list[FileResult]] = [
            [] for _ in model_entries
        ]
        ensemble_file_results: list[FileResult] = []

        with torch.no_grad():
            for path, label_int in zip(paths, labels):
                # Preprocess segments once
                features_list = self._extract_features(path, segment_samples)
                true_label = int_to_label[label_int]

                # Run ensemble prediction
                ensemble_probs = [
                    ensemble.predict(features) for features in features_list
                ]
                ens_p_max = max(ensemble_probs) if ensemble_probs else 0.0
                ens_p_mean = (
                    sum(ensemble_probs) / len(ensemble_probs)
                    if ensemble_probs
                    else 0.0
                )
                ens_p_agg = aggregator.aggregate(ensemble_probs)
                ens_predicted = int_to_label[1 if ens_p_agg >= 0.5 else 0]

                ensemble_file_results.append(
                    FileResult(
                        filename=path.name,
                        true_label=true_label,
                        predicted_label=ens_predicted,
                        p_agg=ens_p_agg,
                        p_max=ens_p_max,
                        p_mean=ens_p_mean,
                        correct=(ens_predicted == true_label),
                    )
                )

                # Run each individual model
                for idx, clf in enumerate(individual_classifiers):
                    model_probs = [clf.predict(features) for features in features_list]
                    m_p_max = max(model_probs) if model_probs else 0.0
                    m_p_mean = (
                        sum(model_probs) / len(model_probs)
                        if model_probs
                        else 0.0
                    )
                    m_p_agg = aggregator.aggregate(model_probs)
                    m_predicted = int_to_label[1 if m_p_agg >= 0.5 else 0]

                    per_model_file_results[idx].append(
                        FileResult(
                            filename=path.name,
                            true_label=true_label,
                            predicted_label=m_predicted,
                            p_agg=m_p_agg,
                            p_max=m_p_max,
                            p_mean=m_p_mean,
                            correct=(m_predicted == true_label),
                        )
                    )

        # Build ensemble result
        result = self._build_result(ensemble_file_results)

        # Compute per-model metrics
        for idx, entry in enumerate(model_entries):
            fr_list = per_model_file_results[idx]
            tp, fp, tn, fn = self._compute_confusion(fr_list)
            accuracy, precision, recall, f1 = self._compute_metrics(tp, fp, tn, fn)
            result.per_model_results.append(
                PerModelResult(
                    model_type=entry.type,
                    model_path=entry.path,
                    weight=entry.weight,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )

        return result

    def _extract_features(
        self, path: Path, segment_samples: int
    ) -> list[torch.Tensor]:
        """Extract feature tensors for all segments in a WAV file."""
        audio, sr = sf.read(str(path), dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self._config.sample_rate:
            waveform = torch.from_numpy(audio).float()
            resampler = torchaudio.transforms.Resample(sr, self._config.sample_rate)
            audio = resampler(waveform).numpy()

        features_list: list[torch.Tensor] = []
        n_segments = max(1, len(audio) // segment_samples)

        for i in range(n_segments):
            start = i * segment_samples
            end = start + segment_samples

            if end <= len(audio):
                segment = audio[start:end]
            else:
                segment = np.zeros(segment_samples, dtype=np.float32)
                remaining = audio[start:]
                segment[: len(remaining)] = remaining

            features = mel_spectrogram_from_segment(segment, self._config)
            features_list.append(features)

        return features_list

    def _process_file(
        self, path: Path, segment_samples: int, classifier: Classifier
    ) -> list[float]:
        """Process a single WAV file through a classifier and return segment probs."""
        features_list = self._extract_features(path, segment_samples)
        return [classifier.predict(features) for features in features_list]

    def _build_result(self, file_results: list[FileResult]) -> EvaluationResult:
        """Build an EvaluationResult from file results."""
        tp, fp, tn, fn = self._compute_confusion(file_results)
        accuracy, precision, recall, f1 = self._compute_metrics(tp, fp, tn, fn)

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
