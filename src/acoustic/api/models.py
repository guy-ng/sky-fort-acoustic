"""Pydantic response models for the acoustic service API."""

from __future__ import annotations

from pydantic import BaseModel

from acoustic.evaluation.models import EvaluationResult


class TargetState(BaseModel):
    """A detected target with current state."""

    id: str  # UUID string
    class_label: str  # "unknown" for Phase 2 placeholder
    speed_mps: float | None  # None until Phase 3 Doppler
    az_deg: float  # Azimuth in degrees
    el_deg: float  # Elevation in degrees
    confidence: float  # 0.0-1.0


class BeamformingMapResponse(BaseModel):
    """Beamforming map as JSON grid with metadata."""

    az_min: float  # -90.0
    az_max: float  # 90.0
    el_min: float  # -45.0
    el_max: float  # 45.0
    az_resolution: float  # 1.0
    el_resolution: float  # 1.0
    width: int  # 181 (azimuth grid points)
    height: int  # 91 (elevation grid points)
    data: list[list[float]]  # 2D grid [elevation][azimuth] -- row-major for canvas
    peak: dict | None  # {"az_deg": float, "el_deg": float, "power": float} if detected


class HeatmapHandshake(BaseModel):
    """Initial WebSocket handshake message with grid dimensions."""

    type: str = "handshake"
    width: int  # 181
    height: int  # 91
    az_min: float
    az_max: float
    el_min: float
    el_max: float


# --- Training models (Phase 9) ---


class TrainingStartRequest(BaseModel):
    """Request to start a training run with optional overrides."""

    model_name: str
    model_type: str | None = None  # "research_cnn" or "efficientat_mn10"
    learning_rate: float | None = None
    batch_size: int | None = None
    max_epochs: int | None = None
    patience: int | None = None
    augmentation_enabled: bool | None = None
    data_root: str | None = None


class TrainingStartResponse(BaseModel):
    """Response after starting a training run."""

    message: str  # e.g. "Training started with 50 max epochs, lr=0.001, batch_size=32"


class ConfusionMatrixResponse(BaseModel):
    """Confusion matrix counts."""

    tp: int
    fp: int
    tn: int
    fn: int


class TrainingProgressResponse(BaseModel):
    """Current training progress snapshot."""

    status: str  # idle/running/completed/cancelled/failed
    epoch: int
    total_epochs: int
    batch: int = 0
    total_batches: int = 0
    train_loss: float
    val_loss: float
    val_acc: float
    best_val_loss: float
    best_epoch: int = 0
    confusion_matrix: ConfusionMatrixResponse
    error: str | None = None
    model_name: str | None = None  # Name set at training start, persists across reloads
    cache_loaded: int = 0    # Audio samples cached in memory
    cache_total: int = 0     # Total audio samples in dataset
    stage: int = 0           # Current training stage (0=N/A, 1-3 for EfficientAT)


class TrainingCancelResponse(BaseModel):
    """Response after cancelling a training run."""

    message: str  # "Training cancelled." or "No training is currently running."


# --- Evaluation models (Phase 9) ---


class EvalRunRequest(BaseModel):
    """Request to run model evaluation on labeled test data."""

    model_path: str | None = None
    data_dir: str | None = None
    ensemble_config_path: str | None = None


class DistributionStatsResponse(BaseModel):
    """Percentile distribution for a probability metric."""

    p25: float
    p50: float
    p75: float
    p95: float


class ClassDistributionResponse(BaseModel):
    """Distribution stats for all probability metrics of a single class."""

    p_agg: DistributionStatsResponse
    p_max: DistributionStatsResponse
    p_mean: DistributionStatsResponse


class FileResultResponse(BaseModel):
    """Per-file evaluation result."""

    filename: str
    true_label: str
    predicted_label: str
    p_agg: float
    correct: bool


class EvalSummaryResponse(BaseModel):
    """Aggregate evaluation metrics summary."""

    total: int
    correct: int
    incorrect: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: ConfusionMatrixResponse


class PerModelResultResponse(BaseModel):
    """Per-model metrics within an ensemble evaluation."""

    model_type: str
    model_path: str
    weight: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class EvalResultResponse(BaseModel):
    """Complete evaluation result response."""

    summary: EvalSummaryResponse
    distribution: dict[str, ClassDistributionResponse]  # {"drone": ..., "background": ...}
    per_file: list[FileResultResponse]
    model_path: str
    data_dir: str
    message: str  # "Evaluated 556 files: 512 correct, 44 incorrect (92.1% accuracy)"
    per_model_results: list[PerModelResultResponse] | None = None

    @staticmethod
    def from_evaluation(
        result: EvaluationResult, model_path: str, data_dir: str
    ) -> EvalResultResponse:
        """Convert domain EvaluationResult to API response model."""
        summary = EvalSummaryResponse(
            total=result.total_files,
            correct=result.total_correct,
            incorrect=result.total_files - result.total_correct,
            accuracy=result.accuracy,
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            confusion_matrix=ConfusionMatrixResponse(
                tp=result.tp, fp=result.fp, tn=result.tn, fn=result.fn
            ),
        )

        def _to_stats(ds) -> DistributionStatsResponse:
            return DistributionStatsResponse(
                p25=ds.p25, p50=ds.p50, p75=ds.p75, p95=ds.p95
            )

        distribution = {
            "drone": ClassDistributionResponse(
                p_agg=_to_stats(result.drone_p_agg),
                p_max=_to_stats(result.drone_p_max),
                p_mean=_to_stats(result.drone_p_mean),
            ),
            "background": ClassDistributionResponse(
                p_agg=_to_stats(result.background_p_agg),
                p_max=_to_stats(result.background_p_max),
                p_mean=_to_stats(result.background_p_mean),
            ),
        }

        per_file = [
            FileResultResponse(
                filename=fr.filename,
                true_label=fr.true_label,
                predicted_label=fr.predicted_label,
                p_agg=fr.p_agg,
                correct=fr.correct,
            )
            for fr in result.files
        ]

        message = (
            f"Evaluated {result.total_files} files: "
            f"{result.total_correct} correct, "
            f"{result.total_files - result.total_correct} incorrect "
            f"({result.accuracy * 100:.1f}% accuracy)"
        )

        # Convert per-model results if present (ensemble evaluation)
        per_model = None
        if result.per_model_results:
            per_model = [
                PerModelResultResponse(
                    model_type=pmr.model_type,
                    model_path=pmr.model_path,
                    weight=pmr.weight,
                    accuracy=pmr.accuracy,
                    precision=pmr.precision,
                    recall=pmr.recall,
                    f1=pmr.f1,
                )
                for pmr in result.per_model_results
            ]

        return EvalResultResponse(
            summary=summary,
            distribution=distribution,
            per_file=per_file,
            model_path=model_path,
            data_dir=data_dir,
            message=message,
            per_model_results=per_model,
        )


# --- Model listing (Phase 9) ---


class ModelInfo(BaseModel):
    """Information about a saved model file."""

    filename: str
    path: str
    size_bytes: int
    modified: str  # ISO 8601


class ModelListResponse(BaseModel):
    """List of available model files."""

    models: list[ModelInfo]


# --- Pipeline activation (Phase 12) ---


class ActivateModelRequest(BaseModel):
    """Request to activate a trained model in the live detection pipeline."""

    model_path: str


class ActivateModelResponse(BaseModel):
    """Response after activating a model."""

    message: str
    model_path: str
    active: bool


# --- Pipeline control ---


class PipelineStartRequest(BaseModel):
    """Request to start the detection pipeline with custom parameters."""

    model_path: str
    confidence: float = 0.90
    time_frame: float = 2.0
    positive_detections: int = 2
    gain: float = 3.0


class PipelineStatusResponse(BaseModel):
    """Current pipeline detection status."""

    running: bool
    model_path: str | None = None
    confidence: float | None = None
    time_frame: float | None = None
    positive_detections: int | None = None
    gain: float | None = None
    detection_state: str | None = None
    drone_probability: float | None = None


class DetectionLogEntry(BaseModel):
    """A single detection log entry."""

    timestamp: float
    drone_probability: float
    detection_state: str
    message: str
