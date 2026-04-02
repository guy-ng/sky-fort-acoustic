"""REST API endpoints for CNN training lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from acoustic.api.models import (
    ConfusionMatrixResponse,
    TrainingCancelResponse,
    TrainingProgressResponse,
    TrainingStartRequest,
    TrainingStartResponse,
)
from acoustic.training.config import TrainingConfig
from acoustic.training.manager import TrainingManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(body: TrainingStartRequest, request: Request) -> TrainingStartResponse | JSONResponse:
    """Start a training run with optional hyperparameter overrides.

    Returns 409 if training is already in progress (per UI-SPEC D-06).
    """
    manager: TrainingManager = request.app.state.training_manager

    if manager.is_training():
        return JSONResponse(
            status_code=409,
            content={
                "message": "Training is already in progress. Cancel the current run before starting a new one."
            },
        )

    # Build config with overrides from request body
    config = TrainingConfig()
    overrides = {k: v for k, v in body.model_dump().items() if v is not None}
    if overrides:
        config = config.model_copy(update=overrides)

    # Validate data_root if provided
    if body.data_root is not None and not Path(body.data_root).is_dir():
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Training data directory not found: {body.data_root}. Provide labeled audio subdirectories."
            },
        )

    manager.start(config)
    return TrainingStartResponse(
        message=f"Training started with {config.max_epochs} max epochs, lr={config.learning_rate}, batch_size={config.batch_size}"
    )


@router.get("/progress", response_model=TrainingProgressResponse)
async def get_progress(request: Request) -> TrainingProgressResponse:
    """Return current training progress snapshot."""
    manager: TrainingManager = request.app.state.training_manager
    progress = manager.get_progress()

    return TrainingProgressResponse(
        status=progress.status.value,
        epoch=progress.epoch,
        total_epochs=progress.total_epochs,
        train_loss=progress.train_loss,
        val_loss=progress.val_loss,
        val_acc=progress.val_acc,
        best_val_loss=progress.best_val_loss,
        best_epoch=0,
        confusion_matrix=ConfusionMatrixResponse(
            tp=progress.tp, fp=progress.fp, tn=progress.tn, fn=progress.fn
        ),
        error=progress.error,
    )


@router.post("/cancel", response_model=TrainingCancelResponse)
async def cancel_training(request: Request) -> TrainingCancelResponse | JSONResponse:
    """Cancel a running training run.

    Returns 409 if no training is currently running.
    """
    manager: TrainingManager = request.app.state.training_manager

    if not manager.is_training():
        return JSONResponse(
            status_code=409,
            content={"message": "No training is currently running."},
        )

    manager.cancel()
    return TrainingCancelResponse(
        message="Training cancelled. Partial results may be available."
    )
