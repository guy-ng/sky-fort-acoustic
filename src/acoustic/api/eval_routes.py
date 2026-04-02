"""REST API endpoints for model evaluation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from acoustic.api.models import EvalResultResponse, EvalRunRequest
from acoustic.classification.config import MelConfig
from acoustic.evaluation import Evaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["evaluation"])

DEFAULT_TEST_DIR = "Acoustic-UAV-Identification-main-main/Recorded Audios/Real World Testing"


def _validate_data_dir(data_dir: str) -> JSONResponse | None:
    """Return a 404 JSONResponse if data_dir doesn't exist, else None."""
    if not Path(data_dir).is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Directory not found: {data_dir}. Provide a valid data_dir containing labeled subdirectories."
            },
        )
    return None


@router.post("/run", response_model=EvalResultResponse)
async def run_evaluation(body: EvalRunRequest, request: Request) -> EvalResultResponse | JSONResponse:
    """Run model evaluation on labeled test data.

    Supports both single-model and ensemble evaluation. When ensemble_config_path
    is provided, evaluates the ensemble and returns per-model metrics alongside
    ensemble metrics.

    Runs evaluation in a thread executor to avoid blocking the event loop.
    Returns 404 for missing model, data directory, or ensemble config.
    """
    settings = request.app.state.settings
    data_dir = body.data_dir or DEFAULT_TEST_DIR

    evaluator = Evaluator(mel_config=MelConfig())
    loop = asyncio.get_event_loop()

    # Ensemble evaluation path
    if body.ensemble_config_path:
        config_path = Path(body.ensemble_config_path)
        if not config_path.is_file():
            return JSONResponse(
                status_code=404,
                content={
                    "message": f"Ensemble config file not found at {body.ensemble_config_path}. Verify the ensemble_config_path parameter."
                },
            )

        # Validate data directory for ensemble path
        dir_err = _validate_data_dir(data_dir)
        if dir_err:
            return dir_err

        try:
            from acoustic.classification.ensemble import (
                EnsembleClassifier,
                EnsembleConfig,
                load_model,
            )

            config = EnsembleConfig.from_file(body.ensemble_config_path)

            # Validate all model files exist
            for entry in config.models:
                if not Path(entry.path).is_file():
                    return JSONResponse(
                        status_code=404,
                        content={
                            "message": f"Model file not found at {entry.path} (type: {entry.type}). Check ensemble config."
                        },
                    )

            classifiers = [load_model(e.type, e.path) for e in config.models]
            ensemble = EnsembleClassifier(
                classifiers,
                [e.weight for e in config.models],
                live_mode=False,
            )

            result = await loop.run_in_executor(
                None, evaluator.evaluate_ensemble, ensemble, config.models, data_dir
            )
            return EvalResultResponse.from_evaluation(
                result, body.ensemble_config_path, data_dir
            )

        except ValueError as e:
            return JSONResponse(status_code=400, content={"message": str(e)})
        except Exception as e:
            logger.exception("Ensemble evaluation failed")
            return JSONResponse(status_code=500, content={"message": str(e)})

    # Single-model evaluation path (preserve original validation order)
    model_path = body.model_path or settings.cnn_model_path

    # Validate model exists
    if not Path(model_path).is_file():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Model file not found at {model_path}. Verify the model_path parameter or train a model first."
            },
        )

    # Validate data directory exists
    dir_err = _validate_data_dir(data_dir)
    if dir_err:
        return dir_err

    # Run evaluation in executor to avoid blocking event loop
    try:
        result = await loop.run_in_executor(None, evaluator.evaluate, model_path, data_dir)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})

    return EvalResultResponse.from_evaluation(result, model_path, data_dir)
