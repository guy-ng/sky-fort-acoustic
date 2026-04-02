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


@router.post("/run", response_model=EvalResultResponse)
async def run_evaluation(body: EvalRunRequest, request: Request) -> EvalResultResponse | JSONResponse:
    """Run model evaluation on labeled test data.

    Runs evaluation in a thread executor to avoid blocking the event loop.
    Returns 404 for missing model or data directory (per UI-SPEC D-07, D-08).
    """
    settings = request.app.state.settings

    model_path = body.model_path or settings.cnn_model_path
    data_dir = body.data_dir or DEFAULT_TEST_DIR

    # Validate model exists
    if not Path(model_path).is_file():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Model file not found at {model_path}. Verify the model_path parameter or train a model first."
            },
        )

    # Validate data directory exists
    if not Path(data_dir).is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Directory not found: {data_dir}. Provide a valid data_dir containing labeled subdirectories."
            },
        )

    # Run evaluation in executor to avoid blocking event loop
    evaluator = Evaluator(mel_config=MelConfig())
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, evaluator.evaluate, model_path, data_dir)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})

    return EvalResultResponse.from_evaluation(result, model_path, data_dir)
