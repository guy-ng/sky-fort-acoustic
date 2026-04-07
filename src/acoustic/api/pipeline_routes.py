"""REST API endpoints for pipeline control (model activation, start/stop detection)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from acoustic.api.models import (
    ActivateModelRequest,
    ActivateModelResponse,
    PipelineStartRequest,
    PipelineStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/activate", response_model=ActivateModelResponse)
async def activate_model(body: ActivateModelRequest, request: Request) -> ActivateModelResponse | JSONResponse:
    """Load a trained model and hot-swap it into the live CNN pipeline.

    This allows the user to train a model, evaluate it, and then activate it
    for real-time detection without restarting the service.

    Returns 404 if model file not found, 400 if CNN worker not available.
    """
    model_path = body.model_path

    # Validate model file exists
    if not Path(model_path).is_file():
        return JSONResponse(
            status_code=404,
            content={"message": f"Model file not found at {model_path}"},
        )

    # Check CNN worker is available
    pipeline = request.app.state.pipeline
    cnn_worker = getattr(pipeline, '_cnn_worker', None)
    if cnn_worker is None:
        return JSONResponse(
            status_code=400,
            content={"message": "CNN worker not initialized. Cannot activate model."},
        )

    try:
        import torch

        from acoustic.classification.research_cnn import ResearchCNN, ResearchClassifier

        model = ResearchCNN()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Validate model with dummy forward pass
        dummy = torch.zeros(1, 1, 128, 64)
        with torch.no_grad():
            out = model(dummy)
        if out.shape != (1, 1):
            return JSONResponse(
                status_code=400,
                content={"message": f"Unexpected model output shape: {out.shape}. Expected (1, 1)."},
            )

        classifier = ResearchClassifier(model)
        cnn_worker.set_classifier(classifier)

        logger.info("Model activated in pipeline: %s", model_path)
        return ActivateModelResponse(
            message=f"Model activated: {Path(model_path).name}",
            model_path=model_path,
            active=True,
        )

    except Exception as e:
        logger.exception("Failed to activate model: %s", model_path)
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to load model: {e}"},
        )


@router.post("/start")
async def start_detection(body: PipelineStartRequest, request: Request):
    """Start a detection session with custom parameters.

    Loads the specified model and configures the detection pipeline with
    the given confidence threshold, time frame, positive detection count, and gain.
    """
    pipeline = request.app.state.pipeline

    # Check if already running a detection session
    if pipeline.detection_session is not None:
        return JSONResponse(
            status_code=409,
            content={"message": "Detection session already running. Stop it first."},
        )

    # Device-specific gain cap. Only fallback devices that publish a
    # `recommended_gain` (e.g. ReSpeaker raw mic 1) get clamped — the UMA-16v2
    # path is untouched and the user-provided gain still applies as-is.
    device_info = getattr(request.app.state, "device_info", None)
    if device_info is not None and device_info.recommended_gain is not None:
        if body.gain > device_info.recommended_gain:
            logger.warning(
                "Clamping requested gain %.1f → %.1f for fallback device '%s' "
                "(UMA-tuned gain clips on this mic)",
                body.gain,
                device_info.recommended_gain,
                device_info.name,
            )
            body.gain = device_info.recommended_gain

    # Validate model file exists
    if not Path(body.model_path).is_file():
        return JSONResponse(
            status_code=404,
            content={"message": f"Model file not found: {body.model_path}"},
        )

    # Load and activate the model
    cnn_worker = getattr(pipeline, '_cnn_worker', None)
    if cnn_worker is None:
        return JSONResponse(
            status_code=400,
            content={"message": "CNN worker not initialized."},
        )

    try:
        # Use ensemble loader which handles multiple model types
        import acoustic.classification.efficientat  # noqa: F401
        from acoustic.classification.ensemble import load_model

        model_path = body.model_path
        # Detect model type from filename
        if "efficientat" in os.path.basename(model_path).lower() or "mn10" in os.path.basename(model_path).lower():
            model_type = "efficientat_mn10"
        else:
            model_type = "research_cnn"

        classifier = load_model(model_type, model_path)
        cnn_worker.set_classifier(classifier)

        # Preprocessor selection — DO NOT recreate the preprocessor on every
        # session start. The lifespan-startup picks the right type based on
        # cnn_model_type and we only need to (a) hot-update the calibration
        # gain on the raw-audio path, or (b) swap the type if the user picked
        # a different model family than the one configured at startup.
        from acoustic.classification.preprocessing import (
            RawAudioPreprocessor,
            ResearchPreprocessor,
        )
        existing_pp = getattr(cnn_worker, "_preprocessor", None)
        if model_type == "efficientat_mn10":
            if isinstance(existing_pp, RawAudioPreprocessor):
                # Reuse — preserves debug-dump sequence and any cached resampler.
                existing_pp.set_input_gain(body.gain)
            else:
                # D-34: keep the RMS normalization target in sync with the
                # service default so swapping the model mid-session still
                # lands CNN inputs in the trainer-matched regime.
                from acoustic.config import AcousticSettings
                _rms_target = AcousticSettings().cnn_rms_normalize_target
                cnn_worker.set_preprocessor(
                    RawAudioPreprocessor(
                        input_gain=body.gain,
                        rms_normalize_target=_rms_target,
                    )
                )
        else:
            if not isinstance(existing_pp, ResearchPreprocessor):
                cnn_worker.set_preprocessor(ResearchPreprocessor())
    except Exception as e:
        logger.exception("Failed to load model for detection session")
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to load model: {e}"},
        )

    # Start detection session
    try:
        pipeline.start_detection_session(
            model_path=body.model_path,
            confidence=body.confidence,
            time_frame=body.time_frame,
            positive_detections=body.positive_detections,
            gain=body.gain,
            model_type=model_type,
            interval_seconds=body.interval_seconds,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})

    session = pipeline.detection_session
    return {
        "message": f"Detection started with {Path(body.model_path).name}",
        "model_path": body.model_path,
        "confidence": body.confidence,
        "time_frame": body.time_frame,
        "positive_detections": body.positive_detections,
        "gain": body.gain,
        "window_seconds": session.window_seconds if session else None,
        "interval_seconds": session.interval_seconds if session else None,
    }


@router.post("/stop")
async def stop_detection(request: Request) -> dict:
    """Stop the current detection session."""
    pipeline = request.app.state.pipeline

    if pipeline.detection_session is None:
        return {"message": "No detection session running."}

    pipeline.stop_detection_session()
    return {"message": "Detection session stopped."}


@router.get("/status", response_model=PipelineStatusResponse)
async def get_status(request: Request) -> PipelineStatusResponse:
    """Get the current pipeline detection status."""
    pipeline = request.app.state.pipeline
    session = pipeline.detection_session

    return PipelineStatusResponse(
        running=session is not None,
        model_path=session.model_path if session else None,
        confidence=session.confidence if session else None,
        time_frame=session.time_frame if session else None,
        positive_detections=session.positive_detections if session else None,
        gain=session.gain if session else None,
        window_seconds=session.window_seconds if session else None,
        interval_seconds=session.interval_seconds if session else None,
        detection_state=pipeline.latest_detection_state,
        drone_probability=pipeline.latest_drone_probability,
    )


@router.get("/log")
async def get_detection_log(request: Request) -> dict:
    """Get the detection log from the current session."""
    pipeline = request.app.state.pipeline
    session = pipeline.detection_session

    if session is None:
        return {"entries": []}

    return {
        "entries": [
            {
                "timestamp": e.timestamp,
                "drone_probability": e.drone_probability,
                "detection_state": e.detection_state,
                "message": e.message,
            }
            for e in session.log
        ]
    }
