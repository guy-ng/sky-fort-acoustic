"""REST API endpoints for model file listing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request

from acoustic.api.models import ModelInfo, ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """List available .pt model files with metadata.

    Scans the directory containing the configured CNN model path.
    """
    settings = request.app.state.settings
    model_dir = Path(settings.cnn_model_path).parent

    models: list[ModelInfo] = []
    if model_dir.is_dir():
        for f in sorted(model_dir.glob("*.pt")):
            stat = f.stat()
            models.append(
                ModelInfo(
                    filename=f.name,
                    path=str(f),
                    size_bytes=stat.st_size,
                    modified=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                )
            )

    return ModelListResponse(models=models)
