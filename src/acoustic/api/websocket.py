"""WebSocket endpoints for live heatmap and target streaming."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from acoustic.api.models import HeatmapHandshake
from acoustic.types import placeholder_target_from_peak

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/heatmap")
async def ws_heatmap(websocket: WebSocket) -> None:
    """Stream beamforming heatmap as binary float32 frames.

    Protocol:
    1. Send JSON handshake with grid dimensions
    2. Send binary float32 frames (row-major: [elevation][azimuth]) at ~20 Hz
    """
    await websocket.accept()
    settings = websocket.app.state.settings
    pipeline = websocket.app.state.pipeline

    # Send handshake with grid dimensions
    handshake = HeatmapHandshake(
        width=int((2 * settings.az_range / settings.az_resolution) + 1),
        height=int((2 * settings.el_range / settings.el_resolution) + 1),
        az_min=-settings.az_range,
        az_max=settings.az_range,
        el_min=-settings.el_range,
        el_max=settings.el_range,
    )
    await websocket.send_json(handshake.model_dump())

    last_map_id = None
    try:
        while True:
            current_map = pipeline.latest_map
            if current_map is not None and id(current_map) != last_map_id:
                last_map_id = id(current_map)
                # Transpose to row-major [elevation][azimuth] before sending
                frame = current_map.T.astype(np.float32).tobytes()
                await websocket.send_bytes(frame)
            await asyncio.sleep(0.05)  # 20 Hz poll
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Heatmap WebSocket client disconnected")


@router.websocket("/ws/targets")
async def ws_targets(websocket: WebSocket) -> None:
    """Stream target state updates as JSON arrays at ~2 Hz."""
    await websocket.accept()
    pipeline = websocket.app.state.pipeline

    try:
        while True:
            peak = pipeline.latest_peak
            if peak is not None:
                target = placeholder_target_from_peak(peak)
                await websocket.send_json([target])
            else:
                await websocket.send_json([])
            await asyncio.sleep(0.5)  # 2 Hz -- targets change slowly
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Targets WebSocket client disconnected")
