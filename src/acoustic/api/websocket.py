"""WebSocket endpoints for live heatmap, target, and status streaming."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from acoustic.api.models import HeatmapHandshake
from acoustic.audio.monitor import DeviceMonitor
from acoustic.types import placeholder_target_from_peak

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/heatmap")
async def ws_heatmap(websocket: WebSocket) -> None:
    """Stream beamforming heatmap as binary float32 frames.

    Protocol:
    1. Send JSON handshake with grid dimensions
    2. Send binary float32 frames (row-major: [elevation][azimuth]) at ~20 Hz
    3. On device disconnect, send JSON: {"type": "device_disconnected", "scanning": true}
    4. On device reconnect, send JSON: {"type": "device_reconnected"} then resume frames

    Re-fetches pipeline each iteration so a lifecycle swap is picked up
    without dropping the WebSocket connection.
    """
    await websocket.accept()
    settings = websocket.app.state.settings
    monitor: DeviceMonitor = websocket.app.state.device_monitor
    status_queue = monitor.subscribe()

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
    device_ok = monitor.detected
    try:
        while True:
            # Check for device status changes (non-blocking)
            try:
                while True:
                    status = status_queue.get_nowait()
                    if not status.detected and device_ok:
                        await websocket.send_json({
                            "type": "device_disconnected",
                            "scanning": status.scanning,
                        })
                        device_ok = False
                        last_map_id = None
                    elif status.detected and not device_ok:
                        await websocket.send_json({"type": "device_reconnected"})
                        device_ok = True
            except asyncio.QueueEmpty:
                pass

            if device_ok:
                try:
                    pipeline = websocket.app.state.pipeline
                    current_map = pipeline.latest_map
                    if current_map is not None and id(current_map) != last_map_id:
                        last_map_id = id(current_map)
                        frame = current_map.T.astype(np.float32).tobytes()
                        await websocket.send_bytes(frame)
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception:
                    logger.debug("Heatmap frame skipped (pipeline may be mid-swap)")

            await asyncio.sleep(0.05)  # 20 Hz poll
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Heatmap WebSocket client disconnected")
    finally:
        monitor.unsubscribe(status_queue)


@router.websocket("/ws/targets")
async def ws_targets(websocket: WebSocket) -> None:
    """Stream target state updates as JSON arrays at ~2 Hz.

    Sends {"type": "device_disconnected"} / {"type": "device_reconnected"}
    on device state changes. Re-fetches pipeline each iteration to survive
    lifecycle swaps.
    """
    await websocket.accept()
    monitor: DeviceMonitor = websocket.app.state.device_monitor
    status_queue = monitor.subscribe()
    device_ok = monitor.detected

    try:
        while True:
            # Check for device status changes (non-blocking)
            try:
                while True:
                    status = status_queue.get_nowait()
                    if not status.detected and device_ok:
                        await websocket.send_json({
                            "type": "device_disconnected",
                            "scanning": status.scanning,
                        })
                        device_ok = False
                    elif status.detected and not device_ok:
                        await websocket.send_json({"type": "device_reconnected"})
                        device_ok = True
            except asyncio.QueueEmpty:
                pass

            if device_ok:
                try:
                    pipeline = websocket.app.state.pipeline
                    peak = pipeline.latest_peak
                    if peak is not None:
                        target = placeholder_target_from_peak(peak)
                        await websocket.send_json([target])
                    else:
                        await websocket.send_json([])
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception:
                    logger.debug("Target frame skipped (pipeline may be mid-swap)")
                    await websocket.send_json([])
            await asyncio.sleep(0.5)  # 2 Hz -- targets change slowly
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Targets WebSocket client disconnected")
    finally:
        monitor.unsubscribe(status_queue)


@router.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    """Push device status changes to clients in real time.

    Protocol:
    1. On connect, immediately sends current device status as JSON
    2. On each state change, sends updated status:
       {"device_detected": bool, "device_name": str|null, "scanning": bool}
    """
    await websocket.accept()
    monitor: DeviceMonitor = websocket.app.state.device_monitor
    queue = monitor.subscribe()

    try:
        # Send current state immediately
        current = monitor.current_status()
        await websocket.send_json({
            "device_detected": current.detected,
            "device_name": current.name,
            "scanning": current.scanning,
        })

        # Stream state changes
        while True:
            status = await queue.get()
            await websocket.send_json({
                "device_detected": status.detected,
                "device_name": status.name,
                "scanning": status.scanning,
            })
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Status WebSocket client disconnected")
    finally:
        monitor.unsubscribe(queue)
