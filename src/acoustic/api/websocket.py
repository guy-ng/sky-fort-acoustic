"""WebSocket endpoints for live heatmap, target, event, and status streaming."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from acoustic.api.models import HeatmapHandshake
from acoustic.audio.monitor import DeviceMonitor
from acoustic.tracking.events import EventBroadcaster
from acoustic.recording.manager import RecordingManager
from acoustic.training.manager import TrainingManager, TrainingProgress, TrainingStatus

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
                            "targets": [],
                            "drone_probability": None,
                            "detection_state": None,
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
                    targets = pipeline.latest_targets
                    drone_prob = pipeline.latest_drone_probability
                    det_state = pipeline.latest_detection_state
                    await websocket.send_json({
                        "targets": targets,
                        "drone_probability": drone_prob,
                        "detection_state": det_state,
                    })
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception:
                    logger.debug("Target frame skipped (pipeline may be mid-swap)")
                    await websocket.send_json({"targets": [], "drone_probability": None, "detection_state": None})
            await asyncio.sleep(0.5)  # 2 Hz -- targets change slowly
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Targets WebSocket client disconnected")
    finally:
        monitor.unsubscribe(status_queue)


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Stream detection events (new, update, lost) to external consumers.

    Separate from /ws/targets which streams current target state for the UI.
    This endpoint broadcasts detection lifecycle events as they occur.
    """
    await websocket.accept()
    broadcaster: EventBroadcaster | None = getattr(websocket.app.state, "event_broadcaster", None)
    if broadcaster is None:
        # CNN not initialized -- close with reason
        await websocket.close(code=1011, reason="Event broadcasting not available")
        return

    queue = broadcaster.subscribe()
    try:
        while True:
            event_data = await queue.get()
            await websocket.send_json(event_data)
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Events WebSocket client disconnected")
    finally:
        broadcaster.unsubscribe(queue)


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


@router.websocket("/ws/recording")
async def ws_recording(websocket: WebSocket) -> None:
    """Stream recording state (status, elapsed, remaining, level_db) at 10Hz.

    Sends JSON with current recording state on every change,
    polled at 10Hz for responsive level meter feedback.
    """
    await websocket.accept()
    manager: RecordingManager = websocket.app.state.recording_manager
    last_state: dict | None = None
    try:
        while True:
            state = manager.get_state()
            # Always send while recording (timer/level change every tick)
            # Only deduplicate when idle to avoid unnecessary traffic
            if state.get("status") == "recording" or state != last_state:
                await websocket.send_json(state)
                last_state = state
            await asyncio.sleep(0.1)  # 10Hz for level meter
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Recording WebSocket client disconnected")


def _progress_to_ws_dict(progress: TrainingProgress) -> dict:
    """Format training progress for WebSocket transmission (per D-12, D-13)."""
    d: dict = {"status": progress.status.value}
    if progress.status in (TrainingStatus.RUNNING, TrainingStatus.COMPLETED):
        d.update({
            "epoch": progress.epoch,
            "total_epochs": progress.total_epochs,
            "train_loss": progress.train_loss,
            "val_loss": progress.val_loss,
            "val_acc": progress.val_acc,
            "confusion_matrix": {
                "tp": progress.tp,
                "fp": progress.fp,
                "tn": progress.tn,
                "fn": progress.fn,
            },
        })
    elif progress.status == TrainingStatus.FAILED:
        d["error"] = progress.error or "Unknown error"
    return d


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    """Push training progress updates to clients.

    Protocol (per D-13):
    1. On connect, send current status JSON
    2. If status is completed/failed, include last results
    3. Push updates when epoch or status changes
    4. No periodic heartbeats
    """
    await websocket.accept()
    manager: TrainingManager = websocket.app.state.training_manager

    # Send current state immediately
    progress = manager.get_progress()
    await websocket.send_json(_progress_to_ws_dict(progress))

    last_epoch = progress.epoch
    last_status = progress.status
    try:
        while True:
            await asyncio.sleep(0.5)  # Poll at 2 Hz
            progress = manager.get_progress()
            if progress.epoch != last_epoch or progress.status != last_status:
                await websocket.send_json(_progress_to_ws_dict(progress))
                last_epoch = progress.epoch
                last_status = progress.status
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Training WebSocket client disconnected")
