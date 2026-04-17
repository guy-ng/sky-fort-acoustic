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

    # Send handshake with grid dimensions.
    # Height is halved because pipeline crops to el >= 0 (planar array fold).
    full_el = int((2 * settings.el_range / settings.el_resolution) + 1)
    half_el = full_el // 2 + 1  # el=0 through el=+el_range
    handshake = HeatmapHandshake(
        width=int((2 * settings.az_range / settings.az_resolution) + 1),
        height=half_el,
        az_min=-settings.az_range,
        az_max=settings.az_range,
        el_min=0.0,
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
                        # Map is (n_az, n_el_half). .T gives (n_el_half, n_az) row-major.
                        # Row 0 = el=0 (horizon), last row = el=+el_max.
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
            settings = websocket.app.state.settings
            interval = 1.0 / max(0.5, min(settings.ws_targets_hz, 10.0))
            await asyncio.sleep(interval)
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


@router.websocket("/ws/spectrum")
async def ws_spectrum(websocket: WebSocket) -> None:
    """Stream frequency band energy levels at ~10 Hz.

    Protocol: sends JSON {"bands": [{"name": str, "fmin": int, "fmax": int, "db": float}, ...]}
    """
    await websocket.accept()
    try:
        while True:
            pipeline = websocket.app.state.pipeline
            spectrum = pipeline.latest_spectrum
            if spectrum is not None:
                await websocket.send_json(spectrum)
            await asyncio.sleep(0.1)  # 10 Hz
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Spectrum WebSocket client disconnected")


@router.websocket("/ws/bf-peaks")
async def ws_bf_peaks(websocket: WebSocket) -> None:
    """Stream beamforming peak detections at ~10 Hz.

    Protocol: sends JSON with peak az/el degrees and power for each detected source.
    """
    await websocket.accept()
    try:
        while True:
            pipeline = websocket.app.state.pipeline
            peaks = pipeline.latest_peaks
            if peaks:
                peak_data = [
                    {
                        "az_deg": round(p.az_deg, 2),
                        "el_deg": round(p.el_deg, 2),
                        "power": round(p.power, 4),
                        "threshold": round(p.threshold, 4),
                    }
                    for p in peaks
                ]
            else:
                peak_data = []
            primary = peaks[0] if peaks else None
            await websocket.send_json({
                "peaks": peak_data,
                "primary": {
                    "az_deg": round(primary.az_deg, 2),
                    "el_deg": round(primary.el_deg, 2),
                } if primary else None,
                "mass_center": pipeline.latest_mass_center,
                "raw_recording": pipeline.raw_recording_state,
                "playback": pipeline.playback_state,
                "target_recording": pipeline.target_recording_state,
            })
            await asyncio.sleep(0.1)  # 10 Hz
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("BF-peaks WebSocket client disconnected")


@router.websocket("/ws/sound-level")
async def ws_sound_level(websocket: WebSocket) -> None:
    """Stream raw (pre-gain) mic RMS level in dBFS at 10 Hz.

    Protocol: sends `{"level_db": float | null}` every 100 ms. `null` is sent
    when no chunk has been processed yet (e.g., device not connected).
    """
    await websocket.accept()
    try:
        while True:
            pipeline = websocket.app.state.pipeline
            level = pipeline.latest_audio_level_db
            await websocket.send_json({"level_db": level})
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Sound-level WebSocket client disconnected")


@router.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket) -> None:
    """Stream pipeline detection status and log entries at 2 Hz.

    Sends JSON with detection state, drone probability, and new log entries.
    Used by the Pipeline tab for real-time detection monitoring.
    """
    await websocket.accept()
    last_log_len = 0
    try:
        while True:
            pipeline = websocket.app.state.pipeline
            session = pipeline.detection_session
            if session is not None:
                entries = list(session.log)
                new_entries = entries[last_log_len:]
                last_log_len = len(entries)
                await websocket.send_json({
                    "running": True,
                    "detection_state": pipeline.latest_detection_state,
                    "drone_probability": pipeline.latest_drone_probability,
                    "new_log_entries": [
                        {
                            "timestamp": e.timestamp,
                            "drone_probability": e.drone_probability,
                            "detection_state": e.detection_state,
                            "message": e.message,
                        }
                        for e in new_entries
                    ],
                })
            else:
                last_log_len = 0
                await websocket.send_json({
                    "running": False,
                    "detection_state": None,
                    "drone_probability": None,
                    "new_log_entries": [],
                })
            await asyncio.sleep(0.5)  # 2 Hz
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Pipeline WebSocket client disconnected")


def _progress_to_ws_dict(progress: TrainingProgress) -> dict:
    """Format training progress for WebSocket transmission (per D-12, D-13)."""
    d: dict = {
        "status": progress.status.value,
        "model_name": progress.model_name,
        "cache_loaded": progress.cache_loaded,
        "cache_total": progress.cache_total,
    }
    if progress.status in (TrainingStatus.RUNNING, TrainingStatus.COMPLETED):
        d.update({
            "epoch": progress.epoch,
            "total_epochs": progress.total_epochs,
            "batch": progress.batch,
            "total_batches": progress.total_batches,
            "train_loss": progress.train_loss,
            "val_loss": progress.val_loss,
            "val_acc": progress.val_acc,
            "stage": progress.stage,
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
    4. Heartbeat every 15s to keep connection alive during long epochs
    """
    await websocket.accept()
    manager: TrainingManager = websocket.app.state.training_manager

    # Send current state immediately
    progress = manager.get_progress()
    await websocket.send_json(_progress_to_ws_dict(progress))

    last_epoch = progress.epoch
    last_status = progress.status
    last_batch = progress.batch
    last_cache = progress.cache_loaded
    last_send_time = asyncio.get_event_loop().time()
    try:
        while True:
            await asyncio.sleep(0.5)  # Poll at 2 Hz
            progress = manager.get_progress()
            now = asyncio.get_event_loop().time()
            changed = (
                progress.epoch != last_epoch
                or progress.status != last_status
                or progress.batch != last_batch
                or progress.cache_loaded != last_cache
            )
            heartbeat_due = (now - last_send_time) >= 15.0
            if changed or heartbeat_due:
                await websocket.send_json(_progress_to_ws_dict(progress))
                last_epoch = progress.epoch
                last_status = progress.status
                last_batch = progress.batch
                last_cache = progress.cache_loaded
                last_send_time = now
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("Training WebSocket client disconnected")
