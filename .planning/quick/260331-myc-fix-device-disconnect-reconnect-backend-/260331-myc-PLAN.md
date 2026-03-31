---
type: quick
tasks: 3
autonomous: true
files_modified:
  - src/acoustic/audio/capture.py
  - src/acoustic/pipeline.py
  - src/acoustic/api/websocket.py
  - src/acoustic/main.py
---

<objective>
Fix backend crash and stale state when UMA-16v2 USB device is unplugged/replugged.

Purpose: The DeviceMonitor detects hot-plug events but nothing reacts to them -- AudioCapture dies on unplug, pipeline reads stale data, WS handlers crash accessing dead pipeline state, and replug never recreates the audio chain.

Output: Resilient device lifecycle -- unplug gracefully degrades (WS sends empty frames, status shows disconnected), replug automatically recreates capture + pipeline.
</objective>

<context>
@src/acoustic/main.py
@src/acoustic/audio/capture.py
@src/acoustic/audio/monitor.py
@src/acoustic/pipeline.py
@src/acoustic/api/websocket.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Harden AudioCapture.stop() and add Pipeline.clear_state()</name>
  <files>src/acoustic/audio/capture.py, src/acoustic/pipeline.py</files>
  <action>
1. In `capture.py` -- `AudioCapture.stop()`: Wrap both `self._stream.stop()` and `self._stream.close()` in individual try/except blocks catching `sd.PortAudioError` and `Exception`. Log warnings on failure but do not raise. The device may already be gone when stop() is called.

2. In `pipeline.py` -- Add a `clear_state()` method to `BeamformingPipeline`:
   - Sets `self.latest_map = None`
   - Sets `self.latest_peak = None`
   - Sets `self._last_process_time = None`
   - Logs "Pipeline state cleared (device disconnected)"

3. In `pipeline.py` -- Add a `restart(ring_buffer: AudioRingBuffer)` method:
   - Calls `self.stop()` (safe even if already stopped)
   - Calls `self.clear_state()`
   - Calls `self.start(ring_buffer)` with the new ring buffer
   - Logs "Pipeline restarted with new ring buffer"
  </action>
  <verify>
    python -c "
from acoustic.audio.capture import AudioCapture, AudioRingBuffer
from acoustic.pipeline import BeamformingPipeline
from acoustic.config import AcousticSettings
# Verify clear_state exists and works
p = BeamformingPipeline(AcousticSettings())
p.clear_state()
assert p.latest_map is None and p.latest_peak is None
print('OK: clear_state works')
# Verify restart exists
assert hasattr(p, 'restart')
print('OK: restart method exists')
"
  </verify>
  <done>AudioCapture.stop() never raises on dead device. Pipeline has clear_state() and restart() methods.</done>
</task>

<task type="auto">
  <name>Task 2: Make WS handlers resilient to pipeline swap</name>
  <files>src/acoustic/api/websocket.py</files>
  <action>
1. In `ws_heatmap`: Do NOT cache `pipeline` outside the loop. Each iteration, read `pipeline = websocket.app.state.pipeline`. This way when the lifecycle manager swaps the pipeline reference, the WS handler picks up the new one. Wrap the map access and send in a try/except that catches `Exception` -- on error, log debug and continue the loop (the pipeline may be mid-swap). When `current_map is None`, just sleep and continue (do not send anything -- this naturally happens during disconnect).

2. In `ws_targets`: Same pattern -- re-read `pipeline = websocket.app.state.pipeline` each iteration. Wrap peak access and send in try/except. When `peak is None`, send empty array `[]` as it already does.

3. Keep the outer `except (WebSocketDisconnect, RuntimeError)` for clean client disconnect handling.

Key: The `settings` object does not change, so it can stay cached. Only `pipeline` must be re-fetched each iteration.
  </action>
  <verify>
    python -c "
import ast, sys
with open('src/acoustic/api/websocket.py') as f:
    source = f.read()
# Verify pipeline is fetched inside while loop, not before it
tree = ast.parse(source)
print('File parses OK')
# Check that pipeline assignment appears after while True
assert 'websocket.app.state.pipeline' in source
# Check there's exception handling inside the loop
assert 'except' in source
print('OK: WS handlers updated')
"
  </verify>
  <done>WS heatmap and target handlers re-fetch pipeline each iteration and gracefully handle None/swap states without crashing.</done>
</task>

<task type="auto">
  <name>Task 3: Add device lifecycle background task in main.py</name>
  <files>src/acoustic/main.py</files>
  <action>
Add an async function `_device_lifecycle_loop(app: FastAPI)` and start it as a background asyncio task in `lifespan()`. This is the core fix -- it bridges DeviceMonitor events to AudioCapture/Pipeline lifecycle.

1. Define `_device_lifecycle_loop(app)`:
   - Subscribe to `app.state.device_monitor` via `monitor.subscribe()`
   - Loop forever reading from the subscription queue
   - On `status.detected == False` (device disconnected):
     a. Log warning "Device disconnected -- stopping audio pipeline"
     b. `app.state.pipeline.stop()` wrapped in try/except
     c. `app.state.pipeline.clear_state()`
     d. Try `app.state.capture.stop()` wrapped in try/except (stream may be dead)
     e. Set `app.state.capture = None` (sentinel: no active capture)
   - On `status.detected == True` (device reconnected):
     a. Log info "Device reconnected -- restarting audio pipeline"
     b. Get fresh device info from `app.state.device_monitor.device_info`
     c. Create new `AudioCapture` with `device=device_info.index`, same settings as original
     d. Call `capture.start()`
     e. Set `app.state.capture = capture`
     f. Call `app.state.pipeline.restart(capture.ring)` to restart with new ring buffer
     g. Log info "Audio pipeline restarted successfully"
   - Wrap the entire handler body in try/except to log and continue on unexpected errors
   - On `asyncio.CancelledError`, unsubscribe and return cleanly

2. In `lifespan()`, after setting `app.state.device_monitor`:
   - Only start the lifecycle task when NOT in simulated mode (`settings.audio_source != "simulated"`)
   - `lifecycle_task = asyncio.create_task(_device_lifecycle_loop(app))`
   - In the shutdown section (after yield), cancel the task: `lifecycle_task.cancel()` then `await asyncio.gather(lifecycle_task, return_exceptions=True)`

3. In `health()` endpoint: Guard `capture.ring.overflow_count` and `capture.last_frame_time` against `capture is None` -- return 0 and None respectively when no capture is active. Update status to return "degraded" when capture is None.
  </action>
  <verify>
    python -c "
import ast
with open('src/acoustic/main.py') as f:
    source = f.read()
tree = ast.parse(source)
# Check the lifecycle function exists
func_names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
assert '_device_lifecycle_loop' in func_names, f'Missing _device_lifecycle_loop, found: {func_names}'
print('OK: lifecycle function exists')
assert 'create_task' in source
print('OK: background task created')
assert 'capture is None' in source or 'capture is not None' in source
print('OK: health endpoint guards null capture')
"
  </verify>
  <done>
    - Device unplug: pipeline stops, capture cleaned up, WS handlers send empty/nothing, health reports degraded
    - Device replug: new AudioCapture created, pipeline restarted with new ring buffer, WS handlers resume streaming
    - Simulated mode: lifecycle task not started (no hardware to monitor)
  </done>
</task>

</tasks>

<verification>
Manual test sequence (requires hardware or simulated disconnect):
1. Start service with device connected -- heatmap streams normally
2. Unplug device -- heatmap goes empty, /health shows degraded, no backend crash
3. Replug device -- heatmap resumes streaming within ~3-6 seconds (one monitor poll cycle)
4. WS connections stay alive through the entire cycle

Code-level:
- `python -c "from acoustic.main import app; print('Import OK')"` -- no import errors
- All three task verify commands pass
</verification>

<success_criteria>
- Backend never crashes on USB device unplug
- Audio pipeline automatically restarts on device replug
- WebSocket connections survive disconnect/reconnect cycle
- Health endpoint accurately reflects degraded state during disconnect
</success_criteria>
