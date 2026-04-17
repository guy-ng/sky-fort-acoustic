---
phase: quick
plan: 260417-bgv
type: execute
wave: 1
depends_on: []
files_modified:
  - src/acoustic/api/routes.py
  - src/acoustic/pipeline.py
  - web/src/components/heatmap/BeamformingControls.tsx
  - web/src/pages/MonitorPage.tsx
  - web/src/hooks/useBfPeaksSocket.ts
autonomous: true
must_haves:
  truths:
    - "User can click a button to start recording target locations"
    - "User can click the button again to stop recording"
    - "A JSON file is saved to data/target_recordings/ with timestamped target locations"
    - "The button shows recording state (active/idle)"
  artifacts:
    - path: "src/acoustic/api/routes.py"
      provides: "POST /api/target-recording/start and /stop endpoints"
    - path: "src/acoustic/pipeline.py"
      provides: "target_recording_state property + start/stop methods + accumulation logic"
    - path: "web/src/components/heatmap/BeamformingControls.tsx"
      provides: "Target recording button with state indicator"
  key_links:
    - from: "BeamformingControls.tsx"
      to: "/api/target-recording/start"
      via: "fetch POST on button click"
    - from: "pipeline.py accumulation"
      to: "data/target_recordings/*.json"
      via: "json.dump on stop"
---

<objective>
Add a "Record Targets" feature that captures timestamped target locations (az_deg, el_deg, pan_deg, tilt_deg, confidence, class_label) to a JSON file on disk. The user starts/stops recording from the Monitor page controls panel.

Purpose: Enables collecting ground-truth target trajectory data for analysis, replay, and training validation.
Output: Backend recording logic in pipeline.py, REST endpoints in routes.py, UI button in BeamformingControls.tsx.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/acoustic/api/routes.py
@src/acoustic/pipeline.py
@src/acoustic/api/websocket.py
@web/src/components/heatmap/BeamformingControls.tsx
@web/src/pages/MonitorPage.tsx
@web/src/hooks/useBfPeaksSocket.ts

<interfaces>
From src/acoustic/tracking/tracker.py:
```python
@dataclass
class TrackedTarget:
    id: str
    class_label: str
    az_deg: float
    el_deg: float
    confidence: float
    speed_mps: float | None = None
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    lost: bool = False

class TargetTracker:
    def get_target_states(self) -> list[dict]:
        # Returns dicts with: id, class_label, speed_mps, az_deg, el_deg, pan_deg, tilt_deg, confidence
```

From src/acoustic/pipeline.py (existing recording pattern):
```python
# Raw recording pattern already exists:
pipeline.raw_recording_state  # -> dict with status, id, elapsed_s, remaining_s
pipeline.start_raw_recording() -> str  # returns rec_id
pipeline.stop_raw_recording() -> dict | None
```

From src/acoustic/api/websocket.py ws/bf-peaks:
```python
# Already broadcasts raw_recording state and playback state.
# We will add target_recording state to this same payload.
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add target recording logic to pipeline and API endpoints</name>
  <files>src/acoustic/pipeline.py, src/acoustic/api/routes.py</files>
  <action>
**Pipeline (pipeline.py):**

Add target recording state management to the `AcousticPipeline` class, following the existing raw_recording pattern:

1. Add instance variables in `__init__`:
   - `self._target_recording: bool = False`
   - `self._target_recording_id: str | None = None`
   - `self._target_recording_start: float | None = None`
   - `self._target_recording_frames: list[dict] = []`

2. Add a `_sample_targets` method called from `_process_chunk` (or wherever `latest_targets` gets updated -- grep for where `self._tracker.update_multi` is called). After the tracker update, if `self._target_recording` is True, append a snapshot:
   ```python
   def _sample_targets(self) -> None:
       if not self._target_recording:
           return
       import time
       targets = self.latest_targets
       if targets:
           self._target_recording_frames.append({
               "t": round(time.time(), 3),
               "targets": targets,
           })
   ```
   Call `self._sample_targets()` at the end of `_process_chunk`, after peak detection and tracker update complete.

3. Add `start_target_recording() -> str`:
   - Generate ID with `datetime.now().strftime("%Y%m%d_%H%M%S")`
   - Set `_target_recording = True`, clear frames list, record start time with `time.time()`
   - Return the ID

4. Add `stop_target_recording() -> dict | None`:
   - If not recording, return None
   - Set `_target_recording = False`
   - Write JSON file to `data/target_recordings/{id}.json` with structure:
     ```json
     {
       "id": "20260417_143022",
       "started_at": 1745123422.123,
       "stopped_at": 1745123482.456,
       "duration_s": 60.333,
       "total_samples": 1200,
       "frames": [
         {"t": 1745123422.200, "targets": [{"id": "...", "az_deg": 12.3, ...}]},
         ...
       ]
     }
     ```
   - Create `data/target_recordings/` dir if it doesn't exist (use `Path.mkdir(parents=True, exist_ok=True)`)
   - Return `{"id": id, "status": "saved", "path": str(path), "total_samples": len(frames), "duration_s": ...}`

5. Add `target_recording_state` property:
   ```python
   @property
   def target_recording_state(self) -> dict:
       if self._target_recording:
           import time
           elapsed = time.time() - (self._target_recording_start or 0)
           return {"status": "recording", "id": self._target_recording_id, "elapsed_s": round(elapsed, 1), "samples": len(self._target_recording_frames)}
       return {"status": "idle"}
   ```

**Routes (routes.py):**

Add 3 endpoints following the existing raw-recording pattern:

1. `POST /api/target-recording/start` -- calls `pipeline.start_target_recording()`, returns 409 if already recording
2. `POST /api/target-recording/stop` -- calls `pipeline.stop_target_recording()`, returns 404 if not recording
3. `GET /api/target-recording/status` -- returns `pipeline.target_recording_state`

Also add `GET /api/target-recordings` to list saved JSON files (similar to raw-recordings list but reads JSON metadata from each file -- just id, duration_s, total_samples, file size).
  </action>
  <verify>
    <automated>cd /Users/guyelisha/Projects/sky-fort-acoustic && python -c "from acoustic.pipeline import AcousticPipeline; print('pipeline imports OK')" && python -c "from acoustic.api.routes import router; print('routes imports OK')"</automated>
  </verify>
  <done>
    - Pipeline has start/stop target recording methods and state property
    - 4 new API endpoints exist: start, stop, status, list
    - JSON files written to data/target_recordings/ with timestamped frames
  </done>
</task>

<task type="auto">
  <name>Task 2: Add target recording button to UI and wire WebSocket state</name>
  <files>web/src/components/heatmap/BeamformingControls.tsx, web/src/pages/MonitorPage.tsx, web/src/hooks/useBfPeaksSocket.ts, src/acoustic/api/websocket.py</files>
  <action>
**WebSocket (websocket.py):**

In `ws_bf_peaks`, add `target_recording` to the JSON payload alongside `raw_recording` and `playback`:
```python
"target_recording": pipeline.target_recording_state,
```

**Hook (useBfPeaksSocket.ts):**

Add `TargetRecordingState` interface:
```typescript
export interface TargetRecordingState {
  status: 'idle' | 'recording'
  id?: string
  elapsed_s?: number
  samples?: number
}
```

Add `target_recording: TargetRecordingState` to `BfPeaksData` interface.

**MonitorPage.tsx:**

Add target recording state and handlers (same pattern as raw recording):
```typescript
const targetRecordingState = bfPeaks?.target_recording ?? { status: 'idle' as const }

const handleTargetRecordStart = useCallback(async () => {
  await fetch(`${API_BASE}/api/target-recording/start`, { method: 'POST' })
}, [])

const handleTargetRecordStop = useCallback(async () => {
  await fetch(`${API_BASE}/api/target-recording/stop`, { method: 'POST' })
}, [])
```

Pass to BeamformingControls:
```
onTargetRecordStart={handleTargetRecordStart}
onTargetRecordStop={handleTargetRecordStop}
targetRecordingState={targetRecordingState}
```

**BeamformingControls.tsx:**

Add props for target recording:
```typescript
onTargetRecordStart: () => void
onTargetRecordStop: () => void
targetRecordingState: { status: string; elapsed_s?: number; samples?: number }
```

Add a new section below the raw recording button (inside the border-t div, or in a new border-t section). Use the same button pattern:

- When idle: grey button with crosshairs icon (`track_changes`), label "Record Targets"
- When recording: red/danger button with pulsing dot, label "Stop Target Rec" with elapsed time and sample count below
- Style matches the existing raw recording button exactly (same classes)

The section should look like:
```tsx
{/* Target location recording */}
<div className="border-t border-hud-border/50 pt-2">
  {isTargetRecording ? (
    <div className="space-y-1">
      <button
        onClick={onTargetRecordStop}
        className="w-full py-1.5 rounded bg-amber-600/80 hover:bg-amber-600 text-white text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1.5"
      >
        <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
        Stop Target Rec
      </button>
      <div className="flex justify-between text-hud-text-dim tabular-nums">
        <span>{targetRecordingState.elapsed_s?.toFixed(0) ?? 0}s</span>
        <span>{targetRecordingState.samples ?? 0} samples</span>
      </div>
    </div>
  ) : (
    <button
      onClick={onTargetRecordStart}
      className="w-full py-1.5 rounded bg-hud-border hover:bg-hud-accent/30 text-hud-text text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1.5"
    >
      <span className="material-symbols-outlined text-sm">track_changes</span>
      Record Targets
    </button>
  )}
</div>
```

Use amber color (not red) to visually distinguish from raw audio recording.
  </action>
  <verify>
    <automated>cd /Users/guyelisha/Projects/sky-fort-acoustic/web && npx tsc --noEmit 2>&1 | head -20</automated>
  </verify>
  <done>
    - BeamformingControls shows "Record Targets" button below raw recording
    - Button toggles between idle (grey) and recording (amber pulse) states
    - Elapsed time and sample count displayed during recording
    - WebSocket broadcasts target_recording state alongside existing fields
    - TypeScript compiles without errors
  </done>
</task>

</tasks>

<verification>
1. Backend imports cleanly: `python -c "from acoustic.pipeline import AcousticPipeline; from acoustic.api.routes import router"`
2. Frontend compiles: `cd web && npx tsc --noEmit`
3. Manual: Start the service, click "Record Targets", wait a few seconds, click "Stop Target Rec", check `data/target_recordings/` for the JSON file
</verification>

<success_criteria>
- "Record Targets" button visible in the Controls panel on the Monitor page
- Button shows recording state with elapsed time and sample count
- Stopping recording writes a JSON file to data/target_recordings/ with timestamped target snapshots
- JSON file contains: id, started_at, stopped_at, duration_s, total_samples, and frames array
- Each frame has Unix timestamp and target array with az_deg, el_deg, pan_deg, tilt_deg, confidence, class_label
</success_criteria>

<output>
After completion, create `.planning/quick/260417-bgv-add-api-record-button-to-generate-json-f/260417-bgv-SUMMARY.md`
</output>
