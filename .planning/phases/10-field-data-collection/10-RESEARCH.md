# Phase 10: Field Data Collection - Research

**Researched:** 2026-04-02
**Domain:** Audio recording, metadata management, FastAPI file handling, React recording UI
**Confidence:** HIGH

## Summary

Phase 10 adds a recording subsystem that captures live audio from the UMA-16 ring buffer, saves mono WAV files with sidecar JSON metadata, and exposes the entire workflow through a web UI. The backend is straightforward: read chunks from the existing `AudioRingBuffer`, average 16 channels to mono, write to WAV via `soundfile`, and manage metadata as JSON sidecar files. The frontend needs a recording control panel with start/stop, an inline label form, and a recordings list with metadata editing.

The key integration constraint is that output directories must be directly consumable by `collect_wav_files()` from the training pipeline -- meaning `{root}/{label}/` structure with WAV files inside. The `DroneAudioDataset` already handles multi-channel to mono conversion and resampling, so the recording format has some flexibility, but saving at 16kHz mono is the lowest-friction option.

**Primary recommendation:** Save recordings as 16kHz mono WAV (matches `MelConfig.sample_rate=16000`) into `data/field/{label}/` directories. Use sidecar JSON for metadata. Backend exposes REST endpoints for recording lifecycle and metadata CRUD. Frontend adds a recording panel to the existing dashboard layout.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Record-first, label later. User hits record immediately without pre-selecting a label. Recording saves to a temporary/unlabeled location until labeled.
- **D-02:** Configurable max recording duration with auto-stop. Default TBD by implementation (e.g., 5 minutes). Prevents accidentally huge files.
- **D-03:** Live recording feedback shows elapsed time, remaining time (when max is set), and a simple audio level meter so the user can confirm capture is working.
- **D-04:** After stopping, an inline label form appears in the UI (not a modal). User selects top-level label (required), optional sub-label and metadata, then saves. Recording is filed into the correct directory on save.
- **D-05:** Hierarchical labeling: fixed top-level labels (drone, background, other) determine directory placement. Sub-labels within each top-level (e.g., Mavic, Matrice, 5-inch under "drone") are stored in metadata only.
- **D-06:** Top-level labels are a fixed preset list (drone, background, other). No user-defined top-level labels. This ensures compatibility with `collect_wav_files()` and the training pipeline's `label_map`.
- **D-07:** Sub-labels are user-definable (free text or pick from suggestions). Stored in sidecar JSON metadata, not directory structure.
- **D-08:** Sidecar JSON file per recording (e.g., `rec_001.json` alongside `rec_001.wav`). Simple, portable, no database dependency.
- **D-09:** Only top-level label is required. All other metadata fields are optional.
- **D-10:** Metadata fields: top-level label (required), sub-label (e.g., drone type), distance estimate, altitude estimate, weather/conditions, free-text notes, recording timestamp, duration.
- **D-11:** Metadata is editable after recording -- user can update any field from the recordings list or detail view.
- **D-12:** Mono downmix -- average all 16 channels to a single channel for recording. Matches research data format and saves ~16x disk space vs full 16-channel.
- **D-13:** Claude's Discretion on sample rate: choose between saving at 16kHz or 48kHz native.

### Claude's Discretion
- Recording file naming convention (timestamps, sequential IDs, UUIDs)
- Where to place the recording module code (e.g., `src/acoustic/recording/` or extend existing audio module)
- How the inline label form integrates with existing frontend component patterns
- Audio level meter implementation (RMS from ring buffer vs dedicated channel)
- How temporary/unlabeled recordings are managed before the user labels them
- Directory location: `data/recordings/{label}/` vs `audio-data/data/{label}/`

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COL-01 | User can start a labeled recording session from the web UI, specifying drone type and recording conditions | Recording REST API (start/stop/label), React recording panel with inline label form, WebSocket for live feedback |
| COL-02 | User can attach and edit metadata on recordings (drone type, distance, altitude, conditions, notes) | Sidecar JSON schema, PATCH endpoint for metadata updates, recordings list with edit capability |
| COL-03 | Recordings saved into directory structure that training pipeline can directly consume | `data/field/{label}/` layout compatible with `collect_wav_files()` and `TrainingConfig.data_root` |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| soundfile | 0.13.1 | WAV file writing | Already used in `DroneAudioDataset` for reading. `sf.write()` handles mono float32 natively. |
| NumPy | >=1.26 | Audio buffer manipulation | Channel averaging (`audio.mean(axis=1)`), RMS calculation. Already the DSP core. |
| FastAPI | >=0.135 | REST endpoints | Existing pattern -- `APIRouter` with prefix. |
| Pydantic | v2 | Request/response models | Existing pattern -- `BaseModel` for API models, `BaseSettings` for config. |
| React 19 | ^19 | Recording UI | Existing frontend framework. |
| TanStack Query | ^5 | Server state for recordings list | Existing pattern -- `useQuery` / `useMutation`. |

### Supporting (no new dependencies needed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchaudio.functional.resample | >=2.11 | Resample 48kHz to 16kHz | Only needed if saving at 16kHz (recommended). Already available via torchaudio. |
| scipy.signal.resample_poly | >=1.14 | Alternative resampling | Lighter alternative to torchaudio for simple downsampling. Already in deps. |

**No new dependencies required.** Everything needed is already installed.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| soundfile for WAV write | wave (stdlib) | wave is lower-level, no float32 support, requires manual byte conversion. soundfile is already a dependency. |
| Sidecar JSON | SQLite database | JSON is portable, git-friendly, no migration complexity. SQLite adds operational burden for simple metadata. |
| 16kHz save | 48kHz native save | 48kHz preserves full bandwidth but 3x file size, requires resampling at training time. 16kHz is training-ready. |

## Architecture Patterns

### Recommended Project Structure
```
src/acoustic/
  recording/
    __init__.py
    config.py          # RecordingConfig (BaseSettings, ACOUSTIC_RECORDING_ prefix)
    recorder.py        # RecordingSession class -- captures from ring buffer to WAV
    metadata.py        # Sidecar JSON read/write, RecordingMetadata dataclass
    manager.py         # RecordingManager -- orchestrates sessions, file management
  api/
    recording_routes.py  # REST endpoints: start, stop, list, get, update metadata

web/src/
  components/
    recording/
      RecordingPanel.tsx     # Main recording controls + inline label form
      RecordingsList.tsx     # List of saved recordings with metadata
      MetadataEditor.tsx     # Inline metadata edit form
      AudioLevelMeter.tsx    # Real-time audio level visualization
  hooks/
    useRecordingSocket.ts    # WebSocket hook for live recording state
    useRecordings.ts         # TanStack Query hooks for recordings CRUD
```

### Pattern 1: Recording Session Lifecycle
**What:** A `RecordingSession` class that reads from the `AudioRingBuffer`, averages channels to mono, resamples to 16kHz, and streams to a temporary WAV file.
**When to use:** Every recording start/stop cycle.
**Example:**
```python
# Source: project patterns (AudioCapture ring buffer + soundfile)
import numpy as np
import soundfile as sf
from pathlib import Path

class RecordingSession:
    """Captures audio from ring buffer to a temporary WAV file."""

    def __init__(self, ring_buffer, output_path: Path, target_sr: int = 16000, source_sr: int = 48000):
        self._ring = ring_buffer
        self._path = output_path
        self._target_sr = target_sr
        self._source_sr = source_sr
        self._running = False
        self._samples_written = 0

    def start(self):
        """Open WAV file and begin capturing."""
        self._file = sf.SoundFile(
            str(self._path), mode='w',
            samplerate=self._target_sr, channels=1, format='WAV', subtype='FLOAT'
        )
        self._running = True

    def capture_chunk(self, chunk: np.ndarray):
        """Process one ring buffer chunk: mono downmix, resample, write."""
        if not self._running:
            return
        # 16-channel -> mono (D-12)
        mono = chunk.mean(axis=1)  # (chunk_samples,)
        # Resample 48kHz -> 16kHz (factor of 3)
        from scipy.signal import resample_poly
        resampled = resample_poly(mono, up=1, down=3)
        self._file.write(resampled)
        self._samples_written += len(resampled)

    def stop(self) -> float:
        """Close file, return duration in seconds."""
        self._running = False
        self._file.close()
        return self._samples_written / self._target_sr
```

### Pattern 2: Sidecar JSON Metadata
**What:** Each WAV file has a companion `.json` file with the same stem.
**When to use:** On label/save and metadata edits.
**Example:**
```python
# Sidecar JSON schema
{
    "label": "drone",              # Required: top-level label
    "sub_label": "Mavic 3",       # Optional: drone type
    "distance_m": 50.0,           # Optional: estimated distance
    "altitude_m": 30.0,           # Optional: estimated altitude
    "conditions": "light wind",   # Optional: weather/conditions
    "notes": "hovering, then approach",  # Optional: free text
    "recorded_at": "2026-04-02T14:30:00Z",  # Auto: ISO 8601 timestamp
    "duration_s": 12.5,           # Auto: recording duration
    "sample_rate": 16000,         # Auto: saved sample rate
    "channels": 1,                # Auto: always mono
    "original_sr": 48000,         # Auto: source sample rate
    "filename": "20260402_143000_abc123.wav"  # Auto: WAV filename
}
```

### Pattern 3: Record-First Flow with Temporary Storage
**What:** Recordings save immediately to a `_unlabeled/` directory. On label assignment, the file moves to `{label}/` and the sidecar JSON is created.
**When to use:** Implements D-01 (record-first, label later).
**Example:**
```
data/field/
  _unlabeled/          # Temporary: recordings awaiting label
    20260402_143000_abc123.wav
    20260402_143000_abc123.json   # Partial metadata (no label yet)
  drone/
    20260402_141500_def456.wav
    20260402_141500_def456.json
  background/
    20260402_140000_ghi789.wav
    20260402_140000_ghi789.json
  other/
    ...
```

### Pattern 4: REST API Design
**What:** Recording endpoints following existing `APIRouter` pattern.
**When to use:** All recording operations.
**Example:**
```python
router = APIRouter(prefix="/api/recordings", tags=["recordings"])

# POST /api/recordings/start     -> Start recording
# POST /api/recordings/stop      -> Stop recording
# GET  /api/recordings            -> List all recordings
# GET  /api/recordings/{id}       -> Get single recording metadata
# PATCH /api/recordings/{id}      -> Update metadata (label, sub_label, etc.)
# DELETE /api/recordings/{id}     -> Delete recording + metadata
# POST /api/recordings/{id}/label -> Assign label (moves from _unlabeled to {label}/)
```

### Anti-Patterns to Avoid
- **Blocking the audio thread:** Never do file I/O in the `AudioCapture._callback`. Recording must read from the ring buffer in a separate thread.
- **Recording directly from InputStream:** Don't create a second `sounddevice.InputStream`. The ring buffer is the single source of audio data -- tap into it, don't duplicate capture.
- **Storing metadata in filenames:** Filenames should be IDs only. All metadata goes in sidecar JSON. Don't encode label/conditions into the filename.
- **Modal dialogs for labeling:** D-04 explicitly requires inline forms, not modals.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WAV file writing | Raw byte manipulation | `soundfile.SoundFile` (streaming write mode) | Handles headers, float32 encoding, proper closing |
| Audio resampling | Custom interpolation | `scipy.signal.resample_poly(mono, 1, 3)` | Polyphase anti-aliasing filter, exact integer ratio 48k->16k |
| UUID generation | Custom ID scheme | `uuid.uuid4().hex[:8]` or full UUID | Collision-free, standard |
| ISO 8601 timestamps | Manual string formatting | `datetime.now(UTC).isoformat()` | Timezone-aware, standard format |
| JSON serialization | Custom file format | `json.dumps()` / Pydantic `.model_dump_json()` | Standard, portable, git-friendly |

**Key insight:** The recording pipeline is simple glue code. The audio capture (ring buffer), file writing (soundfile), and training integration (collect_wav_files) already exist. This phase connects them with a thin orchestration layer and a UI.

## Common Pitfalls

### Pitfall 1: Ring Buffer Contention
**What goes wrong:** Recording thread reads from the same ring buffer as the beamforming pipeline, causing dropped chunks for one or both.
**Why it happens:** `AudioRingBuffer.read()` consumes the chunk (advances read index). Two consumers can't both read.
**How to avoid:** The recording thread must NOT call `ring.read()`. Instead, it should tap the data at the write side. Options: (a) add a secondary ring buffer that the callback also writes to, (b) snapshot the most recent N chunks by reading from the buffer array directly (peek, not consume), (c) have the pipeline thread forward chunks to the recorder.
**Warning signs:** Beamforming map freezes or stutters when recording starts.

### Pitfall 2: File Not Properly Closed on Crash
**What goes wrong:** If the service crashes or recording is interrupted, the WAV file header may be invalid (wrong data size).
**Why it happens:** WAV headers contain the total data length, written at close time.
**How to avoid:** Use `soundfile.SoundFile` in streaming write mode -- it updates headers on flush/close. Add a finalizer or try/finally to ensure close on any exit path. Consider periodic flush.
**Warning signs:** WAV files that can't be opened after unexpected stops.

### Pitfall 3: Resampling Ratio Mismatch
**What goes wrong:** `resample_poly(mono, up=1, down=3)` assumes exactly 48000->16000 (ratio 3:1). If the source sample rate is different, audio is pitch-shifted.
**Why it happens:** Hardcoded ratio instead of computing from actual sample rates.
**How to avoid:** Compute ratio from `AcousticSettings.sample_rate` and `RecordingConfig.target_sr`. For 48000->16000, `resample_poly(x, 1, 3)` is exact. For non-integer ratios, use `torchaudio.functional.resample()`.
**Warning signs:** Recorded audio sounds higher or lower pitched than expected.

### Pitfall 4: Large File Accumulation
**What goes wrong:** Users forget to stop recordings, or max duration is set too high, filling disk.
**Why it happens:** 16kHz mono float32 = 64 KB/s = ~3.8 MB/min. A 5-min max recording is ~19 MB. Still manageable, but without limits, unbounded.
**How to avoid:** D-02 mandates configurable max duration with auto-stop. Default to 5 minutes (300s). Backend auto-stops and saves when limit is reached.
**Warning signs:** Disk usage growing unexpectedly, recordings directory becoming very large.

### Pitfall 5: Frontend State Desync
**What goes wrong:** UI shows "recording" but backend already stopped (or vice versa).
**Why it happens:** REST request/response latency, or backend auto-stop not communicated.
**How to avoid:** Use a WebSocket for recording state (recording/stopped/error, elapsed time, audio levels). The existing `/ws/training` pattern (poll at 2Hz with change detection) is a good model.
**Warning signs:** Record button stuck in wrong state, timer not updating.

## Code Examples

### Recording Config (Pydantic BaseSettings pattern)
```python
# Source: existing TrainingConfig pattern in src/acoustic/training/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class RecordingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACOUSTIC_RECORDING_")

    data_root: str = "data/field"          # Root for field recordings
    max_duration_s: float = 300.0          # 5 minutes default (D-02)
    target_sample_rate: int = 16000        # Match MelConfig.sample_rate
    top_labels: list[str] = ["drone", "background", "other"]  # D-06 fixed list
```

### Mono Downmix + Resample
```python
# Source: DroneAudioDataset already does multi->mono (line 99 of dataset.py)
import numpy as np
from scipy.signal import resample_poly

def process_chunk(chunk: np.ndarray) -> np.ndarray:
    """16-channel chunk (samples, 16) -> mono 16kHz float32."""
    mono = chunk.mean(axis=1)  # (samples,) float32
    # 48000/16000 = 3, exact integer ratio
    resampled = resample_poly(mono, up=1, down=3).astype(np.float32)
    return resampled
```

### RMS Audio Level Calculation
```python
# Source: standard DSP pattern
import numpy as np

def rms_db(chunk: np.ndarray) -> float:
    """Calculate RMS level in dB from mono audio chunk."""
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)
```

### Training Pipeline Compatibility
```python
# Source: collect_wav_files in src/acoustic/training/dataset.py
# This function scans {root}/{label}/ for WAV files.
# Field recordings in data/field/ with drone/, background/, other/ subdirs
# are directly compatible when data_root="data/field" is passed to TrainingConfig.

# Usage: training can point to field recordings:
# ACOUSTIC_TRAINING_DATA_ROOT=data/field python -m acoustic.main
# Or via the training start API: POST /api/training/start {"data_root": "data/field"}
```

### WebSocket Recording State Pattern
```python
# Source: existing /ws/training pattern in websocket.py
# Recording state WebSocket follows same poll + change-detection pattern
@router.websocket("/ws/recording")
async def ws_recording(websocket: WebSocket):
    await websocket.accept()
    manager = websocket.app.state.recording_manager
    last_state = None
    try:
        while True:
            state = manager.get_state()  # {status, elapsed_s, remaining_s, level_db}
            if state != last_state:
                await websocket.send_json(state)
                last_state = state
            await asyncio.sleep(0.1)  # 10 Hz for responsive level meter
    except (WebSocketDisconnect, RuntimeError):
        pass
```

## Discretion Recommendations

### Sample Rate: Save at 16kHz (RECOMMENDED)
**Rationale:** `MelConfig.sample_rate = 16000`. `ResearchPreprocessor.process()` can resample from arbitrary rates, but `mel_spectrogram_from_segment()` (used in `DroneAudioDataset`) does NOT resample -- it assumes input is already at 16kHz (line 100-103 of dataset.py: "assume sr matches mel_config.sample_rate"). Saving at 16kHz eliminates this mismatch and makes field recordings directly training-ready.
**Tradeoff:** Loses frequency content above 8kHz. Drone fundamental frequencies are 100-2000 Hz, so no meaningful signal is lost.

### File Naming: ISO timestamp + short random suffix
**Format:** `YYYYMMDD_HHMMSS_{6-char-hex}.wav` (e.g., `20260402_143000_a1b2c3.wav`)
**Rationale:** Timestamp gives human-readable ordering. Random suffix prevents collisions if two recordings start in the same second. Shorter than full UUID, still collision-safe for field use.

### Module Location: `src/acoustic/recording/`
**Rationale:** Recording is a distinct concern from audio capture (`audio/`) and training (`training/`). A new module keeps responsibilities clean. Follows the existing pattern of `classification/`, `evaluation/`, `tracking/` as separate packages.

### Directory Location: `data/field/` (separate from `audio-data/data/`)
**Rationale:** `audio-data/data/` contains the research dataset (490+ background files, 14 drone files, 150+ other files). Field recordings should not mix with this curated dataset. `data/field/` is a clean namespace. Training can point to either via `data_root` parameter. The training API already accepts `data_root` override in `TrainingStartRequest`.

### Temporary Storage: `data/field/_unlabeled/`
**Rationale:** Underscore prefix makes it sort first and visually distinct. `collect_wav_files()` only scans subdirectories matching `label_map` keys (`drone`, `background`, `other`), so `_unlabeled` is automatically excluded from training. Clean separation without special logic.

### Audio Level Meter: RMS from recording thread
**Rationale:** The recording thread already reads chunks from the ring buffer. Computing RMS on each chunk (one `np.mean` + `np.sqrt`) is negligible overhead. No need for a dedicated channel or separate computation.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flask sync recording endpoints | FastAPI async with background threads | Project decision (Phase 0) | Recording state via WebSocket, non-blocking |
| Full 16-channel WAV storage | Mono downmix at capture time | D-12 decision | 16x space savings, training-compatible |
| Database for metadata | Sidecar JSON per file | D-08 decision | No DB dependency, portable, git-friendly |

## Open Questions

1. **Ring buffer multi-consumer**
   - What we know: Current `AudioRingBuffer` is single-consumer (read advances index). Beamforming pipeline is the sole consumer.
   - What's unclear: Best pattern for recording to also consume audio without stealing from pipeline.
   - Recommendation: Add a `subscribe()` / observer pattern to the ring buffer, or have the pipeline thread forward chunks to registered consumers. Alternatively, recording can re-read from the raw buffer array using a separate read index (add `peek_latest(n)` method). The simplest approach: the recording thread maintains its own read pointer into the ring buffer's backing array, advancing independently of the pipeline's read pointer.

2. **Frontend navigation for recording UI**
   - What we know: Current frontend is a single-view dashboard (`App.tsx` renders `DashboardLayout` directly). No router.
   - What's unclear: Whether recording UI should be a new "page" (requiring a router) or integrated into the existing dashboard.
   - Recommendation: Add recording controls as a new panel within the existing dashboard grid. No router needed -- use conditional rendering or a tab/section toggle in the sidebar. This keeps the single-page architecture and avoids adding react-router as a new dependency.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio |
| Config file | `pyproject.toml` (existing) |
| Quick run command | `pytest tests/unit/test_recording*.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COL-01 | Start/stop recording via REST API | integration | `pytest tests/integration/test_recording_api.py -x -q` | Wave 0 |
| COL-01 | Recording saves WAV file to disk | unit | `pytest tests/unit/test_recorder.py -x -q` | Wave 0 |
| COL-02 | Metadata CRUD via REST API | integration | `pytest tests/integration/test_recording_api.py::test_update_metadata -x -q` | Wave 0 |
| COL-02 | Sidecar JSON write/read/update | unit | `pytest tests/unit/test_recording_metadata.py -x -q` | Wave 0 |
| COL-03 | Directory structure matches collect_wav_files() | unit | `pytest tests/unit/test_recording_integration.py -x -q` | Wave 0 |
| COL-03 | Label assignment moves file to correct directory | unit | `pytest tests/unit/test_recorder.py::test_label_moves_file -x -q` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_recording*.py tests/unit/test_recorder.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_recorder.py` -- RecordingSession unit tests (mono downmix, resampling, WAV write)
- [ ] `tests/unit/test_recording_metadata.py` -- Sidecar JSON CRUD
- [ ] `tests/integration/test_recording_api.py` -- REST endpoint tests

## Sources

### Primary (HIGH confidence)
- Project source code: `src/acoustic/training/dataset.py` -- `collect_wav_files()` contract (lines 18-60)
- Project source code: `src/acoustic/training/config.py` -- `TrainingConfig` with `data_root` and `label_map`
- Project source code: `src/acoustic/classification/config.py` -- `MelConfig` with `sample_rate=16000`
- Project source code: `src/acoustic/classification/preprocessing.py` -- `ResearchPreprocessor.process()` handles resampling; `mel_spectrogram_from_segment()` does NOT
- Project source code: `src/acoustic/audio/capture.py` -- `AudioRingBuffer` single-consumer pattern
- Project source code: `src/acoustic/api/training_routes.py` -- REST endpoint patterns
- Project source code: `src/acoustic/api/websocket.py` -- WebSocket poll + change-detection pattern

### Secondary (MEDIUM confidence)
- soundfile documentation: `sf.SoundFile` streaming write mode for WAV creation
- scipy.signal.resample_poly: integer ratio resampling (48000->16000 = 3:1)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new dependencies
- Architecture: HIGH -- follows established project patterns (APIRouter, BaseSettings, WebSocket, Panel components)
- Pitfalls: HIGH -- ring buffer contention is the main risk, well-understood with clear mitigation options
- Frontend: MEDIUM -- no router exists yet, recording panel integration needs design but pattern is clear

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable -- no external dependency changes expected)
