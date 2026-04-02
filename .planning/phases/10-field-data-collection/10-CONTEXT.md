# Phase 10: Field Data Collection - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a web UI for recording labeled audio clips from the live UMA-16 microphone array, with metadata capture and auto-organized directory structure that the training pipeline can directly consume. This phase delivers: recording start/stop controls, inline labeling form, metadata editing, configurable max duration, live recording feedback (timer + levels), and auto-organized `{label}/` directory output. This phase does NOT include recording playback through the detection pipeline (Phase 4 scope), ensemble support (Phase 11), or training UI (Phase 9).

</domain>

<decisions>
## Implementation Decisions

### Recording Session Flow
- **D-01:** Record-first, label later. User hits record immediately without pre-selecting a label. Recording saves to a temporary/unlabeled location until labeled.
- **D-02:** Configurable max recording duration with auto-stop. Default TBD by implementation (e.g., 5 minutes). Prevents accidentally huge files.
- **D-03:** Live recording feedback shows elapsed time, remaining time (when max is set), and a simple audio level meter so the user can confirm capture is working.
- **D-04:** After stopping, an inline label form appears in the UI (not a modal). User selects top-level label (required), optional sub-label and metadata, then saves. Recording is filed into the correct directory on save.

### Label Taxonomy & Directory Layout
- **D-05:** Hierarchical labeling: fixed top-level labels (drone, background, other) determine directory placement. Sub-labels within each top-level (e.g., Mavic, Matrice, 5-inch under "drone") are stored in metadata only.
- **D-06:** Top-level labels are a fixed preset list (drone, background, other). No user-defined top-level labels. This ensures compatibility with `collect_wav_files()` and the training pipeline's `label_map`.
- **D-07:** Sub-labels are user-definable (free text or pick from suggestions). Stored in sidecar JSON metadata, not directory structure.

### Claude's Discretion: Directory Location
- Whether field recordings go into a separate `data/recordings/{label}/` tree or into the existing `audio-data/data/{label}/` tree. Choose based on how `collect_wav_files()` and `TrainingConfig.data_root` work -- the key requirement is that training can point to field recordings as a data source.

### Metadata Schema
- **D-08:** Sidecar JSON file per recording (e.g., `rec_001.json` alongside `rec_001.wav`). Simple, portable, no database dependency.
- **D-09:** Only top-level label is required. All other metadata fields are optional.
- **D-10:** Metadata fields: top-level label (required), sub-label (e.g., drone type), distance estimate, altitude estimate, weather/conditions, free-text notes, recording timestamp, duration.
- **D-11:** Metadata is editable after recording -- user can update any field from the recordings list or detail view.

### Audio Capture Format
- **D-12:** Mono downmix -- average all 16 channels to a single channel for recording. Matches research data format and saves ~16x disk space vs full 16-channel.
- **D-13:** Claude's Discretion on sample rate: choose between saving at 16kHz (matches MelConfig SR=16000, training-ready) or 48kHz native (preserves full frequency content, ResearchPreprocessor resamples on-the-fly). Pick based on what minimizes friction in the training pipeline.

### Claude's Discretion
- Recording file naming convention (timestamps, sequential IDs, UUIDs)
- Where to place the recording module code (e.g., `src/acoustic/recording/` or extend existing audio module)
- How the inline label form integrates with existing frontend component patterns
- Audio level meter implementation (RMS from ring buffer vs dedicated channel)
- How temporary/unlabeled recordings are managed before the user labels them

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training Pipeline Integration
- `src/acoustic/training/dataset.py` -- `collect_wav_files()` function scans `{root}/{label}/` subdirectories for WAV files. Phase 10 directory output MUST be compatible with this scanner.
- `src/acoustic/training/config.py` -- `TrainingConfig` with `data_root` parameter. Field recordings directory must be usable as a `data_root` value.

### Audio Capture
- `src/acoustic/audio/capture.py` -- `AudioCapture` class with callback-based ring buffer. Source of raw 16-channel audio data for recording.
- `src/acoustic/audio/monitor.py` -- Device monitoring. Recording must handle device disconnect gracefully.

### Preprocessing (for sample rate decision)
- `src/acoustic/classification/config.py` -- `MelConfig` with SR=16000. Defines target sample rate for training.
- `src/acoustic/classification/preprocessing.py` -- `ResearchPreprocessor` with torchaudio pipeline. Check if it handles resampling from arbitrary input rates.

### Existing Frontend Patterns
- `web/src/components/layout/DashboardLayout.tsx` -- Main layout structure. Recording UI integrates here.
- `web/src/components/layout/Panel.tsx` -- Panel component pattern for UI sections.
- `web/src/components/layout/Sidebar.tsx` -- Navigation pattern.
- `web/src/hooks/useDeviceStatus.ts` -- Device status hook. Recording should respect device state.

### Existing Data
- `audio-data/data/` -- Existing training data with `background/`, `drone/`, `other/` subdirectories. Reference for directory convention.
- `audio-data/data/index.jsonl` -- Existing metadata index. Phase 10 uses sidecar JSON instead, but this shows prior metadata patterns.

### Prior Phase Context
- `.planning/phases/08-pytorch-training-pipeline/08-CONTEXT.md` -- D-04 (directory convention), D-01 (segment extraction from any-length WAVs).
- `.planning/phases/09-evaluation-harness-and-api/09-CONTEXT.md` -- D-16/D-17 (test/training data directory conventions).

### Requirements
- `.planning/ROADMAP.md` -- COL-01, COL-02, COL-03 (referenced but not yet detailed in REQUIREMENTS.md -- define acceptance criteria from roadmap success criteria).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AudioCapture` in `audio/capture.py` -- Callback-based 16-channel ring buffer. Source for recording raw audio.
- `collect_wav_files()` in `training/dataset.py` -- Directory scanner for labeled WAVs. Defines the output contract.
- `Panel`, `DashboardLayout`, `Sidebar` components -- Established frontend layout patterns.
- `useDeviceStatus` hook -- Device state awareness for recording controls.
- `HeatmapCanvas` -- Reference for real-time data rendering (audio level meter pattern).

### Established Patterns
- Config via Pydantic `BaseSettings` with `ACOUSTIC_` env prefix
- FastAPI `APIRouter` with route prefixes for REST endpoints
- WebSocket pattern: accept, send initial state, then poll/push loop
- React components with TanStack Query for server state
- Tailwind CSS v4 styling consistent with sky-fort-dashboard

### Integration Points
- Recording reads from `AudioCapture` ring buffer (same source as beamforming pipeline)
- Recording output directory must be usable as `TrainingConfig.data_root`
- Recording REST endpoints alongside existing `/api/map`, `/api/targets`, `/api/training/`, `/api/eval/`
- Frontend recording UI integrates into existing dashboard layout (new panel or sidebar section)

</code_context>

<specifics>
## Specific Ideas

- Record-first flow prioritizes speed of capture over organization -- important for field conditions where a drone appears unexpectedly
- Inline label form (not modal) keeps the user in context and allows quick labeling workflow
- Hierarchical labels (fixed top-level + free sub-label) give structure for training while preserving detail for analysis
- Sidecar JSON matches the portable, git-friendly approach -- no database to manage in field deployment
- Mono downmix is the pragmatic choice since all training and inference operates on mono audio

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 10-field-data-collection*
*Context gathered: 2026-04-02*
