# Phase 1: Audio Capture, Beamforming, and Infrastructure - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a running Docker container that captures 16-channel audio from the UMA-16v2 mic array via callback-based streaming, produces a real-time SRP-PHAT beamforming spatial map, calculates peak azimuth/elevation, and exposes a health check — all configurable via environment variables. No web UI, no CNN, no ZeroMQ in this phase.

</domain>

<decisions>
## Implementation Decisions

### Audio Pipeline Architecture
- **D-01:** Callback-based `sounddevice.InputStream` in a dedicated thread writes chunks into a lock-free ring buffer. Beamforming consumer reads from the ring buffer asynchronously. This decouples capture from processing so neither blocks the other.
- **D-02:** Chunk size stays at POC's 150ms (7200 samples at 48kHz) — proven compromise between latency and low-frequency content for the 100-2000 Hz drone band.
- **D-03:** Ring buffer sized for ~2 seconds of audio (≈14 chunks) to absorb processing jitter without dropping frames.

### Development Without Hardware
- **D-04:** A simulated audio source provides synthetic 16-channel audio (sine waves with configurable direction-of-arrival) when no UMA-16v2 is detected. Enabled automatically on device absence or via environment variable (`AUDIO_SOURCE=simulated`).
- **D-05:** Test fixtures use short pre-recorded WAV snippets (from `audio-data/`) for deterministic beamforming validation.

### Beamforming Output Format
- **D-06:** Spatial map is a 2D NumPy array (azimuth × elevation grid). Resolution and angular range configurable, defaulting to 1° steps over ±90° azimuth, ±45° elevation.
- **D-07:** Peak detection returns azimuth and elevation in degrees (pan/tilt), matching the coordinate system downstream consumers expect.
- **D-08:** Adaptive noise threshold uses percentile-based calibration (per BF-04) — beamforming peak must exceed the Nth percentile of the map by a configurable margin to count as a detection.

### Docker and Device Access
- **D-09:** Base image: `python:3.11-slim` with ALSA libraries (`libasound2-dev`, `libportaudio2`) installed. No PulseAudio.
- **D-10:** USB passthrough via `--device /dev/snd` and `-v /dev/bus/usb:/dev/bus/usb`. Document `--privileged` as fallback if device mapping fails.
- **D-11:** Single Dockerfile, no multi-stage yet (frontend comes in Phase 2). Multi-stage build added when React UI is introduced.

### Claude's Discretion
All four gray areas were deferred to Claude's judgment. The following are open for downstream agents to decide:
- Exact ring buffer implementation (stdlib `queue`, custom numpy circular buffer, etc.)
- Project directory structure and module layout
- Logging framework choice and verbosity levels
- Test structure (unit vs integration split)
- FastAPI app skeleton scope in Phase 1 (minimal health endpoint only, or full app structure)

### Folded Todos
None.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### POC Reference Implementation
- `POC-code/PT520/PTZ/radar_gui_all_mics_fast_drone.py` — SRP-PHAT beamforming, GCC-PHAT, mic position geometry (4x4 URA, 42mm spacing), frequency band filtering. Port the algorithm, not the blocking capture pattern.

### Hardware Documentation
- `Acoustic Overview.docx` — UMA-16v2 array overview
- `Acoustic Technical Info.docx` — Technical specifications for the mic array

### Training Data Format Reference
- `audio-data/data/background/rrr_20260211_111941_take0001_background_d10m.json` — Example metadata schema for existing recordings (sample rate, device info, WAV paths). Relevant for D-05 test fixture design.

### Project Configuration
- `CLAUDE.md` §Technology Stack — Pinned versions and rationale for all dependencies

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- POC `build_mic_positions()` — UMA-16v2 channel-to-position mapping (verified geometry). Port directly.
- POC SRP-PHAT + GCC-PHAT — ~180 lines of NumPy. Core algorithm to extract and refactor into clean modules.
- Existing `audio-data/data/` — background, drone, and other categories with JSON metadata + WAV files for test fixtures.

### Established Patterns
- POC uses `sounddevice` with ALSA device names (`hw:X,0`) — same approach for Docker.
- POC constants: FS=48000, NUM_CHANNELS=16, CHUNK_SECONDS=0.15, C=343.0, SPACING=0.042m, DRONE_FMIN=100Hz, DRONE_FMAX=2000Hz.
- Existing recordings use per-channel WAV files with JSON sidecar metadata.

### Integration Points
- No existing project structure — this phase creates the foundation.
- FastAPI app (minimal in Phase 1) will be extended in Phase 2 for REST/WebSocket endpoints.
- Ring buffer interface must be consumable by future CNN inference (Phase 3) and recording (Phase 4) modules.

</code_context>

<specifics>
## Specific Ideas

- User opened an existing recording metadata JSON — the `devices` array shows real UMA-16v2 device info (`"UMA16v2: USB Audio (hw:2,0)"`, ALSA hostapi, 16 input channels). Use this as reference for device detection logic.
- POC records per-channel WAVs, but this service should capture all 16 channels in a single interleaved stream for beamforming efficiency.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-audio-capture-beamforming-and-infrastructure*
*Context gathered: 2026-03-29*
