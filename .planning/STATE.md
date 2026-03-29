# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** Reliably detect and classify drones acoustically in real time, publishing target events over ZeroMQ so downstream systems can act on them.
**Current focus:** Phase 1 - Audio Capture, Beamforming, and Infrastructure

## Current Position

Phase: 1 of 5 (Audio Capture, Beamforming, and Infrastructure)
Plan: 0 of 3 in current phase
Status: Ready to plan
Last activity: 2026-03-29 -- Roadmap created

Progress: [..........] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- PyTorch over TensorFlow for CNN (research recommendation -- existing .h5 model cannot be reused, retraining required)
- Custom SRP-PHAT over Acoular (POC's 180-line implementation is simpler and sufficient for 4x4 array)
- Callback-based sounddevice.InputStream over blocking sd.rec() (irreversible architecture decision)

### Pending Todos

None yet.

### Blockers/Concerns

- Doppler speed estimation feasibility uncertain (UMA-16v2 aperture may be too small -- validate during Phase 3)
- UMA-16v2 channel mapping needs empirical verification (tap test) before beamforming work
- Callback-based capture not yet proven with UMA-16v2 (prototype early in Phase 1)

## Session Continuity

Last session: 2026-03-29
Stopped at: Roadmap created, ready to plan Phase 1
Resume file: None
