# Phase 19: Functional Beamforming Visualization - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

The heatmap displays a clean, sidelobe-suppressed beamforming map using functional beamforming (power-map exponent with configurable nu parameter) and the corrected 500-4000 Hz frequency band. This phase adds the functional beamforming transform to the existing pipeline and adjusts the frontend rendering to match.

</domain>

<decisions>
## Implementation Decisions

### Functional Beamforming Algorithm
- **D-01:** Use power-map exponent approach: raise the normalized SRP-PHAT power map element-wise to the nu-th power (`map_norm ** nu`). This is a post-processing step after `srp_phat_2d()` — no engine restructuring needed.
- **D-02:** The transform is applied in the backend pipeline (pipeline.py), not the frontend. Backend owns all DSP; frontend stays a dumb color-mapper.

### Nu Parameter Control
- **D-03:** Add `bf_nu: float = 100.0` to `AcousticSettings` (env var `ACOUSTIC_BF_NU`). Default nu=100 per VIZ-02.
- **D-04:** Add a PATCH `/api/settings` endpoint (or extend existing) to update nu at runtime without restart. Pipeline reads `settings.bf_nu` each chunk iteration. Matches the existing config pattern used by `bf_freq_min`, `bf_holdoff_seconds`, etc.

### Frequency Band Wiring
- **D-05:** The pipeline already passes `bf_freq_min=500` / `bf_freq_max=4000` to `srp_phat_2d()` correctly (pipeline.py:222-223). VIZ-01 is already satisfied at the backend level.
- **D-06:** Update `srp_phat_2d()` default arguments from `fmin=100.0, fmax=2000.0` to `fmin=500.0, fmax=4000.0` for consistency, since the old defaults are never used but could cause confusion.

### Heatmap Visual Rendering
- **D-07:** Backend normalizes the functional beamforming output to [0,1] before sending via WebSocket. The nu-power transform already produces the desired contrast.
- **D-08:** Remove the frontend `v * v` (squared normalization) in `HeatmapCanvas.tsx`. Since the backend now owns all contrast/normalization through the functional beamforming transform, the frontend simply maps [0,1] values to colormap indices. One code path, no conditional logic.

### Claude's Discretion
- Exact normalization strategy (min-max vs. max-only) for the functional beamforming output
- Whether to clamp very small values to zero for cleaner rendering
- Implementation details of the PATCH settings endpoint

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Beamforming Engine
- `src/acoustic/beamforming/srp_phat.py` — SRP-PHAT engine producing spatial power maps. Update default fmin/fmax args.
- `src/acoustic/beamforming/bandpass.py` — Bandpass pre-filter already at 500-4000 Hz (BF-11)
- `src/acoustic/beamforming/__init__.py` — Public exports for beamforming package

### Pipeline
- `src/acoustic/pipeline.py` — `BeamformingPipeline.process_chunk()` where functional beamforming transform must be inserted (after line 224, before `self.latest_map` assignment)
- `src/acoustic/config.py` — `AcousticSettings` where `bf_nu` must be added

### WebSocket & Frontend
- `src/acoustic/api/websocket.py` — `/ws/heatmap` endpoint that streams binary float32 frames
- `web/src/components/heatmap/HeatmapCanvas.tsx` — Frontend renderer where `v * v` squaring must be removed
- `web/src/hooks/useHeatmapSocket.ts` — Heatmap WebSocket hook

### API
- `src/acoustic/api/routes.py` — REST API routes where PATCH settings endpoint should be added

### Requirements
- `.planning/REQUIREMENTS.md` §v4.0 — VIZ-01 (500-4000 Hz band heatmap), VIZ-02 (functional beamforming with nu)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `srp_phat_2d()` — Returns `(n_az, n_el)` float64 power map. Functional beamforming is a 2-line post-process on this output.
- `AcousticSettings` — Pydantic settings with `ACOUSTIC_` env prefix. Adding `bf_nu` follows the exact same pattern as `bf_freq_min`.
- `HeatmapCanvas` — Already renders binary float32 frames with jet colormap LUT. Only change is removing the `v * v` line.

### Established Patterns
- Pipeline processes chunks in a background thread, stores result in `self.latest_map` as float32
- WebSocket endpoint reads `pipeline.latest_map` and sends as binary
- Config values are read from `self._settings` each iteration (no caching), so runtime updates via API take effect immediately

### Integration Points
- `pipeline.py:224-225` — Insert functional beamforming between `srp_phat_2d()` return and `self.latest_map` assignment
- `config.py:29-31` — Add `bf_nu` alongside existing `bf_freq_min` / `bf_freq_max`
- `routes.py` — Add or extend PATCH endpoint for runtime config updates
- `HeatmapCanvas.tsx:57-59` — Remove squared normalization

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches within the decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 19-functional-beamforming-visualization*
*Context gathered: 2026-04-10*
