# Phase 19: Functional Beamforming Visualization - Research

**Researched:** 2026-04-10
**Domain:** DSP post-processing (functional beamforming), FastAPI settings API, frontend rendering
**Confidence:** HIGH

## Summary

Phase 19 is a focused, well-scoped phase that applies a standard DSP post-processing technique (functional beamforming via power-map exponent) to the existing SRP-PHAT pipeline output, adds a runtime-configurable nu parameter, and simplifies the frontend renderer. All decisions are locked in CONTEXT.md. The codebase is well-structured for this change: the pipeline already stores `latest_map` as float32, the WebSocket streams it as binary, and the config system uses pydantic-settings with env var prefix. The implementation is approximately 4 touch points: config.py (add field), pipeline.py (2-line post-process + normalization), routes.py (PATCH endpoint), HeatmapCanvas.tsx (remove v*v).

**Primary recommendation:** Implement as a single plan with 3-4 tasks covering backend config, pipeline transform, API endpoint, and frontend simplification. This is a small, surgical phase.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use power-map exponent approach: raise the normalized SRP-PHAT power map element-wise to the nu-th power (`map_norm ** nu`). This is a post-processing step after `srp_phat_2d()` -- no engine restructuring needed.
- **D-02:** The transform is applied in the backend pipeline (pipeline.py), not the frontend. Backend owns all DSP; frontend stays a dumb color-mapper.
- **D-03:** Add `bf_nu: float = 100.0` to `AcousticSettings` (env var `ACOUSTIC_BF_NU`). Default nu=100 per VIZ-02.
- **D-04:** Add a PATCH `/api/settings` endpoint (or extend existing) to update nu at runtime without restart. Pipeline reads `settings.bf_nu` each chunk iteration. Matches the existing config pattern used by `bf_freq_min`, `bf_holdoff_seconds`, etc.
- **D-05:** The pipeline already passes `bf_freq_min=500` / `bf_freq_max=4000` to `srp_phat_2d()` correctly (pipeline.py:222-223). VIZ-01 is already satisfied at the backend level.
- **D-06:** Update `srp_phat_2d()` default arguments from `fmin=100.0, fmax=2000.0` to `fmin=500.0, fmax=4000.0` for consistency, since the old defaults are never used but could cause confusion.
- **D-07:** Backend normalizes the functional beamforming output to [0,1] before sending via WebSocket. The nu-power transform already produces the desired contrast.
- **D-08:** Remove the frontend `v * v` (squared normalization) in `HeatmapCanvas.tsx`. Since the backend now owns all contrast/normalization through the functional beamforming transform, the frontend simply maps [0,1] values to colormap indices. One code path, no conditional logic.

### Claude's Discretion
- Exact normalization strategy (min-max vs. max-only) for the functional beamforming output
- Whether to clamp very small values to zero for cleaner rendering
- Implementation details of the PATCH settings endpoint

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VIZ-01 | Heatmap displays beamforming output using corrected 500-4000 Hz frequency band | D-05 confirms pipeline already uses 500-4000 Hz. D-06 updates default args for consistency. No functional change needed -- already satisfied. |
| VIZ-02 | Functional Beamforming with configurable nu parameter produces sidelobe-suppressed clean maps for display | D-01 through D-04 define the power-map exponent approach, config field, and runtime API. Core implementation work of this phase. |
</phase_requirements>

## Standard Stack

No new dependencies. This phase uses only existing libraries already in the project:

| Library | Version | Purpose | Already Installed |
|---------|---------|---------|-------------------|
| NumPy | >=1.26 | Power-map exponent (`map ** nu`) | Yes |
| FastAPI | >=0.135 | PATCH endpoint for settings | Yes |
| Pydantic Settings | >=2.0 | `bf_nu` config field | Yes |

**No installation needed.**

## Architecture Patterns

### Pattern 1: Functional Beamforming Post-Processing

**What:** Element-wise power-map exponent applied after SRP-PHAT, before storage/streaming.

**Math:** Given raw SRP-PHAT output `P(az, el)`:
1. Normalize: `P_norm = P / max(P)` (max-only normalization to [0, 1])
2. Apply functional beamforming: `P_fb = P_norm ** nu`
3. Store as `latest_map`

**Why max-only (not min-max):** The SRP-PHAT output is always non-negative (sum of GCC-PHAT correlation values). Min-max would boost the noise floor by shifting the minimum to 0 -- but the minimum is already near 0 for a well-formed power map. Max-only division preserves the natural dynamic range and is simpler. [VERIFIED: srp_phat.py returns correlation sums which are non-negative]

**Why this works for sidelobe suppression:** When nu=100, a sidelobe at 0.7 relative power becomes 0.7^100 ~ 3e-16 (effectively zero), while the main peak at 1.0 stays at 1.0. This is the standard functional beamforming technique from Dougherty (2014). [ASSUMED]

**Clamping recommendation:** Clamp values below 1e-6 to 0.0 after the nu-power transform. At nu=100, anything below ~0.87 normalized power becomes numerically negligible. Clamping avoids floating-point noise in the rendered heatmap. [ASSUMED]

**Where in pipeline.py (line 224-225):**
```python
# Current:
self.latest_map = srp_map.astype(np.float32)

# New (D-01, D-07):
max_val = srp_map.max()
if max_val > 0:
    normalized = srp_map / max_val
    nu = self._settings.bf_nu
    fb_map = normalized ** nu
    fb_map[fb_map < 1e-6] = 0.0
    self.latest_map = fb_map.astype(np.float32)
else:
    self.latest_map = srp_map.astype(np.float32)
```

### Pattern 2: Runtime Settings Update via PATCH

**What:** A PATCH `/api/settings` endpoint that accepts a partial JSON body to update mutable AcousticSettings fields at runtime.

**Existing pattern:** The pipeline reads `self._settings.bf_nu` each iteration (no caching), so mutating the settings object takes effect on the next chunk. This is the same pattern used for `bf_freq_min`, `bf_holdoff_seconds`, etc. as noted in CONTEXT.md.

**Implementation approach:**
```python
from pydantic import BaseModel

class SettingsUpdate(BaseModel):
    bf_nu: float | None = None
    # Extensible: add other mutable fields as needed

@router.patch("/settings")
async def update_settings(request: Request, update: SettingsUpdate) -> dict:
    settings = request.app.state.settings
    changed = {}
    if update.bf_nu is not None:
        if update.bf_nu < 1.0:
            raise HTTPException(400, "bf_nu must be >= 1.0")
        settings.bf_nu = update.bf_nu
        changed["bf_nu"] = update.bf_nu
    return {"updated": changed}
```

**Note on Pydantic Settings mutability:** `BaseSettings` instances are mutable by default in pydantic-settings v2 (fields can be assigned directly). No `model_config` change needed. [VERIFIED: config.py uses pydantic_settings.BaseSettings with no frozen=True]

### Pattern 3: Frontend Simplification

**What:** Remove the `v * v` squaring in HeatmapCanvas.tsx, replace with direct LUT mapping.

**Current code (line 57-59):**
```typescript
const v = floats[i]
const normalized = v * v
const lutIdx = Math.round(normalized * 255) * 3
```

**New code:**
```typescript
const v = floats[i]
const lutIdx = Math.round(v * 255) * 3
```

This is a one-line change. The backend now owns all contrast enhancement through functional beamforming. [VERIFIED: HeatmapCanvas.tsx lines 56-59]

### Anti-Patterns to Avoid
- **Applying functional beamforming in the frontend:** Violates D-02. The frontend is a dumb color-mapper.
- **Using min-max normalization:** Would amplify noise floor when all values are near-zero (no source present).
- **Caching bf_nu in pipeline __init__:** Would break runtime updates via PATCH. Read from settings each iteration.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Power-map exponent | Custom loop | `np.power()` or `**` operator | NumPy vectorized, handles edge cases |
| Settings validation | Manual range checks | Pydantic Field validators | Consistent with project pattern |

## Common Pitfalls

### Pitfall 1: Division by Zero in Normalization
**What goes wrong:** When the beamforming map is all zeros (no signal, gate blocked), dividing by max gives NaN.
**Why it happens:** Pipeline returns zero map when gate blocks beamforming or device is disconnected.
**How to avoid:** Guard with `if max_val > 0` before normalization. Return zero map unchanged.
**Warning signs:** NaN values in WebSocket frames, blank/white heatmap.

### Pitfall 2: Numerical Underflow with Large Nu
**What goes wrong:** `0.5 ** 100 = 7.9e-31` -- extremely small values that are meaningless but non-zero.
**Why it happens:** IEEE 754 float64 has range down to ~5e-324, so these don't underflow to zero.
**How to avoid:** Clamp values below a threshold (1e-6) to zero after the power transform.
**Warning signs:** Heatmap shows faint noise speckle instead of clean black background.

### Pitfall 3: Nu Must Be >= 1
**What goes wrong:** nu < 1 would reverse the effect (broaden peaks instead of sharpening).
**Why it happens:** `x ** 0.5` is a square root, which compresses contrast.
**How to avoid:** Validate bf_nu >= 1.0 in the PATCH endpoint and/or Pydantic validator.
**Warning signs:** Heatmap becomes more smeared, not less.

### Pitfall 4: Float32 Precision Loss at High Nu
**What goes wrong:** Computing `map ** 100` in float32 loses precision compared to float64.
**Why it happens:** SRP-PHAT returns float64, but latest_map stores float32.
**How to avoid:** Do the normalization and power transform in float64 (which srp_map already is), then cast to float32 only at the final assignment.
**Warning signs:** Subtle differences in peak sharpness.

### Pitfall 5: Breaking Existing Peak Detection
**What goes wrong:** Functional beamforming is applied BEFORE peak detection, corrupting the MCRA noise floor.
**Why it happens:** Incorrect insertion point in process_chunk.
**How to avoid:** Insert functional beamforming AFTER peak detection (line 256), only for the `latest_map` assignment used for visualization. Peak detection must operate on the raw SRP-PHAT output.
**Warning signs:** Peaks not detected, or spurious peaks.

**CRITICAL INSIGHT (Pitfall 5):** The CONTEXT.md says to insert "after line 224, before `self.latest_map` assignment" -- but the current code has peak detection, MCRA, and interpolation AFTER line 224. The functional beamforming transform must be applied ONLY to the visualization map (`latest_map`), NOT to the map used for peak detection. The correct insertion point is at line 225 (the `self.latest_map = srp_map.astype(np.float32)` assignment), transforming only the value stored in `latest_map` while leaving `srp_map` untouched for downstream peak detection. Looking at the actual code flow:

```
line 215-224: srp_map = srp_phat_2d(...)
line 225:     self.latest_map = srp_map.astype(np.float32)  # <-- VISUALIZATION
line 228:     noise_floor = self._mcra.update(srp_map)       # <-- DETECTION (uses raw)
line 231-253: peaks = detect_multi_peak(srp_map, ...)        # <-- DETECTION (uses raw)
line 255:     self.latest_peaks = peaks
```

The current line 225 sets `latest_map` BEFORE peak detection. So replacing line 225 with the functional beamforming version is correct -- it transforms only the visualization output while `srp_map` (the local variable) remains raw for MCRA and peak detection. This is safe.

## Code Examples

### Complete Pipeline Transform (Python)
```python
# In pipeline.py process_chunk(), replace line 225:
# OLD: self.latest_map = srp_map.astype(np.float32)

# VIZ-02: Functional beamforming for sidelobe suppression
max_val = srp_map.max()
if max_val > 0:
    fb_map = (srp_map / max_val) ** self._settings.bf_nu
    fb_map[fb_map < 1e-6] = 0.0
    self.latest_map = fb_map.astype(np.float32)
else:
    self.latest_map = np.zeros_like(srp_map, dtype=np.float32)
```

### Config Field Addition (Python)
```python
# In config.py AcousticSettings, after bf_peak_threshold:
# VIZ-02: Functional beamforming exponent
bf_nu: float = 100.0  # Functional beamforming nu parameter. Higher = sharper peaks.
```

### PATCH Settings Endpoint (Python)
```python
# In routes.py
from fastapi import HTTPException
from pydantic import BaseModel, Field

class SettingsUpdate(BaseModel):
    bf_nu: float | None = Field(None, ge=1.0, le=1000.0)

@router.patch("/settings")
async def update_settings(request: Request, body: SettingsUpdate) -> dict:
    settings = request.app.state.settings
    updated = {}
    for field_name, value in body.model_dump(exclude_none=True).items():
        setattr(settings, field_name, value)
        updated[field_name] = value
    return {"updated": updated}
```

### Frontend Change (TypeScript)
```typescript
// In HeatmapCanvas.tsx, replace lines 57-59:
// OLD:
//   const normalized = v * v
//   const lutIdx = Math.round(normalized * 255) * 3
// NEW:
const lutIdx = Math.round(v * 255) * 3
```

### srp_phat_2d Default Args Update (Python)
```python
# In srp_phat.py, update function signature:
def srp_phat_2d(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    c: float,
    az_grid_deg: np.ndarray,
    el_grid_deg: np.ndarray,
    fmin: float = 500.0,   # Was 100.0 -- updated per D-06
    fmax: float = 4000.0,  # Was 2000.0 -- updated per D-06
) -> np.ndarray:
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio |
| Config file | `pytest.ini` or `pyproject.toml` |
| Quick run command | `pytest tests/integration/test_pipeline_beamforming.py -x` |
| Full suite command | `pytest tests/ -x --timeout=60` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VIZ-01 | Pipeline uses 500-4000 Hz band | integration | `pytest tests/integration/test_pipeline_beamforming.py -x` | Yes (existing tests verify pipeline runs with current settings) |
| VIZ-02 | Functional beamforming suppresses sidelobes | unit | `pytest tests/unit/test_functional_beamforming.py -x` | No -- Wave 0 |
| VIZ-02 | bf_nu configurable at runtime via PATCH | integration | `pytest tests/integration/test_settings_api.py -x` | No -- Wave 0 |
| VIZ-02 | Frontend receives [0,1] normalized map | integration | Existing heatmap WebSocket tests (if any), else manual | N/A |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_functional_beamforming.py tests/integration/test_pipeline_beamforming.py -x`
- **Per wave merge:** `pytest tests/ -x --timeout=60`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_functional_beamforming.py` -- covers VIZ-02 (power-map exponent math, edge cases)
- [ ] `tests/integration/test_settings_api.py` -- covers PATCH /api/settings endpoint

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Functional beamforming via power-map exponent is the standard technique from Dougherty (2014) | Architecture Patterns | LOW -- the math is straightforward regardless of attribution |
| A2 | Clamping threshold of 1e-6 is appropriate | Architecture Patterns | LOW -- easily tunable, conservative value |

## Open Questions

None. All decisions are locked. Implementation is straightforward.

## Sources

### Primary (HIGH confidence)
- `src/acoustic/pipeline.py` lines 200-258 -- verified current pipeline flow, insertion point, and that srp_map variable is separate from latest_map
- `src/acoustic/config.py` -- verified AcousticSettings structure, pydantic-settings v2, no frozen config
- `src/acoustic/beamforming/srp_phat.py` -- verified srp_phat_2d returns raw float64 correlation sums, current default args
- `src/acoustic/api/routes.py` -- verified no existing PATCH endpoint for settings
- `src/acoustic/api/websocket.py` lines 72-79 -- verified heatmap WebSocket sends latest_map.T.astype(float32).tobytes()
- `web/src/components/heatmap/HeatmapCanvas.tsx` lines 55-65 -- verified v*v squaring at line 58
- `19-CONTEXT.md` -- all locked decisions

### Secondary (MEDIUM confidence)
- None needed -- all information came from codebase inspection

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing
- Architecture: HIGH -- verified all touch points in codebase, math is simple
- Pitfalls: HIGH -- identified critical insertion point concern (Pitfall 5) and verified it's safe

**Research date:** 2026-04-10
**Valid until:** 2026-05-10 (stable -- DSP math doesn't change)
