---
phase: 9
slug: evaluation-harness-and-api
status: draft
shadcn_initialized: false
preset: none
created: 2026-04-02
---

# Phase 9 — UI Design Contract

> Visual and interaction contract for Phase 9. This phase is **backend-only** — no new React components are built. The contract documents the established design system (for consistency) and prescribes API response copywriting, status labels, and data shapes that future UI phases will render.

---

## Design System

| Property | Value |
|----------|-------|
| Tool | none |
| Preset | not applicable |
| Component library | none (custom Tailwind components) |
| Icon library | material-symbols (outlined) |
| Font | Inter (400, 500) + JetBrains Mono (400) |

Source: `web/package.json`, `web/src/index.css`

---

## Spacing Scale

Declared values (must be multiples of 4):

| Token | Value | Usage |
|-------|-------|-------|
| xs | 4px | Icon gaps, inline padding |
| sm | 8px | Compact element spacing |
| lg | 16px | Panel content padding, header horizontal padding |
| xl | 24px | Section padding |
| 2xl | 32px | Layout gaps |
| 3xl | 48px | Major section breaks |

Component-specific overrides (not part of the canonical scale): `Panel.tsx` and `Header.tsx` use Tailwind `p-3` (12px) for interior padding. This is a legacy pattern carried from early components. New components should use 8px (`p-2`) or 16px (`p-4`) from the scale above. Existing `p-3` usage is tolerated but not prescribed.

---

## Typography

| Role | Size | Weight | Line Height | Font |
|------|------|--------|-------------|------|
| Body | 14px (`text-sm`) | 400 (regular) | 1.5 | Inter |
| Label | 12px (`text-xs`) | 500 (medium) | 1.5 | Inter |
| Heading | 14px (`text-sm`) | 500 (medium) | 1.5 | Inter |
| Display | 18px (`text-lg`) | 500 (medium) | 1.3 | JetBrains Mono |

Notes:
- Panel titles use `text-sm font-medium uppercase tracking-wider` in `text-hud-text-dim`
- Status values use `text-sm font-mono` in `text-hud-text`
- Labels use `text-xs text-hud-text-dim uppercase tracking-wider`
- The app title uses `font-mono font-medium text-hud-accent text-lg`
- All numeric/metric values should use JetBrains Mono for tabular alignment
- Two weights only: 400 (regular) for body text, 500 (medium) for all emphasis (labels, headings, display)

Source: `Panel.tsx`, `Header.tsx`, `Sidebar.tsx`

---

## Color

| Role | Value | Usage |
|------|-------|-------|
| Dominant (60%) | `#0a0e17` (`hud-bg`) | Page background, body |
| Secondary (30%) | `#111827` (`hud-panel`) | Cards, panels, header, sidebar |
| Border | `#1f2937` (`hud-border`) | Panel borders, dividers, stat row separators |
| Text primary | `#e5e7eb` (`hud-text`) | Body text, values, data |
| Text secondary | `#9ca3af` (`hud-text-dim`) | Labels, headings, status descriptions |
| Accent (10%) | `#3b82f6` (`hud-accent`) | App title, active nav items, primary action links |
| Success | `#22c55e` (`hud-success`) | Running status dot, healthy states, pass indicators |
| Warning | `#f59e0b` (`hud-warning`) | Scanning/in-progress states, threshold warnings |
| Destructive | `#ef4444` (`hud-danger`) | Stopped status, errors, failed states, cancel actions |

Accent reserved for: app title, primary CTA buttons, active/selected states, hyperlinks. Never for status indicators (use success/warning/danger).

Source: `web/src/index.css` `@theme` block

---

## Copywriting Contract

Phase 9 builds REST endpoints and a WebSocket. The following copy appears in **API responses** (JSON string values) that future UI phases will render. These are the canonical labels.

### Training Status Labels

| Status Value | Display Label | Description |
|--------------|---------------|-------------|
| `idle` | Idle | No training in progress |
| `running` | Training | Training is actively running |
| `completed` | Completed | Training finished successfully |
| `failed` | Failed | Training encountered an error |
| `cancelled` | Cancelled | Training was cancelled by user |

### Evaluation Response Copy

| Element | Copy |
|---------|------|
| Empty result (no files found) | `"No WAV files found in the specified directory. Check the data_dir path and ensure it contains labeled subdirectories (e.g., drone/, no drone/)."` |
| Evaluation success summary | `"Evaluated {total} files: {correct} correct, {incorrect} incorrect ({accuracy}% accuracy)"` |
| Model not found error | `"Model file not found at {path}. Verify the model_path parameter or train a model first."` |
| Invalid directory error | `"Directory not found: {path}. Provide a valid data_dir containing labeled subdirectories."` |
| No models available | `"No model files found. Train a model first using POST /api/training/start."` |

### Training Endpoint Copy

| Element | Copy |
|---------|------|
| Training already running | `"Training is already in progress. Cancel the current run before starting a new one."` |
| Training started | `"Training started with {epochs} max epochs, lr={lr}, batch_size={batch_size}"` |
| Training cancelled | `"Training cancelled. Partial results may be available."` |
| No training to cancel | `"No training is currently running."` |
| No data directory | `"Training data directory not found: {path}. Provide labeled audio subdirectories."` |

### WebSocket `/ws/training` Message Copy

| Element | Copy / Shape |
|---------|--------------|
| Connect (idle) | `{"status": "idle"}` |
| Connect (running) | `{"status": "running", "epoch": N, "total_epochs": N}` |
| Connect (completed) | `{"status": "completed", "results": {...}}` |
| Connect (failed) | `{"status": "failed", "error": "description"}` |
| Epoch update | `{"status": "running", "epoch": N, "total_epochs": N, "train_loss": float, "val_loss": float, "val_acc": float, "confusion_matrix": {"tp": N, "fp": N, "tn": N, "fn": N}}` |

### Destructive Actions

| Action | API Endpoint | Confirmation Approach |
|--------|-------------|----------------------|
| Cancel training | `POST /api/training/cancel` | No confirmation required (API-level). Future UI should show: "Cancel training? Partial progress will be lost." with Cancel/Confirm buttons. |

---

## API Response Data Shapes

These Pydantic response schemas define the visual contract for future UI rendering.

### `GET /api/models` Response

```json
{
  "models": [
    {
      "filename": "research_cnn_20260402.pt",
      "path": "/models/research_cnn_20260402.pt",
      "size_bytes": 1048576,
      "modified": "2026-04-02T10:30:00Z"
    }
  ]
}
```

Display: model filename in `font-mono text-sm`, file size human-readable (KB/MB), modification date relative ("2 hours ago").

### `POST /api/eval/run` Response

```json
{
  "summary": {
    "total": 556,
    "correct": 512,
    "incorrect": 44,
    "accuracy": 0.921,
    "precision": 0.935,
    "recall": 0.908,
    "f1": 0.921,
    "confusion_matrix": {"tp": 247, "fp": 17, "tn": 265, "fn": 27}
  },
  "distribution": {
    "drone": {"p_agg": {"p25": 0.72, "p50": 0.89, "p75": 0.95, "p95": 0.99}, "p_max": {}, "p_mean": {}},
    "background": {"p_agg": {"p25": 0.01, "p50": 0.05, "p75": 0.12, "p95": 0.28}, "p_max": {}, "p_mean": {}}
  },
  "per_file": [
    {"filename": "drone_001.wav", "true_label": "drone", "predicted_label": "drone", "p_agg": 0.92, "correct": true}
  ],
  "model_path": "/models/research_cnn.pt",
  "data_dir": "/data/test",
  "message": "Evaluated 556 files: 512 correct, 44 incorrect (92.1% accuracy)"
}
```

Display guidance for future UI:
- Accuracy as large `font-mono text-lg font-medium` number, colored `hud-success` (>= 0.85) / `hud-warning` (0.70-0.85) / `hud-danger` (< 0.70)
- Confusion matrix as 2x2 grid with `font-mono` values
- Per-file table with alternating row shading, incorrect rows highlighted with `hud-danger` background tint
- Distribution percentiles as horizontal bar charts or box plots

### `GET /api/training/progress` Response

```json
{
  "status": "running",
  "epoch": 5,
  "total_epochs": 50,
  "train_loss": 0.234,
  "val_loss": 0.312,
  "val_acc": 0.891,
  "confusion_matrix": {"tp": 45, "fp": 3, "tn": 42, "fn": 10},
  "best_val_loss": 0.298,
  "best_epoch": 3
}
```

Display guidance for future UI:
- Progress bar: epoch/total_epochs ratio, `hud-accent` fill
- Loss values in `font-mono text-sm`, 3 decimal places
- Accuracy percentage in `font-mono font-medium`

---

## Registry Safety

| Registry | Blocks Used | Safety Gate |
|----------|-------------|-------------|
| shadcn official | none | not applicable |

No component libraries or third-party registries are used. The project uses custom Tailwind CSS components with the `hud-*` design token system defined in `index.css`.

---

## Phase-Specific Notes

1. **No new React components in this phase.** Phase 9 builds backend REST endpoints, a WebSocket, and an evaluation harness. The copywriting contract and data shapes above define what future UI phases will render.

2. **Established patterns to follow for future UI integration:**
   - Panel component wraps content sections with title bar
   - StatRow component for label-value pairs
   - StatusDot component for boolean state indicators
   - WebSocket hooks (`useHeatmapSocket`, `useTargetSocket`) as pattern for `useTrainingSocket`
   - `@tanstack/react-query` for REST endpoint data fetching

3. **WebSocket `/ws/training` follows existing patterns:** connect, send initial state JSON, then push updates on change. No periodic heartbeats (per D-13).

---

## Checker Sign-Off

- [ ] Dimension 1 Copywriting: PASS
- [ ] Dimension 2 Visuals: PASS
- [ ] Dimension 3 Color: PASS
- [ ] Dimension 4 Typography: PASS
- [ ] Dimension 5 Spacing: PASS
- [ ] Dimension 6 Registry Safety: PASS

**Approval:** pending
