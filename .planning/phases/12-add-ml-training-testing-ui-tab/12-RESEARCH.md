# Phase 12: Add ML Training & Testing UI Tab - Research

**Researched:** 2026-04-02
**Domain:** React frontend -- training/evaluation UI, WebSocket real-time updates, charting
**Confidence:** HIGH

## Summary

Phase 12 is a frontend-only phase that adds a third sidebar tab (TRAINING) to surface existing backend APIs for CNN training control, evaluation results, and model management. All backend endpoints already exist (Phase 8/9): REST for training start/cancel/progress, evaluation run, and model listing, plus a WebSocket at `/ws/training` for live epoch-by-epoch progress updates.

The implementation follows established patterns: `useState` tab switching (identical to SYSTEM/RECORDINGS), TanStack Query for REST mutations/queries (identical to `useRecordings.ts`), and a WebSocket hook with reconnection (identical to `useRecordingSocket.ts`). The only new dependency is Recharts for the loss chart. The accordion UI is custom (no library needed -- a simple collapsible div with state toggle matches the project's minimal-dependency approach).

**Primary recommendation:** Follow the existing Recording panel pattern exactly. New hooks (`useTraining`, `useTrainingSocket`, `useEvaluation`, `useModels`) mirror `useRecordings` and `useRecordingSocket`. Recharts `LineChart` for loss curves. Custom accordion via `useState<string | null>` for open section tracking.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** New third sidebar tab "TRAINING" alongside existing SYSTEM and RECORDINGS tabs. Uses the same local `useState` tab switching pattern (Phase 10 decision).
- **D-02:** Content organized as collapsible accordion sections within the tab. Sections: Train, Evaluate, Models. Only one or two open at a time to save vertical space.
- **D-03:** "Start Training" button with an expandable "Advanced" section showing configurable hyperparameters (lr, batch_size, epochs, patience, augmentation toggle, data_root). All optional with defaults pre-filled from backend.
- **D-04:** Live training progress displayed as a Recharts line chart showing train_loss and val_loss per epoch (data from `/ws/training` WebSocket), plus current epoch/total, val_acc, and confusion matrix numbers below the chart.
- **D-05:** Cancel button visible during active training. Status indicator (idle/running/completed/failed) always visible.
- **D-06:** Summary metrics (accuracy, precision, recall, F1) as prominent numbers, a visual 2x2 confusion matrix grid, and distribution stats. Per-file results available in a collapsible table below.
- **D-07:** Auto-evaluate after training completes -- automatically trigger evaluation on the newly trained model using default test data. Manual evaluation also available via button.
- **D-08:** Evaluate button includes a model dropdown (from GET `/api/models`) to select which .pt model to evaluate. Optional data_dir override in advanced section.
- **D-09:** List of available .pt models from GET `/api/models` showing name, file size, modification date. Each model has an "Evaluate" button to trigger evaluation on that specific model.
- **D-10:** Active model (the one used for live detection, from `cnn_model_path` config) is visually highlighted/badged in the list.
- **D-11:** Display only -- no model switching from the UI. Hot-swap is a v2 feature.

### Claude's Discretion
- Accordion component implementation (existing library or custom)
- Recharts chart configuration (colors, axis labels, tooltips)
- How auto-eval is triggered (frontend polls training status, or WebSocket signals completion)
- Layout/spacing within each accordion section
- Loading and error states for evaluation results
- How the model dropdown syncs with the models list section

### Deferred Ideas (OUT OF SCOPE)
- Model hot-swap from UI -- v2 feature, requires new backend endpoint
- Ensemble management UI -- future phase if Phase 11 ensemble is activated
- Training history/log persistence -- current progress is ephemeral (Phase 8 D-13)
- Data directory browser in training config -- would need file system browsing API
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRN-04 | Web UI displays training progress and results | Full backend API exists: `/ws/training` WebSocket for live progress, `GET /api/training/progress` for polling, `POST /api/eval/run` for evaluation, `GET /api/models` for model listing. Frontend hooks + components needed. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Web UI stack:** React 19, Vite 8, TypeScript 5.9, Tailwind CSS 4 -- consistent with sky-fort-dashboard
- **Charting:** Recharts (project stack recommendation)
- **Server state:** TanStack Query v5 (already installed)
- **No router:** Sidebar uses local `useState` for tab switching
- **GSD workflow:** Use GSD commands for all repo changes

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | ^19.1.0 | UI framework | Already in package.json |
| @tanstack/react-query | ^5.95.2 | REST data fetching/mutations | Already in package.json, used by all existing hooks |
| TypeScript | ~5.9.3 | Type safety | Already in package.json |
| Tailwind CSS | ^4.2.2 | Styling | Already in package.json, hud-* theme vars established |

### New Dependency
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| recharts | ^3.8.1 | Loss chart (LineChart) | D-04: train_loss/val_loss per epoch chart |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom accordion | headless-ui, radix-ui | Overkill -- a `useState<string \| null>` with conditional rendering is 10 lines. No accessibility requirements specified for this internal tool UI. |
| Recharts | visx, Chart.js | Recharts is the project-chosen charting library. Already in STACK.md. React-native API, minimal boilerplate. |

**Installation:**
```bash
cd web && npm install recharts@^3.8.1
```

**Version verification:** `npm view recharts version` returns 3.8.1 (verified 2026-04-02).

## Architecture Patterns

### Recommended Project Structure
```
web/src/
├── components/
│   ├── training/
│   │   ├── TrainingPanel.tsx       # Main accordion container (Train/Evaluate/Models sections)
│   │   ├── TrainSection.tsx        # Start button, hyperparams, progress chart
│   │   ├── TrainingProgress.tsx    # Recharts loss chart + epoch/acc/confusion display
│   │   ├── EvalSection.tsx         # Model dropdown, run button, results display
│   │   ├── EvaluationResults.tsx   # Metrics cards, confusion matrix grid, per-file table
│   │   └── ModelsSection.tsx       # Model list with evaluate buttons, active badge
│   └── layout/
│       └── Sidebar.tsx             # Modified: add TRAINING tab
├── hooks/
│   ├── useTraining.ts              # TanStack mutations: start, cancel + query: progress
│   ├── useTrainingSocket.ts        # WebSocket hook for /ws/training
│   ├── useEvaluation.ts            # TanStack mutation: run eval
│   └── useModels.ts                # TanStack query: list models
└── utils/
    └── types.ts                    # Add training/eval/model TypeScript interfaces
```

### Pattern 1: Tab Extension in Sidebar
**What:** Add 'training' to the existing tab union type and render TrainingPanel
**When to use:** Sidebar.tsx modification
**Example:**
```typescript
// Sidebar.tsx -- extend tab type and add third button
const [tab, setTab] = useState<'system' | 'recordings' | 'training'>('system')

// In JSX, add third tab button with same pattern as RECORDINGS:
<button
  onClick={() => setTab('training')}
  className={`flex-1 px-3 py-2 text-xs uppercase tracking-wider font-semibold ${
    tab === 'training'
      ? 'text-hud-text border-b-2 border-hud-accent'
      : 'text-hud-text-dim hover:text-hud-text'
  }`}
>
  TRAINING
</button>

// Render: tab === 'training' && <TrainingPanel />
```

### Pattern 2: TanStack Query Hook (mirrors useRecordings.ts)
**What:** Mutations for training start/cancel, queries for models list
**When to use:** All REST API interactions
**Example:**
```typescript
// useTraining.ts
export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (params: TrainingStartParams) =>
      fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      }).then(r => {
        if (!r.ok) return r.json().then(e => Promise.reject(e))
        return r.json()
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training-progress'] }),
  })
}

export function useCancelTraining() {
  return useMutation({
    mutationFn: () =>
      fetch('/api/training/cancel', { method: 'POST' }).then(r => r.json()),
  })
}
```

### Pattern 3: WebSocket Hook (mirrors useRecordingSocket.ts)
**What:** Connect to `/ws/training`, receive epoch-by-epoch JSON, reconnect on disconnect
**When to use:** Live training progress
**Example:**
```typescript
// useTrainingSocket.ts -- same structure as useRecordingSocket.ts
export interface TrainingWsState {
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'failed'
  epoch: number
  total_epochs: number
  train_loss: number
  val_loss: number
  val_acc: number
  confusion_matrix: { tp: number; fp: number; tn: number; fn: number }
  error?: string
}

// Accumulate epoch data for the chart:
// Keep a ref of loss history: { epoch: number; train_loss: number; val_loss: number }[]
// Append on each new epoch message, reset when status transitions to 'running' from non-running
```

### Pattern 4: Custom Accordion
**What:** Simple collapsible sections with shared state
**When to use:** TrainingPanel.tsx container
**Example:**
```typescript
function TrainingPanel() {
  const [openSection, setOpenSection] = useState<'train' | 'evaluate' | 'models' | null>('train')

  const toggle = (section: typeof openSection) =>
    setOpenSection(prev => prev === section ? null : section)

  return (
    <div className="flex flex-col gap-1">
      <AccordionHeader title="TRAIN" open={openSection === 'train'} onToggle={() => toggle('train')} />
      {openSection === 'train' && <TrainSection />}

      <AccordionHeader title="EVALUATE" open={openSection === 'evaluate'} onToggle={() => toggle('evaluate')} />
      {openSection === 'evaluate' && <EvalSection />}

      <AccordionHeader title="MODELS" open={openSection === 'models'} onToggle={() => toggle('models')} />
      {openSection === 'models' && <ModelsSection />}
    </div>
  )
}

function AccordionHeader({ title, open, onToggle }: { title: string; open: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="flex items-center justify-between w-full py-2 text-xs uppercase tracking-wider font-semibold text-hud-text-dim hover:text-hud-text"
    >
      {title}
      <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
        {open ? 'expand_less' : 'expand_more'}
      </span>
    </button>
  )
}
```

### Pattern 5: Auto-Eval After Training Completes
**What:** WebSocket signals completion, frontend auto-triggers evaluation
**When to use:** D-07 auto-evaluate behavior
**Recommended approach:** Use WebSocket status transition detection. When `useTrainingSocket` transitions from `running` to `completed`, automatically call `useRunEvaluation().mutate()` with the default model path. This avoids polling and is event-driven.
```typescript
// In TrainSection or TrainingPanel:
const prevStatus = useRef(trainingState.status)
useEffect(() => {
  if (prevStatus.current === 'running' && trainingState.status === 'completed') {
    runEvalMutation.mutate({ model_path: undefined, data_dir: undefined })
  }
  prevStatus.current = trainingState.status
}, [trainingState.status])
```

### Anti-Patterns to Avoid
- **Polling REST for training progress:** The `/ws/training` WebSocket exists specifically for this. Use WebSocket for live updates, REST `GET /progress` only as fallback/initial load.
- **Multiple open accordions eating sidebar space:** D-02 says "one or two open at a time." Using `useState<string | null>` naturally enforces single-open. If two are needed, use `useState<Set<string>>` with max-2 enforcement.
- **Storing chart data in component state without reset:** Loss history array must reset when a new training run starts (status transitions to 'running' from idle/completed/failed).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loss chart | Custom SVG/Canvas chart | Recharts `LineChart` | Axes, tooltips, responsive sizing, legends are complex. Recharts handles all of it. |
| WebSocket reconnection | Custom retry logic from scratch | Copy `useRecordingSocket.ts` pattern | Already proven in this codebase: exponential backoff, cleanup on unmount, state reset on disconnect. |
| HTTP error handling in mutations | Ad-hoc try/catch | TanStack Query `onError` + mutation `.error` state | Consistent with existing hooks, handles loading/error/success states automatically. |

## Common Pitfalls

### Pitfall 1: Loss Chart Data Accumulation Across Runs
**What goes wrong:** Chart shows data from previous training run mixed with new run.
**Why it happens:** Loss history array is not cleared when a new training run starts.
**How to avoid:** Clear the loss history array when WebSocket status transitions to 'running' and epoch resets to 0/1.
**Warning signs:** Chart shows discontinuous jumps or more data points than total_epochs.

### Pitfall 2: WebSocket Message Shape Varies by Status
**What goes wrong:** Destructuring epoch/loss fields from an 'idle' or 'failed' message causes undefined values.
**Why it happens:** The backend `_progress_to_ws_dict` only includes epoch/loss/confusion_matrix when status is 'running' or 'completed'. For 'idle' it sends only `{"status": "idle"}`. For 'failed' it sends `{"status": "failed", "error": "..."}`.
**How to avoid:** Type the WebSocket state with optional fields. Check status before accessing training-specific fields.
**Warning signs:** Chart renders NaN or undefined values.

### Pitfall 3: Auto-Eval Fires Multiple Times
**What goes wrong:** Evaluation runs multiple times when training completes.
**Why it happens:** React strict mode double-invokes effects. Or WebSocket sends multiple messages with 'completed' status.
**How to avoid:** Use a ref to track whether auto-eval has already been triggered for the current run. Reset the ref when a new run starts.
**Warning signs:** Multiple POST /api/eval/run requests in network tab.

### Pitfall 4: Stale Model List After Training
**What goes wrong:** Newly trained model does not appear in the models dropdown/list.
**Why it happens:** Model list query is cached by TanStack Query and not invalidated after training completes.
**How to avoid:** Invalidate `['models']` query key when training status transitions to 'completed'.
**Warning signs:** User has to manually refresh to see new model.

### Pitfall 5: Sidebar Tab Width With Three Tabs
**What goes wrong:** Three `flex-1` tab buttons may make text cramped, especially "RECORDINGS" and "TRAINING" are long words.
**Why it happens:** Fixed sidebar width divided by 3 instead of 2.
**How to avoid:** Consider shorter tab labels or check that the existing sidebar width accommodates three tabs. "SYS / REC / TRAIN" or icon tabs if space is tight.
**Warning signs:** Tab text wraps or overflows.

## Code Examples

### Recharts Loss Chart Configuration
```typescript
// Source: Recharts v3 API (verified)
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface LossDataPoint {
  epoch: number
  train_loss: number
  val_loss: number
}

function LossChart({ data }: { data: LossDataPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="epoch"
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          stroke="#1f2937"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          stroke="#1f2937"
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', fontSize: 12 }}
          labelStyle={{ color: '#e5e7eb' }}
          itemStyle={{ color: '#e5e7eb' }}
        />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" dot={false} name="Train" />
        <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" dot={false} name="Val" />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

### Confusion Matrix 2x2 Grid
```typescript
// Visual confusion matrix matching hud theme
function ConfusionMatrix({ tp, fp, tn, fn }: { tp: number; fp: number; tn: number; fn: number }) {
  return (
    <div className="grid grid-cols-2 gap-1 text-center text-xs">
      <div className="bg-hud-success/20 text-hud-success p-2 rounded">
        <div className="font-mono text-lg">{tp}</div>
        <div>TP</div>
      </div>
      <div className="bg-hud-danger/20 text-hud-danger p-2 rounded">
        <div className="font-mono text-lg">{fp}</div>
        <div>FP</div>
      </div>
      <div className="bg-hud-danger/20 text-hud-danger p-2 rounded">
        <div className="font-mono text-lg">{fn}</div>
        <div>FN</div>
      </div>
      <div className="bg-hud-success/20 text-hud-success p-2 rounded">
        <div className="font-mono text-lg">{tn}</div>
        <div>TN</div>
      </div>
    </div>
  )
}
```

### TypeScript Interfaces for Backend API Responses
```typescript
// Add to web/src/utils/types.ts or a new training-types.ts

export interface TrainingStartParams {
  learning_rate?: number
  batch_size?: number
  max_epochs?: number
  patience?: number
  augmentation_enabled?: boolean
  data_root?: string
}

export interface TrainingProgressResponse {
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'failed'
  epoch: number
  total_epochs: number
  train_loss: number
  val_loss: number
  val_acc: number
  best_val_loss: number
  best_epoch: number
  confusion_matrix: ConfusionMatrixData
  error: string | null
}

export interface ConfusionMatrixData {
  tp: number
  fp: number
  tn: number
  fn: number
}

export interface EvalRunParams {
  model_path?: string
  data_dir?: string
}

export interface EvalResultResponse {
  summary: {
    total: number
    correct: number
    incorrect: number
    accuracy: number
    precision: number
    recall: number
    f1: number
    confusion_matrix: ConfusionMatrixData
  }
  distribution: Record<string, {
    p_agg: DistributionStats
    p_max: DistributionStats
    p_mean: DistributionStats
  }>
  per_file: FileResult[]
  model_path: string
  data_dir: string
  message: string
}

export interface DistributionStats {
  p25: number
  p50: number
  p75: number
  p95: number
}

export interface FileResult {
  filename: string
  true_label: string
  predicted_label: string
  p_agg: number
  correct: boolean
}

export interface ModelInfo {
  filename: string
  path: string
  size_bytes: number
  modified: string  // ISO 8601
}

export interface ModelListResponse {
  models: ModelInfo[]
}
```

### Training Hyperparameters Defaults (from backend TrainingConfig)
```typescript
// Pre-fill these in the "Advanced" section of the Train accordion
const TRAINING_DEFAULTS = {
  learning_rate: 0.001,
  batch_size: 32,
  max_epochs: 50,
  patience: 5,
  augmentation_enabled: true,
  data_root: 'audio-data/data/',
} as const
```

## Backend API Contract Reference

Verified from source code (HIGH confidence):

| Endpoint | Method | Request Body | Response | Notes |
|----------|--------|-------------|----------|-------|
| `/api/training/start` | POST | `TrainingStartRequest` (all fields optional) | `{ message: string }` | 409 if already training |
| `/api/training/progress` | GET | -- | `TrainingProgressResponse` | Polling fallback |
| `/api/training/cancel` | POST | -- | `{ message: string }` | 409 if not training |
| `/api/eval/run` | POST | `{ model_path?: string, data_dir?: string }` | `EvalResultResponse` | Sync, runs in executor. 404 for missing model/data |
| `/api/models` | GET | -- | `{ models: ModelInfo[] }` | Scans directory of configured model |
| `/ws/training` | WS | -- | JSON per epoch | Status-dependent fields (see Pitfall 2) |

### WebSocket `/ws/training` Message Shapes

**Idle:** `{ "status": "idle" }`

**Running:** `{ "status": "running", "epoch": 5, "total_epochs": 50, "train_loss": 0.42, "val_loss": 0.51, "val_acc": 0.85, "confusion_matrix": { "tp": 120, "fp": 8, "tn": 95, "fn": 12 } }`

**Completed:** Same as running but `"status": "completed"`

**Failed:** `{ "status": "failed", "error": "Out of memory" }`

**Cancelled:** `{ "status": "cancelled" }` (no extra fields)

### Active Model Detection
The active model (D-10) is determined by `settings.cnn_model_path`. This path is not directly exposed via API. **Recommendation:** Compare model filenames in the list against the `cnn_model_path` setting. The backend `GET /api/models` does not include an `active` flag. Two options:
1. Add a simple `GET /api/settings/model-path` endpoint (minimal backend change)
2. Use the health endpoint if it exposes model path (it does not currently)
3. Hardcode comparison against a known default path

**Recommended:** Option 1 -- add a tiny endpoint, or include `active_model_path` field in the `ModelListResponse`. This is a 5-line backend addition that makes the frontend cleaner. If avoiding backend changes entirely, the frontend can call `GET /api/training/progress` which might give enough context, but the cleanest solution is a small backend addition.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Recharts v2 (class components) | Recharts v3 (hooks, tree-shakeable) | 2025 | v3 is current. Import from 'recharts' directly. |
| React 18 useEffect patterns | React 19 (same patterns for effects) | 2025 | No change to WebSocket hook pattern. |

## Open Questions

1. **Active model identification (D-10)**
   - What we know: Backend returns model list but no "active" flag. Active model is `settings.cnn_model_path`.
   - What's unclear: How to communicate active model path to frontend without a backend change.
   - Recommendation: Add `active_model_path: str` field to `ModelListResponse` (5-line backend change). If strictly frontend-only, hardcode comparison against known default path `models/research_cnn_trained.pt`.

2. **Evaluation blocking behavior**
   - What we know: `POST /api/eval/run` is synchronous (runs in executor, blocks until complete). Could take seconds to minutes depending on dataset size.
   - What's unclear: Whether the UI should show a loading spinner or stream progress.
   - Recommendation: Show a loading spinner (TanStack Query `isPending` state). Evaluation is typically fast (seconds). No WebSocket needed for eval.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None currently installed for frontend |
| Config file | none -- see Wave 0 |
| Quick run command | `cd web && npx vitest run --reporter=verbose` |
| Full suite command | `cd web && npx vitest run` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRN-04 | Training progress displays in UI | manual-only | Visual inspection of sidebar training tab | N/A |
| TRN-04 | Training hooks call correct endpoints | unit | `cd web && npx vitest run src/hooks/useTraining.test.ts` | Wave 0 |
| TRN-04 | Eval hooks call correct endpoints | unit | `cd web && npx vitest run src/hooks/useEvaluation.test.ts` | Wave 0 |

### Sampling Rate
- **Per task commit:** TypeScript compilation: `cd web && npx tsc -b --noEmit`
- **Per wave merge:** `cd web && npm run build` (full build including type check)
- **Phase gate:** Full build green + manual visual inspection of training tab

### Wave 0 Gaps
- [ ] `vitest` + `@testing-library/react` not installed -- framework setup needed if unit tests required
- [ ] No test config exists (`vitest.config.ts`)
- [ ] Given this is a UI-only phase with no complex logic, type-check + build verification may be sufficient. Unit testing hooks is optional.

## Sources

### Primary (HIGH confidence)
- `web/src/components/layout/Sidebar.tsx` -- existing two-tab pattern (read directly)
- `web/src/hooks/useRecordings.ts` -- TanStack Query mutation pattern (read directly)
- `web/src/hooks/useRecordingSocket.ts` -- WebSocket hook pattern (read directly)
- `web/src/components/recording/RecordingPanel.tsx` -- panel UI state machine (read directly)
- `src/acoustic/api/training_routes.py` -- training REST endpoints (read directly)
- `src/acoustic/api/eval_routes.py` -- evaluation REST endpoints (read directly)
- `src/acoustic/api/model_routes.py` -- model listing endpoint (read directly)
- `src/acoustic/api/models.py` -- all Pydantic request/response models (read directly)
- `src/acoustic/api/websocket.py` lines 233-283 -- training WebSocket protocol (read directly)
- `src/acoustic/training/config.py` -- default hyperparameter values (read directly)
- `web/package.json` -- current dependencies (read directly)

### Secondary (MEDIUM confidence)
- `npm view recharts version` -- confirmed v3.8.1 is latest (2026-04-02)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all dependencies verified, only addition is recharts
- Architecture: HIGH -- follows existing codebase patterns exactly, all backend APIs verified from source
- Pitfalls: HIGH -- derived from reading actual backend code (message shape variation, cache invalidation)

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable -- frontend patterns, existing backend APIs)
