# Phase 12: Add ML Training & Testing UI Tab - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a frontend UI tab in the sidebar for controlling CNN training, viewing evaluation results, and managing models. This phase surfaces the Phase 8/9 backend APIs (training start/cancel/progress, evaluation run/results, model listing) to the operator via a new TRAINING sidebar tab with collapsible accordion sections. This phase does NOT include backend changes (APIs already exist), model hot-swapping (v2), ensemble UI management, or new training features.

</domain>

<decisions>
## Implementation Decisions

### Tab Layout & Navigation
- **D-01:** New third sidebar tab "TRAINING" alongside existing SYSTEM and RECORDINGS tabs. Uses the same local `useState` tab switching pattern (Phase 10 decision).
- **D-02:** Content organized as collapsible accordion sections within the tab. Sections: Train, Evaluate, Models. Only one or two open at a time to save vertical space.

### Training Control Panel
- **D-03:** "Start Training" button with an expandable "Advanced" section showing configurable hyperparameters (lr, batch_size, epochs, patience, augmentation toggle, data_root). All optional with defaults pre-filled from backend.
- **D-04:** Live training progress displayed as a Recharts line chart showing train_loss and val_loss per epoch (data from `/ws/training` WebSocket), plus current epoch/total, val_acc, and confusion matrix numbers below the chart.
- **D-05:** Cancel button visible during active training. Status indicator (idle/running/completed/failed) always visible.

### Evaluation Results View
- **D-06:** Summary metrics (accuracy, precision, recall, F1) as prominent numbers, a visual 2x2 confusion matrix grid, and distribution stats. Per-file results available in a collapsible table below.
- **D-07:** Auto-evaluate after training completes — automatically trigger evaluation on the newly trained model using default test data. Manual evaluation also available via button.
- **D-08:** Evaluate button includes a model dropdown (from GET `/api/models`) to select which .pt model to evaluate. Optional data_dir override in advanced section.

### Model Selection & Management
- **D-09:** List of available .pt models from GET `/api/models` showing name, file size, modification date. Each model has an "Evaluate" button to trigger evaluation on that specific model.
- **D-10:** Active model (the one used for live detection, from `cnn_model_path` config) is visually highlighted/badged in the list.
- **D-11:** Display only — no model switching from the UI. Hot-swap is a v2 feature.

### Claude's Discretion
- Accordion component implementation (existing library or custom)
- Recharts chart configuration (colors, axis labels, tooltips)
- How auto-eval is triggered (frontend polls training status, or WebSocket signals completion)
- Layout/spacing within each accordion section
- Loading and error states for evaluation results
- How the model dropdown syncs with the models list section

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Backend APIs (Phase 12 consumes these)
- `src/acoustic/api/training_routes.py` -- POST /api/training/start, GET /api/training/progress, POST /api/training/cancel
- `src/acoustic/api/eval_routes.py` -- POST /api/eval/run (accepts model_path, data_dir)
- `src/acoustic/api/model_routes.py` -- GET /api/models (returns .pt files with metadata)
- `src/acoustic/api/models.py` -- Pydantic request/response models: TrainingStartRequest, TrainingProgressResponse, EvalRunRequest, EvalResultResponse, ConfusionMatrixResponse

### WebSocket Endpoints
- `src/acoustic/api/websocket.py` -- `/ws/training` endpoint (lines 255-283). Sends per-epoch JSON: status, epoch, total_epochs, train_loss, val_loss, val_acc, confusion_matrix. Initial status on connect, then updates on change.

### Existing Frontend Patterns
- `web/src/components/layout/Sidebar.tsx` -- Current two-tab sidebar (SYSTEM, RECORDINGS). Phase 12 adds third tab here.
- `web/src/components/recording/RecordingPanel.tsx` -- Reference for panel layout, button states, status indicators within a sidebar tab.
- `web/src/components/recording/RecordingsList.tsx` -- Reference for list display with action buttons.
- `web/src/hooks/useRecordings.ts` -- Reference for TanStack Query mutation/query patterns for REST endpoints.
- `web/src/hooks/useRecordingSocket.ts` -- Reference for WebSocket hook pattern (used for /ws/recording).

### Prior Phase Context
- `.planning/phases/08-pytorch-training-pipeline/08-CONTEXT.md` -- Training pipeline decisions (D-06 hyperparameter defaults, D-13 progress state, D-14 cancellation).
- `.planning/phases/09-evaluation-harness-and-api/09-CONTEXT.md` -- API design (D-05 route prefixes, D-11-D-13 WebSocket protocol, D-14 sync eval).

### Requirements
- `.planning/REQUIREMENTS.md` -- TRN-04 (training progress in UI) is the primary requirement this phase satisfies.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Sidebar.tsx` -- Two-tab pattern with useState. Add third tab with same pattern.
- `RecordingPanel.tsx` -- Three-phase UI (idle/recording/labeling). Training panel mirrors this (idle/training/completed).
- `useRecordings.ts` hooks -- TanStack Query patterns for mutations (start/stop/label). Reuse for training start/cancel.
- `useRecordingSocket.ts` -- WebSocket hook pattern. Adapt for `/ws/training` endpoint.
- `AudioLevelMeter.tsx` -- Real-time visualization component. Reference for chart/meter patterns.

### Established Patterns
- Local `useState` for sidebar tab switching (no router)
- TanStack Query for REST data fetching and mutations
- WebSocket hooks with reconnection for real-time data
- Tailwind CSS v4 styling consistent with sky-fort-dashboard
- Panel component for UI sections within sidebar

### Integration Points
- Sidebar.tsx: Add TRAINING tab alongside SYSTEM and RECORDINGS
- New hooks: `useTraining.ts` (start/cancel/progress queries), `useTrainingSocket.ts` (WebSocket), `useEvaluation.ts` (run eval, get results), `useModels.ts` (list models)
- New components: `TrainingPanel.tsx`, `TrainingProgress.tsx` (chart + metrics), `EvaluationResults.tsx`, `ModelsList.tsx`
- Recharts dependency needed for loss chart visualization

</code_context>

<specifics>
## Specific Ideas

- Auto-eval after training means the operator sees results immediately without extra clicks -- important for iterative model development workflow
- Collapsible accordion keeps the sidebar manageable since 3 sections (train, eval, models) is a lot of content for a sidebar panel
- Loss chart via Recharts matches the project's charting choice (already in stack recommendation)
- Model list with per-model evaluate button lets the operator quickly compare models without navigating between sections

</specifics>

<deferred>
## Deferred Ideas

- Model hot-swap from UI -- v2 feature, requires new backend endpoint
- Ensemble management UI -- future phase if Phase 11 ensemble is activated
- Training history/log persistence -- current progress is ephemeral (Phase 8 D-13)
- Data directory browser in training config -- would need file system browsing API

</deferred>

---

*Phase: 12-add-ml-training-testing-ui-tab*
*Context gathered: 2026-04-02*
