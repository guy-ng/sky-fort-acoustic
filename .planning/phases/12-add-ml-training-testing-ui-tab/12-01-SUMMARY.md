---
phase: 12-add-ml-training-testing-ui-tab
plan: 01
subsystem: frontend-data-layer
tags: [hooks, types, recharts, tanstack-query, websocket]
dependency_graph:
  requires: [phase-08-training-api, phase-09-eval-api]
  provides: [training-hooks, eval-hooks, model-hooks, training-ws-hook, ml-types]
  affects: [12-02-ui-components]
tech_stack:
  added: [recharts@^3.8.1, "@types/node"]
  patterns: [tanstack-query-mutations, websocket-reconnect, loss-history-accumulation]
key_files:
  created:
    - web/src/hooks/useTraining.ts
    - web/src/hooks/useTrainingSocket.ts
    - web/src/hooks/useEvaluation.ts
    - web/src/hooks/useModels.ts
  modified:
    - web/src/utils/types.ts
    - web/package.json
    - web/package-lock.json
decisions:
  - "Used useState for lossHistory (not ref) to trigger re-renders on new data points"
  - "Clear loss history when epoch === 1 to detect new training run start"
metrics:
  duration: 4m39s
  completed: 2026-04-02
---

# Phase 12 Plan 01: Training/Eval/Model Data Layer Summary

Installed Recharts, defined all TypeScript interfaces matching backend Pydantic models, and created 4 data hooks (useTraining, useTrainingSocket, useEvaluation, useModels) as the complete frontend data layer for the ML training UI tab.

## What Was Done

### Task 1: Install Recharts and define TypeScript interfaces (b33b769)

- Installed `recharts@^3.8.1` as a dependency
- Added 14 TypeScript interfaces to `web/src/utils/types.ts` covering training, evaluation, model listing, and WebSocket message types
- All interfaces mirror backend Pydantic models field-for-field, including `PerModelResult` and `ensemble_config_path`
- Preserved all existing types (TargetState, HeatmapHandshake, HealthStatus, BeamformingMapResponse)

### Task 2: Create data hooks (0ffaa85)

- **useTraining.ts**: `useStartTraining()` mutation (POST /api/training/start), `useCancelTraining()` mutation (POST /api/training/cancel), `useTrainingProgress()` query (GET /api/training/progress with 5s polling)
- **useTrainingSocket.ts**: WebSocket connection to /ws/training with reconnect logic, returns `TrainingWsMessage` state and `LossDataPoint[]` history. Clears history on new run (epoch 1)
- **useEvaluation.ts**: `useRunEvaluation()` mutation (POST /api/eval/run with EvalRunParams body)
- **useModels.ts**: `useModels()` query (GET /api/models with 30s polling)
- All hooks follow identical patterns to existing `useRecordings.ts` and `useRecordingSocket.ts`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing @types/node for vite.config.ts compilation**
- **Found during:** Task 2 verification
- **Issue:** Pre-existing `tsc -b` failure on vite.config.ts due to missing `@types/node` (NodeJS namespace, http module, console)
- **Fix:** Installed `@types/node` as devDependency
- **Files modified:** web/package.json, web/package-lock.json

## Decisions Made

1. Used `useState` for `lossHistory` (not just a ref) so React components re-render when new training data points arrive
2. Clear loss history array when `epoch === 1` to detect the start of a new training run

## Known Stubs

None - all hooks are fully wired to real API endpoints with proper types.

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | b33b769 | feat(12-01): install recharts and define training/eval/model TypeScript interfaces |
| 2 | 0ffaa85 | feat(12-01): create training/eval/model data hooks and fix node types |

## Self-Check: PASSED

- All 5 files verified present on disk
- Both commits (b33b769, 0ffaa85) verified in git log
