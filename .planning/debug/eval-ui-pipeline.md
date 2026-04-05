---
status: awaiting_human_verify
trigger: "Three issues: eval Start goes to black page, model Evaluate button non-functional, no pipeline start with trained model"
created: 2026-04-03T00:00:00Z
updated: 2026-04-03T00:00:00Z
---

## Current Focus

hypothesis: Three distinct root causes identified — fixing all three
test: Apply fixes and verify
expecting: Eval works inline, models eval shows results, pipeline activation available
next_action: Implement fixes

## Symptoms

expected: |
  1. After training, clicking "Start" evaluation shows evaluation results
  2. "Evaluate" button next to models triggers evaluation
  3. Way to select trained model and start real-time detection pipeline
actual: |
  1. "Start" evaluation navigates to black/empty page
  2. Evaluate button next to models does nothing
  3. No pipeline start functionality exists
errors: No console errors reported — empty/non-functional UI
reproduction: Train a model, then try to evaluate it or use it
started: Recently built UI — current implementation issues

## Eliminated

## Evidence

- timestamp: 2026-04-03
  checked: useEvaluation.ts mutationFn
  found: fetch('/api/eval/run').then(r => r.json()) does NOT check r.ok. Backend returns {message: "..."} on 404 errors (model/data not found), which is not EvalResultResponse shape. EvaluationResults component then crashes accessing result.summary.accuracy on the error object.
  implication: ROOT CAUSE for bug 1 — eval "black page" is a crash when trying to render error response as results

- timestamp: 2026-04-03
  checked: ModelsSection.tsx evaluate button
  found: Button calls evalMutation.mutate({model_path: model.path}) correctly. The mutation fires but results are only rendered in EvalSection (separate accordion section). ModelsSection has no local state to show results from its own evaluate action.
  implication: ROOT CAUSE for bug 2 — evaluate button fires API call but results have nowhere to display in ModelsSection

- timestamp: 2026-04-03
  checked: All API routes and main.py lifespan
  found: No API endpoint exists to activate/load a trained model into the running pipeline. Pipeline CNN is initialized at startup from config. No hot-swap mechanism exposed via REST.
  implication: ROOT CAUSE for bug 3 — no pipeline activation feature exists yet

## Resolution

root_cause: |
  Bug 1: useEvaluation.ts mutationFn did not check response.ok before parsing JSON. Backend returns {message: "..."} on 404 errors, which is not EvalResultResponse shape. EvaluationResults crashes accessing result.summary.accuracy on error response, rendering blank.
  Bug 2: ModelsSection evaluate button fired evalMutation.mutate() correctly, but results only rendered in EvalSection (separate accordion section that was closed). No feedback visible to the user.
  Bug 3: No API endpoint or UI existed to hot-swap a trained model into the running CNN pipeline.
fix: |
  Bug 1: Added proper error handling in useEvaluation.ts — check res.ok, throw Error with backend message on failure. Now shows error in EvalSection error state.
  Bug 2: Lifted eval trigger from ModelsSection to TrainingPanel. ModelsSection.onEvaluateModel callback switches accordion to EVALUATE section and auto-triggers evaluation via requestedModelPath prop on EvalSection.
  Bug 3: Added CNNWorker.set_classifier() for thread-safe hot-swap. Created /api/pipeline/activate endpoint that loads a .pt model, validates it, and swaps the classifier. Added "Activate" button to ModelsSection UI with usePipeline hook.
verification: TypeScript compiles clean, Python routes register correctly, module imports verified.
files_changed:
  - web/src/hooks/useEvaluation.ts
  - web/src/components/training/EvalSection.tsx
  - web/src/components/training/TrainingPanel.tsx
  - web/src/components/training/ModelsSection.tsx
  - web/src/hooks/usePipeline.ts
  - src/acoustic/classification/worker.py
  - src/acoustic/api/pipeline_routes.py
  - src/acoustic/api/models.py
  - src/acoustic/main.py
