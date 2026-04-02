# Phase 12: Add ML Training & Testing UI Tab - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 12-add-ml-training-testing-ui-tab
**Areas discussed:** Tab layout & navigation, Training control panel, Evaluation results view, Model selection & management

---

## Tab Layout & Navigation

| Option | Description | Selected |
|--------|-------------|----------|
| Third sidebar tab | Add a TRAINING tab alongside SYSTEM and RECORDINGS. Contains all training, eval, and model management UI in one place. | ✓ |
| Subtabs within TRAINING tab | TRAINING tab with internal subtabs (Train / Evaluate / Models). More organized but adds navigation depth. | |
| You decide | Claude picks the best approach. | |

**User's choice:** Third sidebar tab
**Notes:** Consistent with existing tab pattern.

| Option | Description | Selected |
|--------|-------------|----------|
| Stacked sections | All sections stacked vertically in one scrollable panel. | |
| Collapsible accordion | Each section is collapsible, only one or two open at a time. | ✓ |
| You decide | Claude picks. | |

**User's choice:** Collapsible accordion
**Notes:** Saves vertical space in the sidebar.

---

## Training Control Panel

| Option | Description | Selected |
|--------|-------------|----------|
| Simple start button | Just a 'Start Training' button with defaults. Maybe augmentation toggle. | |
| Expandable config form | Start button with 'Advanced' expandable section showing lr, batch_size, epochs, patience, augmentation toggle, data_root. | ✓ |
| Full config form | All hyperparameters visible upfront. | |

**User's choice:** Expandable config form
**Notes:** All optional with defaults pre-filled.

| Option | Description | Selected |
|--------|-------------|----------|
| Loss chart + metrics | Live line chart (train_loss, val_loss per epoch via /ws/training), plus epoch counter, val_acc, confusion matrix numbers. Recharts. | ✓ |
| Text-only progress | No chart -- just epoch counter, loss/acc numbers, status text. | |
| You decide | Claude picks. | |

**User's choice:** Loss chart + metrics
**Notes:** Recharts for the chart component.

---

## Evaluation Results View

| Option | Description | Selected |
|--------|-------------|----------|
| Summary metrics + confusion matrix | Accuracy, precision, recall, F1 as prominent numbers, visual 2x2 confusion matrix, distribution stats. Per-file results in collapsible table. | ✓ |
| Compact summary only | Just key numbers and confusion matrix. No per-file breakdown. | |
| Full detailed view | Everything always visible including per-file table. | |

**User's choice:** Summary metrics + confusion matrix with collapsible per-file table.

| Option | Description | Selected |
|--------|-------------|----------|
| Evaluate button with model picker | 'Evaluate' button with dropdown to select model. Optional data_dir override in advanced section. | |
| Auto-eval after training | Automatically run evaluation when training completes, plus manual trigger. | ✓ |
| You decide | Claude picks. | |

**User's choice:** Auto-eval after training (plus manual trigger)
**Notes:** Operator sees results immediately without extra clicks after training completes.

---

## Model Selection & Management

| Option | Description | Selected |
|--------|-------------|----------|
| Model list with eval action | List of .pt models showing name, size, date. Each has 'Evaluate' button. Active model highlighted. | ✓ |
| Simple model list | Just list models with metadata. No actions from list. | |
| You decide | Claude picks. | |

**User's choice:** Model list with eval action and active model highlighted.

| Option | Description | Selected |
|--------|-------------|----------|
| No -- display only | Show active model but no switching from UI. Hot-swap is v2. | ✓ |
| Yes -- activate button | Allow model switching from UI. Needs new backend endpoint. | |

**User's choice:** Display only -- no hot-swap from UI.

---

## Claude's Discretion

- Accordion component implementation
- Recharts configuration (colors, labels, tooltips)
- Auto-eval trigger mechanism
- Loading/error states
- Layout spacing within accordion sections

## Deferred Ideas

- Model hot-swap from UI (v2)
- Ensemble management UI (future phase)
- Training history persistence
- Data directory browser
