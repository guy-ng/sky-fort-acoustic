---
phase: 12
slug: add-ml-training-testing-ui-tab
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest (already configured in web/) |
| **Config file** | `web/vitest.config.ts` |
| **Quick run command** | `cd web && npx vitest run --reporter=verbose` |
| **Full suite command** | `cd web && npx vitest run` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd web && npx tsc -b --noEmit` (type-check -- primary automated gate)
- **After every plan wave:** Run `cd web && npm run build` (full production build)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 12-01-01 | 01 | 1 | TRN-04 | type-check | `cd web && npx tsc -b --noEmit` | pending |
| 12-01-02 | 01 | 1 | TRN-04 | type-check | `cd web && npx tsc -b --noEmit` | pending |
| 12-02-01 | 02 | 2 | TRN-04 | type-check | `cd web && npx tsc -b --noEmit` | pending |
| 12-02-02 | 02 | 2 | TRN-04 | type-check | `cd web && npx tsc -b --noEmit` | pending |
| 12-02-03 | 02 | 2 | TRN-04 | build | `cd web && npm run build` | pending |
| 12-02-04 | 02 | 2 | TRN-04 | visual | `cd web && npm run build` (+ human verify) | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Wave 0 is not needed for this phase. All tasks use `tsc --noEmit` or `npm run build` as automated verification, which are sufficient Nyquist-compliant checks for a UI component phase. The existing vitest infrastructure can be leveraged for additional tests if needed, but type-checking provides the primary automated gate for correctness.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Loss chart renders real-time epoch data | TRN-04 | Recharts rendering + WebSocket stream | Start training, verify chart updates per epoch |
| Accordion sections expand/collapse | UX | Visual interaction | Click each accordion header, verify content toggles |
| Auto-eval triggers after training | D-07 | End-to-end WebSocket flow | Complete training, verify eval results appear automatically |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands (tsc --noEmit or npm run build)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter
- [x] `wave_0_complete: true` set in frontmatter

**Approval:** approved
