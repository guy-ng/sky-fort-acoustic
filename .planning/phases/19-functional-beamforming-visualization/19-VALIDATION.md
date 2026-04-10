---
phase: 19
slug: functional-beamforming-visualization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 19 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + vitest |
| **Config file** | `pyproject.toml` / `web/vitest.config.ts` |
| **Quick run command** | `python -m pytest tests/ -x -q --timeout=10` |
| **Full suite command** | `python -m pytest tests/ -q --timeout=30 && cd web && npx vitest run` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q --timeout=10`
- **After every plan wave:** Run `python -m pytest tests/ -q --timeout=30 && cd web && npx vitest run`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 19-01-01 | 01 | 1 | VIZ-02 | — | N/A | unit | `python -m pytest tests/test_beamforming.py -x -q` | ✅ | ⬜ pending |
| 19-01-02 | 01 | 1 | VIZ-01 | — | N/A | unit | `python -m pytest tests/test_config.py -x -q` | ✅ | ⬜ pending |
| 19-02-01 | 02 | 1 | VIZ-02 | — | N/A | integration | `python -m pytest tests/test_api.py -x -q` | ✅ | ⬜ pending |
| 19-03-01 | 03 | 2 | VIZ-01 | — | N/A | unit | `cd web && npx vitest run --reporter=verbose` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Heatmap shows sharper peaks with nu=100 | VIZ-02 | Visual quality assessment | Open web UI, play audio with known source, verify distinct peaks vs smeared energy |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
