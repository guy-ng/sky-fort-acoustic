# Kaggle Drone Dataset -- Investigation (Phase 22)

**Requested by:** user mid-workflow, 2026-04-08
**Investigated:** 2026-04-12
**Source:** https://www.kaggle.com/code/yehiellevi/audio-drone-sound-detection
**Likely upstream:** https://github.com/saraalemadi/DroneAudioDataset

## Technical Findings

| Field | Value | Source |
|---|---|---|
| License | **NONE** -- no LICENSE file, GitHub API returns `null` | GitHub API + raw fetch (404) |
| Size | ~281 MB (280,899 KB) | GitHub API `size` field |
| Total files | 23,408 audio files | Git tree enumeration |
| Splits | `Binary_Drone_Audio/` (yes_drone, unknown), `Multiclass_Drone_Audio/` (bebop_1, membo_1, unknown) | Directory structure |
| Format | WAV / PCM_16 | soundfile.info probe |
| Sample rate | **16,000 Hz** (all probed files) | soundfile.info probe |
| Channels | **1 (mono)** | soundfile.info probe |
| Labels | Binary: `yes_drone` / `unknown`. Multiclass: `bebop_1` / `membo_1` / `unknown` | Directory structure |
| Duration per clip | ~0.90s - 1.02s | soundfile.info probe (7 files) |
| File size per clip | ~30-33 KB | Git tree metadata |
| Recording environment | **Indoor** -- "recorded of drone propellers noise in an indoor environment" | README |
| Drone models | DJI Bebop, DJI Membo (Mambo?) | File naming + README |
| Noise source for "unknown" | **ESC-50** subset + Speech Commands white noise + custom silence clip | README attribution |
| Paper | Al-Emadi et al., "Audio Based Drone Detection and Identification using Deep Learning", IWCMC 2019 | README BibTeX |
| Repo created | 2018-12-09 | GitHub API |
| Last updated | 2026-04-08 | GitHub API |
| Stars | 102 | GitHub API |
| Kaggle notebook relationship | The notebook by yehiellevi references/uses this dataset for drone sound classification | Notebook metadata |

## Overlap Analysis

- **DADS corpus:** Different source dataset. DADS is a separate drone audio dataset. No direct overlap in recordings. However, both use short clips (~1s) with similar labeling (drone/not-drone).
- **ESC-50 overlap:** The "unknown" class in DroneAudioDataset explicitly uses ESC-50 clips as noise/background. We already have ESC-50 in `data/noise/esc50/` from Phase 20.1. **Direct overlap in the negative class.**
- **2026-04-08 field recordings:** No overlap -- different dates, different array (UMA-16 outdoor vs indoor single mic).
- **Risk of double-counting:** **HIGH for negatives** (ESC-50 clips are reused). **LOW for positives** (indoor Bebop/Membo recordings are unique to this dataset).

## Decision Criteria (for human review in Task 2)

| Criterion | Met? | Notes |
|---|---|---|
| License compatible with project use | **NO** | No license file exists. GitHub API reports `null`. README says "cite via BibTeX" but provides no explicit license grant. Under default copyright law, no license = all rights reserved. Cannot legally redistribute or use without explicit permission from Sara Al-Emadi et al. |
| Sample rate can resample to 32 kHz without loss | Partial | 16 kHz mono. Upsampling to 32 kHz is technically possible but adds no information -- just interpolation. Our pipeline resamples DADS from 16 kHz to 32 kHz already, so this is consistent. |
| Labels map to drone/not-drone cleanly | YES | Binary split maps directly: `yes_drone` -> drone, `unknown` -> not-drone. Multiclass adds model distinction (bebop_1, membo_1) which we could collapse. |
| Not a subset of DADS (no double-counting) | YES (positives) / **NO (negatives)** | Drone recordings are unique (indoor Bebop/Membo). But "unknown" class reuses ESC-50 clips we already have in `data/noise/esc50/`. Would need de-duplication for negatives. |
| Total size manageable (< 5 GB after subset) | YES | ~281 MB total. |
| Recording conditions defensible (not in-cabin / bee-mixed) | **POOR FIT** | **Indoor** recordings. Our operational environment is outdoor UMA-16 array. Indoor acoustics (reverb, no wind, close-range propeller noise) differ significantly from outdoor field conditions. Training on indoor data for an outdoor detector risks distribution shift. |

## Preliminary Recommendation

**REJECT** because:

1. **No license.** The repository has no LICENSE file. The GitHub API reports no license. The README asks for citation but provides no usage grant. Under default copyright, all rights are reserved by the authors. Ingesting unlicensed data into a training pipeline is a legal risk.

2. **Indoor recording environment is a poor operational match.** Our detector operates with a UMA-16v2 outdoor array. Indoor drone recordings capture close-range propeller noise with room reverb, no wind noise, and controlled acoustic conditions. Training on this data could degrade outdoor detection performance through distribution mismatch.

3. **Negative class overlaps with existing ESC-50 corpus.** The "unknown" class is built from ESC-50 clips we already have. Ingesting would double-count these negatives unless de-duplicated, skewing class balance.

4. **Only 2 drone models (Bebop, Membo).** Limited diversity compared to DADS + our field recordings. Marginal generalization benefit.

5. **Phase 22 scope risk.** Adding dataset ingestion (license clearance, de-duplication, manifest update, class balance re-check) to an already 9-plan phase delays the primary goal: fixing the v7 window-contract regression.

## Open Items for Human Review

- **License:** This is the primary blocker. If the user has a relationship with the authors or can obtain explicit permission, the license concern is resolved. Otherwise, this is a hard stop.
- **Operational fit:** User should decide whether indoor Bebop/Membo data adds value for outdoor UMA-16 detection or risks harming it.
- **Alternative use:** Even if rejected for training, the dataset could be useful as a "challenging distribution" eval set to test how the detector handles out-of-distribution (indoor) drone audio. This does not require a training data license -- only an evaluation license.
- **Preferred split if ingested:** Training data, supplementary eval set, or both?
