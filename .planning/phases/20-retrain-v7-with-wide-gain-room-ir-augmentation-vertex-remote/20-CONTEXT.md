# Phase 20: Retrain v7 with Wide Gain + Room-IR Augmentation, Vertex Remote - Context

**Gathered:** 2026-04-06 (auto mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Retrain EfficientAT mn10 to produce `efficientat_mn10_v7.pt` that generalizes to real UMA-16v2 captures. Scope:

1. **Augmentation upgrade** — ±40 dB random gain, room impulse-response (RIR) convolution, UMA-16 ambient noise injection.
2. **Data pipeline upgrade** — Replace random 0.5s segment extraction with 60%-overlap sliding windows. Expand BG negative set with noise sources from `docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md`.
3. **Remote training** — Execute on Vertex AI via existing `scripts/vertex_submit.py` + `scripts/vertex_train.py` infrastructure, not locally.
4. **Deliverable** — `models/efficientat_mn10_v7.pt` checkpoint with measurable improvement on real UMA-16 captures (vs v6 baseline).

Out of scope: model architecture changes, ONNX/TFLite export (Phase 16), pipeline integration changes, new datasets beyond those called out in the referenced compass doc, DOA/beamforming changes.

</domain>

<decisions>
## Implementation Decisions

### Gain Augmentation (wide-range)
- **D-01:** Replace current `waveform_gain_db=6.0` with wide-range gain ±40 dB, uniform distribution, applied per-sample during training. New config key `wide_gain_db: float = 40.0`.
- **D-02:** Apply wide gain BEFORE background noise mixing and BEFORE RIR convolution so the model sees the full deployment gain range with matching noise floor. Clip to [-1, 1] after mixing.
- **D-03:** Justification anchored on debug evidence: UMA-16v2 mono RMS ≈ 8e-5 (−82 dBFS) vs DADS training data ≈ −20 to −30 dBFS. Gap is ~50–60 dB. ±40 dB gain covers that gap with margin.
- **D-04:** Keep existing `audiomentations.Gain ±6 dB` as a secondary small-jitter augmentation inside the audiomentations Compose (stacked, not replaced). The ±40 dB wide gain runs as a separate stage before audiomentations.

### Room Impulse-Response (RIR) Convolution
- **D-05:** Use `pyroomacoustics` procedural RIR generation (not real recorded IR datasets) — zero dataset download cost, reproducible, parameterizable. Add `pyroomacoustics>=0.7` to training requirements.
- **D-06:** Per-sample random room parameters: room dimensions 3×3×2.5 m to 12×12×4 m (uniform), absorption coefficient 0.2–0.7 (uniform wall/ceiling/floor), source-mic distance 1–8 m, RT60 implied by absorption. One mic position, one source position per sample.
- **D-07:** New class `RoomIRAugmentation` in `src/acoustic/training/augmentation.py`. Applied AFTER wide gain and BEFORE BG noise mixing (reverb first, then environmental noise). Probability `rir_probability=0.7` (default).
- **D-08:** Disable RIR augmentation on val/test splits — train augmentation only. Eval metrics must reflect clean-ish inputs plus the dedicated real-UMA-16 eval set (D-20).

### UMA-16 Ambient Noise Negatives
- **D-09:** Collect ≥30 minutes of real UMA-16v2 ambient recordings (no drone present) before training begins. Conditions: indoor quiet, indoor HVAC on, outdoor quiet, outdoor wind. Store as mono 16 kHz WAV in `data/field/uma16_ambient/`.
- **D-10:** Integrate via existing `BackgroundNoiseMixer` (`src/acoustic/training/augmentation.py:120`). Add `data/field/uma16_ambient` to `noise_dirs` config list.
- **D-11:** Use dedicated SNR range for UMA-16 ambient: −5 to +15 dB (slightly tighter than ESC-50's −10 to +20) so the model learns to detect drones in the actual deployment noise floor.
- **D-12:** Also include UMA-16 ambient clips as pure negative samples (label=0) with some probability so the model explicitly learns "UMA-16 ambient alone = no drone". Target ~10% of negative mini-batch sourced from UMA-16 ambient files.

### 60% Overlap Sliding Windows
- **D-13:** Replace random 0.5s segment extraction in `HFDroneDataset.__getitem__` (`src/acoustic/training/hf_dataset.py:70-77`) with deterministic sliding-window enumeration. For each source recording, emit windows at hop = 40% of window length (i.e., 0.2s hop for 0.5s window = 60% overlap).
- **D-14:** Re-index the dataset: `__len__` returns total window count across all split files. `__getitem__(idx)` maps idx → (file_idx, window_offset). Keep per-file label from Arrow column cache.
- **D-15:** **Session-level split preservation is MANDATORY** — overlapping windows from the same source file MUST all go into the same split (train/val/test). Update `split_indices` logic in `parquet_dataset.py:split_indices` to operate on file indices, not window indices, then expand to window indices post-split.
- **D-16:** Apply overlap to train and val only. Test set uses non-overlapping windows (hop = window length) to keep eval metrics unbiased by window count inflation.

### Expanded Background Noise Set
- **D-17:** Keep ESC-50 + UrbanSound8K (Phase 15 baseline).
- **D-18:** Add FSD50K subset covering `Wind`, `Rain`, `Traffic_noise_and_roadway_noise`, `Mechanical_fan`, `Engine`, `Bird` classes — download via `soundata` or manual FSD50K fetch, store in `data/noise/fsd50k_subset/`.
- **D-19:** Add DroneAudioSet non-drone / hard-negative clips if freely available from `augmented-human-lab/DroneAudioSet-code` per compass doc §1.
- **D-20:** New config key `noise_dirs` auto-populates with `[esc50, urbansound8k, fsd50k_subset, uma16_ambient]` when `noise_augmentation_enabled=True` for Phase 20 training runs.

### Vertex Remote Training
- **D-21:** Use existing `scripts/vertex_submit.py` + `scripts/vertex_train.py` + `Dockerfile.vertex`. Do NOT train locally. Local runs are for smoke-testing the augmentation pipeline only.
- **D-22:** Machine: `g2-standard-8` with NVIDIA L4 (1 GPU). Fallback to `NVIDIA_TESLA_T4` if L4 quota is denied.
- **D-23:** Hyperparameters (three-stage EfficientAT recipe, Phase 14 convention): `stage1_epochs=10 @ 1e-3`, `stage2_epochs=15 @ 1e-4`, `stage3_epochs=20 @ 1e-5`, `batch_size=64`, `loss=focal(α=0.25, γ=2.0)`, `patience=7`. Total ≤45 epochs.
- **D-24:** HF dataset source unchanged: `geronimobasso/drone-audio-detection-samples` (DADS). Noise / RIR / UMA-16 ambient directories bundled into the Docker image under `/app/data/noise/` and `/app/data/field/uma16_ambient/` (update `Dockerfile.vertex` COPY layer).
- **D-25:** Output: best checkpoint uploaded to `gs://sky-fort-acoustic/models/vertex/efficientat_mn10_v7/best_model.pt` via existing `AIP_MODEL_DIR` flow. Download locally to `models/efficientat_mn10_v7.pt` after job succeeds.

### Evaluation / Success Metric
- **D-26:** v7 must maintain ≥95% accuracy on DADS test split (no regression vs v6).
- **D-27:** v7 must achieve TPR ≥ 0.80 and FPR ≤ 0.05 on a new real-capture eval set: ≥20 minutes of UMA-16v2 recordings with labeled drone/no-drone segments (≥5 min drone present, ≥15 min ambient). Store at `data/eval/uma16_real/` with `labels.json`.
- **D-28:** Eval harness: extend existing evaluation tooling (Phase 9) to accept a UMA-16 eval set. Produces a confusion matrix + ROC curve report saved alongside the v7 checkpoint.
- **D-29:** Promotion rule: v7 is promoted to `models/efficientat_mn10.pt` (default) ONLY if both D-26 AND D-27 pass. Otherwise keep v6 as default and investigate.

### Claude's Discretion
- Exact pyroomacoustics room shape sampler implementation (uniform vs log-uniform per dimension).
- RIR caching strategy (precompute pool of 500 RIRs vs generate per-sample).
- Whether to use `torchaudio.functional.fftconvolve` or numpy `scipy.signal.fftconvolve` for RIR application.
- Exact FSD50K class-slug → filename glob mapping.
- Number of DataLoader workers on Vertex L4 (4 or 8).
- Docker image layer caching strategy for the new noise/ambient data directories.

### Folded Todos
None.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Augmentation and training research
- `docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md` §1 — Background noise / pretraining datasets table, DroneAudioSet, ESC-50, UrbanSound8K, FSD50K source list
- `docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md` §4 Augmentation pipeline (ordered by impact) — confirms BG noise + RIR + wide gain as top augmentations
- `docs/compass_artifact_wf-6c2ec688-1122-4ac5-898e-12ac7039d309_text_markdown.md` §4 "Data splitting: session-level grouping is non-negotiable" — reinforces D-15

### Root-cause evidence for wide gain
- `.planning/debug/uma16-no-detections.md` — Documents UMA-16 ambient RMS ≈ 8e-5 (−82 dBFS), vs DADS training data ~ −20 to −30 dBFS. Justifies D-01..D-04.

### Existing training infrastructure (modify, don't replace)
- `src/acoustic/training/augmentation.py` — `WaveformAugmentation`, `SpecAugment`, `BackgroundNoiseMixer`, `AudiomentationsAugmentation`, `ComposedAugmentation`. Phase 20 adds `RoomIRAugmentation` + widens gain.
- `src/acoustic/training/hf_dataset.py` §`__getitem__` — Random-segment extraction (lines 70-77) must become sliding-window enumeration per D-13/D-14.
- `src/acoustic/training/parquet_dataset.py` §`split_indices` — Session-level split logic; must operate on file indices per D-15.
- `src/acoustic/training/config.py` — `TrainingConfig` pydantic settings; add `wide_gain_db`, `rir_enabled`, `rir_probability`, `window_hop_ratio` fields.
- `src/acoustic/training/efficientat_trainer.py` — Three-stage EfficientAT training runner (Phase 14); unchanged except reads new config fields.
- `scripts/vertex_submit.py` — Vertex AI job submission; unchanged interface, new job name for v7.
- `scripts/vertex_train.py` — Vertex container entry point; unchanged unless env var plumbing is needed for new config fields.
- `Dockerfile.vertex` — Must COPY the new noise/ambient data directories into the image.

### Prior phase context that carries forward
- `.planning/phases/13-dads-dataset-integration-and-training-data-pipeline/13-CONTEXT.md` — DADS dataset integration, session-level split rule
- `.planning/phases/14-*/...` (no CONTEXT.md, see ROADMAP.md §Phase 14) — EfficientAT mn10 architecture, three-stage unfreezing recipe
- `.planning/phases/15-*/...` (no CONTEXT.md, see ROADMAP.md §Phase 15) — Focal loss, BG noise mixer baseline, audiomentations integration
- `.planning/ROADMAP.md` §Phase 20 (lines 395-404) — Phase goal definition

### External library docs (planner should consult via Context7 or web)
- `pyroomacoustics` — Room / ShoeBox / RoomImpulseResponse API for D-05..D-08
- `audiomentations` — Already vendored; no changes expected
- `torchaudio.functional.fftconvolve` — Candidate for GPU-side RIR application

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BackgroundNoiseMixer` (`augmentation.py:120`) — scans noise dirs, caches, mixes at random SNR. Already supports multiple directories — adding FSD50K + UMA-16 ambient is a config-only change.
- `AudiomentationsAugmentation` (`augmentation.py:204`) — existing Compose chain; keep the ±6 dB Gain step for small jitter, add the wide-gain stage as a separate pre-stage.
- `ComposedAugmentation` (`augmentation.py:251`) — pickle-safe chain class. Use it to sequence: WideGain → RoomIR → AudiomentationsAugmentation → BackgroundNoiseMixer.
- `HFDroneDataset` (`hf_dataset.py:25`) — memory-mapped Arrow backing, already label-cached. Modify `__len__` and `__getitem__` for sliding windows; keep the Arrow fast-path.
- `EfficientATTrainingRunner` — referenced by `vertex_train.py:254`. Three-stage recipe already in place; Phase 20 only changes config, not the runner.
- `scripts/vertex_submit.py` / `scripts/vertex_train.py` — complete remote training pipeline. No architectural changes needed.
- Existing field recording UI (Phase 12) — usable for collecting the UMA-16 ambient set (D-09) and the real-UMA-16 eval set (D-27).

### Established Patterns
- Config-driven behavior: all training knobs via `ACOUSTIC_TRAINING_*` env vars through pydantic `TrainingConfig`. Phase 20 additions follow the same pattern.
- Session-level split enforced at the file level, not sample level (`parquet_dataset.py:split_indices`). Sliding-window change must preserve this invariant.
- Model versioning: `efficientat_mn10_v{N}.pt` in `models/`. v6 is current latest. v7 is this phase's target.
- Remote-only training: Vertex is the training surface; local runs are smoke tests. Dockerfile.vertex is the source of truth for the training environment.

### Integration Points
- `BackgroundNoiseMixer` consumes `noise_dirs: list[str]` from config — extend config list without mixer code changes.
- `HFDroneDataset` is instantiated by `EfficientATTrainingRunner` — ensure runner passes the new composed augmentation and uses the new `__len__`.
- `Dockerfile.vertex` COPY layer — add `data/noise/fsd50k_subset/` and `data/field/uma16_ambient/` so the image ships with training data.
- `models/` directory for checkpoint promotion (D-29). `main.py` loads default model path; v7 promotion is a file rename, not a code change.

</code_context>

<specifics>
## Specific Ideas

- User's phase description explicitly calls out ±40 dB, room-IR, UMA-16 ambient, 60% overlap, remote Vertex, compass doc negatives — this CONTEXT.md locks those as hard requirements, not suggestions.
- Debug doc `.planning/debug/uma16-no-detections.md` is the PROOF that v6 never trained on signal as quiet as real UMA-16 ambient. Wide gain augmentation is the direct remediation.
- "Measurable improvement on real captures" is quantified as D-27 (TPR ≥ 0.80, FPR ≤ 0.05 on ≥20 min real UMA-16 eval set). Planner should not re-open this metric.

</specifics>

<deferred>
## Deferred Ideas

- Real recorded RIR datasets (MIT IR Survey, BUT Reverb, OpenAIR) — deferred in favor of procedural pyroomacoustics for this phase. Could revisit in a future phase if sim-to-real gap is observed.
- Architectural changes to EfficientAT or switching to ResNet-Mamba hybrid (compass doc §3.E) — out of scope for Phase 20.
- ONNX/TFLite export of v7 — belongs to Phase 16.
- Automated hyperparameter sweep on Vertex — out of scope; use fixed three-stage recipe.
- Online / continuous learning from deployed UMA-16 captures — future milestone.
- Doppler / range-based targets — separate domain.

### Reviewed Todos (not folded)
None — no pending todos matched this phase (cross_reference_todos skipped in auto mode because no matches).

</deferred>

---

*Phase: 20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote*
*Context gathered: 2026-04-06 (auto mode)*
