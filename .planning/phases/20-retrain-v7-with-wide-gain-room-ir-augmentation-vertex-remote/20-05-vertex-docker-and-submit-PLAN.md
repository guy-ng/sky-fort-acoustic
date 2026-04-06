---
phase: 20
plan: 05
type: execute
wave: 4
depends_on:
  - "20-00"
  - "20-04"
files_modified:
  - Dockerfile.vertex
  - Dockerfile.vertex-base
  - scripts/vertex_submit.py
  - scripts/vertex_train.py
autonomous: false
requirements:
  - D-21
  - D-22
  - D-23
  - D-24
  - D-25
must_haves:
  truths:
    - "Dockerfile.vertex (or its base image) COPies data/noise/ and data/field/uma16_ambient/ into /app/data/"
    - "vertex_submit.py exposes build_env_vars_v7() containing all Phase 20 ACOUSTIC_TRAINING_* env vars with locked values"
    - "vertex_submit.py preflights L4 quota and falls back to NVIDIA_TESLA_T4 if denied"
    - "v7 job name contains 'v7' and AIP_MODEL_DIR points to gs://sky-fort-acoustic/models/vertex/efficientat_mn10_v7/"
    - "vertex_train.py reads new Phase 20 env vars and constructs TrainingConfig correctly"
    - "Two-image strategy: Dockerfile.vertex-base bakes data + deps; Dockerfile.vertex layers code on top"
  artifacts:
    - path: Dockerfile.vertex-base
      provides: "Base image with pyroomacoustics + noise + ambient data baked in"
      contains: "COPY data/noise"
    - path: Dockerfile.vertex
      provides: "Phase image with code, FROM the base image"
      contains: "FROM"
    - path: scripts/vertex_submit.py
      provides: "build_env_vars_v7() + check_l4_quota() + v7 job submission"
      contains: "build_env_vars_v7"
  key_links:
    - from: scripts/vertex_submit.py
      to: scripts/vertex_train.py
      via: "env_vars dict propagated to CustomContainerTrainingJob.run via aiplatform SDK"
      pattern: "env_vars"
    - from: Dockerfile.vertex
      to: Dockerfile.vertex-base
      via: "FROM us-central1-docker.pkg.dev/.../acoustic-trainer-base:v1"
      pattern: "FROM .*acoustic-trainer-base"
---

<objective>
Add the two-image Docker strategy (base image bakes ~3-4 GB of noise + ambient data once;
phase image FROM base layers only source code, keeping rebuilds <50 MB), extend vertex_submit.py
to (a) emit all Phase 20 ACOUSTIC_TRAINING_* env vars per D-23/D-24, (b) preflight L4 quota and
fall back to T4 per D-22, (c) construct the v7 job name and AIP_MODEL_DIR per D-25; and update
vertex_train.py if any new env-var plumbing is needed.

Purpose: Phase 20 mandates remote-only training (D-21). Without this plan, the trained
augmentations from Plans 01-04 cannot reach a GPU. Without the two-image strategy, every code
iteration would push 5-10 GB to GCR — killing iteration speed. The L4→T4 fallback is mandatory
because L4 quota is not verified for the project (Research Pitfall 5).

Output: Two Dockerfiles, updated vertex_submit.py with Phase 20 env vars + quota check, manual
checkpoint for the actual Vertex job submission and checkpoint download.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-CONTEXT.md
@.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
@scripts/vertex_submit.py
@scripts/vertex_train.py
@Dockerfile.vertex
@requirements-vertex.txt
@tests/unit/test_vertex_submit_phase20.py
@tests/integration/test_vertex_dockerfile_copy.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Two-image Dockerfile + vertex_submit.py Phase 20 env vars + L4/T4 fallback</name>
  <files>
    Dockerfile.vertex-base,
    Dockerfile.vertex,
    scripts/vertex_submit.py,
    scripts/vertex_train.py
  </files>
  <read_first>
    Dockerfile.vertex,
    requirements-vertex.txt,
    scripts/vertex_submit.py,
    scripts/vertex_train.py,
    tests/unit/test_vertex_submit_phase20.py,
    tests/integration/test_vertex_dockerfile_copy.py,
    .planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-RESEARCH.md
  </read_first>
  <behavior>
    After this task:
    - Dockerfile.vertex-base exists; layers: pytorch base → apt libsndfile1 → pip install requirements-vertex.txt → COPY data/noise + data/field/uma16_ambient
    - Dockerfile.vertex FROMs the base image and only COPies src/ + scripts/vertex_train.py
    - scripts/vertex_submit.py has:
        * `build_env_vars_v7() -> dict[str, str]` with all Phase 20 keys (verbatim from Research §"Vertex Submission for v7 Run")
        * `check_l4_quota(project: str, region: str) -> bool` that runs `gcloud compute regions describe ... --format=...` and parses NVIDIA_L4 quota; returns True if >0
        * v7 job submission path that uses NVIDIA_L4 if `check_l4_quota` returns True, else falls back to NVIDIA_TESLA_T4 with a logged warning
        * Job name containing "v7"
        * AIP_MODEL_DIR = `gs://sky-fort-acoustic/models/vertex/efficientat_mn10_v7/`
    - scripts/vertex_train.py: if any new env var requires explicit reading, add it; otherwise pydantic-settings on TrainingConfig handles it automatically
    - Wave 0 RED tests in test_vertex_submit_phase20.py and test_vertex_dockerfile_copy.py turn GREEN
  </behavior>
  <action>
    Step 1 — Create Dockerfile.vertex-base at the repo root with this content:

    ```dockerfile
    # Phase 20 base image: bakes pyroomacoustics + noise + ambient data once.
    # Pushed once per data refresh; rebuilt only when requirements-vertex.txt or
    # data/noise/* changes. Phase 20 v7 code rebuilds layer on top of this image.
    FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

    RUN apt-get update \
        && apt-get install -y --no-install-recommends libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

    WORKDIR /app

    COPY requirements-vertex.txt requirements-vertex.txt
    RUN pip install --no-cache-dir -r requirements-vertex.txt

    # Bake noise + ambient data into the base layer (~3-4 GB, pushed once)
    COPY data/noise /app/data/noise
    COPY data/field/uma16_ambient /app/data/field/uma16_ambient
    ```

    Step 2 — Rewrite Dockerfile.vertex to FROM the base image and layer only code on top:

    ```dockerfile
    # Phase 20 image: code on top of acoustic-trainer-base.
    # Rebuild on every code change; push is <50 MB because base layer is cached.
    ARG BASE_IMAGE=us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v1
    FROM ${BASE_IMAGE}

    WORKDIR /app

    COPY src/ src/
    COPY scripts/vertex_train.py vertex_train.py

    ENV PYTHONPATH=/app/src
    ENV PYTHONUNBUFFERED=1

    ENTRYPOINT ["python", "vertex_train.py"]
    ```

    Step 3 — Open scripts/vertex_submit.py and add (do not break existing v6 submission paths):

    ```python
    import subprocess

    def check_l4_quota(project: str, region: str = "us-central1") -> bool:
        """Pre-flight L4 GPU quota check (Phase 20 D-22, Research Pitfall 5).

        Returns True if NVIDIA_L4 quota in the region is > 0, else False.
        Falls back to True (optimistic) if gcloud is unavailable so local
        development without gcloud installed still works.
        """
        try:
            result = subprocess.run(
                [
                    "gcloud", "compute", "regions", "describe", region,
                    "--project", project,
                    "--format=value(quotas)",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return False
            return "NVIDIA_L4" in result.stdout and "limit: 0" not in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def build_env_vars_v7(
        hf_repo: str = "geronimobasso/drone-audio-detection-samples",
        gcs_pretrained: str = "",
    ) -> dict[str, str]:
        """Phase 20 v7 ACOUSTIC_TRAINING_* env vars (D-01..D-25).
        Values are LOCKED by CONTEXT.md and must not drift.
        """
        return {
            # Source + model
            "ACOUSTIC_TRAINING_DADS_HF_REPO": hf_repo,
            "ACOUSTIC_TRAINING_MODEL_TYPE": "efficientat_mn10",
            # Three-stage recipe (D-23)
            "ACOUSTIC_TRAINING_BATCH_SIZE": "64",
            "ACOUSTIC_TRAINING_LOSS_FUNCTION": "focal",
            "ACOUSTIC_TRAINING_FOCAL_ALPHA": "0.25",
            "ACOUSTIC_TRAINING_FOCAL_GAMMA": "2.0",
            "ACOUSTIC_TRAINING_PATIENCE": "7",
            "ACOUSTIC_TRAINING_STAGE1_EPOCHS": "10",
            "ACOUSTIC_TRAINING_STAGE2_EPOCHS": "15",
            "ACOUSTIC_TRAINING_STAGE3_EPOCHS": "20",
            "ACOUSTIC_TRAINING_STAGE1_LR": "1e-3",
            "ACOUSTIC_TRAINING_STAGE2_LR": "1e-4",
            "ACOUSTIC_TRAINING_STAGE3_LR": "1e-5",
            # Phase 20 augmentation knobs (D-01..D-08)
            "ACOUSTIC_TRAINING_WIDE_GAIN_DB": "40.0",
            "ACOUSTIC_TRAINING_RIR_ENABLED": "true",
            "ACOUSTIC_TRAINING_RIR_PROBABILITY": "0.7",
            # Phase 20 BG noise (D-10, D-18, D-20)
            "ACOUSTIC_TRAINING_NOISE_AUGMENTATION_ENABLED": "true",
            "ACOUSTIC_TRAINING_NOISE_DIRS": '["/app/data/noise/esc50","/app/data/noise/urbansound8k","/app/data/noise/fsd50k_subset","/app/data/field/uma16_ambient"]',
            "ACOUSTIC_TRAINING_UMA16_AMBIENT_DIR": "/app/data/field/uma16_ambient",
            "ACOUSTIC_TRAINING_UMA16_AMBIENT_SNR_LOW": "-5.0",
            "ACOUSTIC_TRAINING_UMA16_AMBIENT_SNR_HIGH": "15.0",
            "ACOUSTIC_TRAINING_UMA16_AMBIENT_PURE_NEGATIVE_RATIO": "0.10",
            # Sliding window (D-13)
            "ACOUSTIC_TRAINING_WINDOW_OVERLAP_RATIO": "0.6",
            # Output (D-25)
            "ACOUSTIC_TRAINING_CHECKPOINT_PATH": "/tmp/models/efficientat_mn10_v7.pt",
            "ACOUSTIC_TRAINING_PRETRAINED_GCS": gcs_pretrained,
        }
    ```

    Then in the existing job submission code path, add a new function or branch (e.g.,
    `submit_v7_job(project, region, ...)`) that:
    - Calls `check_l4_quota(project, region)`
    - Picks `accelerator_type = "NVIDIA_L4"` if True else `"NVIDIA_TESLA_T4"` (logs the choice)
    - Sets `machine_type = "g2-standard-8"`
    - Sets `display_name = "efficientat-mn10-v7-phase20"` or similar containing "v7"
    - Sets `base_output_dir = "gs://sky-fort-acoustic/models/vertex/efficientat_mn10_v7/"`
    - Passes `env_vars=build_env_vars_v7(...)` to the CustomContainerTrainingJob.run call

    Step 4 — Open scripts/vertex_train.py. If TrainingConfig pydantic-settings already reads
    `ACOUSTIC_TRAINING_*` env vars automatically (it should after Plan 02), no changes needed.
    Verify by reading the file and confirming `TrainingConfig()` (no kwargs) is the construction
    pattern. If the script hard-codes specific env var reads, ADD reads for the new Phase 20 env
    vars and pass them through. Add a startup log line that prints the active augmentation chain:
    `print(f"[v7] wide_gain_db={cfg.wide_gain_db} rir_enabled={cfg.rir_enabled} window_overlap={cfg.window_overlap_ratio}")`.
  </action>
  <verify>
    <automated>pytest tests/unit/test_vertex_submit_phase20.py tests/integration/test_vertex_dockerfile_copy.py -x -q && grep -q "FROM " Dockerfile.vertex-base && grep -q "FROM " Dockerfile.vertex</automated>
  </verify>
  <acceptance_criteria>
    - File Dockerfile.vertex-base exists with `COPY data/noise` and `COPY data/field/uma16_ambient` lines
    - File Dockerfile.vertex starts with `FROM` referencing the base image (use ARG BASE_IMAGE for parameterization)
    - `grep -n "def build_env_vars_v7" scripts/vertex_submit.py` returns one match
    - `grep -n "def check_l4_quota" scripts/vertex_submit.py` returns one match
    - `grep -n "NVIDIA_TESLA_T4" scripts/vertex_submit.py` returns at least one match (fallback path present)
    - `grep -n "ACOUSTIC_TRAINING_WIDE_GAIN_DB" scripts/vertex_submit.py` returns one match with value "40.0"
    - `grep -n "ACOUSTIC_TRAINING_RIR_ENABLED" scripts/vertex_submit.py` returns one match with value "true"
    - `grep -n "ACOUSTIC_TRAINING_WINDOW_OVERLAP_RATIO" scripts/vertex_submit.py` returns one match with value "0.6"
    - `grep -n "v7" scripts/vertex_submit.py` returns at least one match in the v7 job definition
    - `pytest tests/unit/test_vertex_submit_phase20.py -x -q` exits 0
    - `pytest tests/integration/test_vertex_dockerfile_copy.py -x -q` exits 0
  </acceptance_criteria>
  <done>
    Two-image Dockerfile strategy in place; build_env_vars_v7 + check_l4_quota implemented;
    L4/T4 fallback wired; v7 job name + AIP_MODEL_DIR set; Wave 0 vertex tests GREEN.
  </done>
</task>

<task type="checkpoint:human-action" gate="blocking">
  <name>Task 2: Build base + phase image, push to GCR, submit v7 Vertex job, wait for completion, download checkpoint</name>
  <what-built>
    Two-image Dockerfile strategy + vertex_submit.py with v7 env vars and L4/T4 fallback.
    The actual `docker build`/`docker push`/Vertex job submission has multiple human-required
    moments (gcloud auth, reading job logs, deciding when to abort/retry, downloading the
    checkpoint to the local models/ directory). Per D-26 and D-27, this is the PHASE GATE for
    the training job — no eval can run until it completes.
  </what-built>
  <how-to-verify>
    All commands below run from the repo root.

    1. Authenticate to GCP and configure docker:
       ```
       gcloud auth login
       gcloud auth configure-docker us-central1-docker.pkg.dev
       ```

    2. Build and push the BASE image (do this only once per data refresh):
       ```
       docker build -f Dockerfile.vertex-base -t us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v1 .
       docker push us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v1
       ```
       Expected: Push completes; image visible in Artifact Registry.

    3. Build and push the PHASE 20 image (rebuilt per code change):
       ```
       docker build -f Dockerfile.vertex -t us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer:phase20 .
       docker push us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer:phase20
       ```
       Expected: Push completes in <2 min (small layer above cached base).

    4. Pre-flight quota check (manual; vertex_submit.py also runs this):
       ```
       python -c "from scripts.vertex_submit import check_l4_quota; print('L4 available:', check_l4_quota('interception-dashboard', 'us-central1'))"
       ```

    5. Submit the v7 job:
       ```
       python scripts/vertex_submit.py --version v7 --image us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer:phase20
       ```
       (Adapt the CLI to match the existing vertex_submit.py interface — the function added in
       Task 1 is the canonical entry point.)

    6. Tail the job logs in the GCP console or with `gcloud ai custom-jobs stream-logs <JOB_ID>`.
       Expected: Three stages run (10 + 15 + 20 epochs ≤45 total). Wall time ≈ 4-7 hours on L4,
       6-12 hours on T4. The job logs `[v7] wide_gain_db=40.0 rir_enabled=True window_overlap=0.6`
       at startup. EarlyStopping (patience=7) may terminate stages early.

    7. After job succeeds, download the checkpoint:
       ```
       gsutil cp gs://sky-fort-acoustic/models/vertex/efficientat_mn10_v7/best_model.pt models/efficientat_mn10_v7.pt
       sha256sum models/efficientat_mn10_v7.pt  # record for D-29 promotion gate verification
       ```

    8. Verify the checkpoint loads:
       ```
       python -c "import torch; ck = torch.load('models/efficientat_mn10_v7.pt', map_location='cpu', weights_only=False); print(type(ck), list(ck.keys()) if isinstance(ck, dict) else 'state')"
       ```
  </how-to-verify>
  <resume-signal>
    Type "v7 trained" when models/efficientat_mn10_v7.pt exists locally and torch.load succeeds.
    Type "v7 failed: <reason>" if the Vertex job fails — the orchestrator will route to gap closure.
  </resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| GCP service account → Vertex AI | Vertex job submission uses Application Default Credentials. |
| GCR (Artifact Registry) → Vertex worker | Docker images pulled by Vertex worker; supply chain. |
| GCS → local | Trained checkpoint downloaded from GCS to models/ directory. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-05-01 | Spoofing | GCP credentials | mitigate | Use Application Default Credentials (`gcloud auth login`); never commit service account JSON. CLAUDE.md security rules already enforce this. |
| T-20-05-02 | Tampering | Base image supply chain | mitigate | Pin base image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (specific tag). Future hardening: pin by digest (`@sha256:...`) — note for follow-up phase. |
| T-20-05-03 | Tampering | Trained checkpoint authenticity (D-29 input) | mitigate | Record sha256 of downloaded checkpoint at acquisition time (Task 2 step 7). Promotion gate (Plan 06) re-verifies by hash before copying to default model path. |
| T-20-05-04 | Information Disclosure | Vertex job env vars | mitigate | Phase 20 env vars contain no secrets (only training hyperparameters). GCS bucket and HF repo are public/project-controlled. |
| T-20-05-05 | DoS | L4 quota denial → stuck PENDING job | mitigate | Pre-flight check_l4_quota() + automatic T4 fallback (D-22). |
</threat_model>

<verification>
- Wave 0 vertex_submit + dockerfile tests pass after Task 1
- Manual checkpoint Task 2 verifies the actual training job ran and produced the checkpoint
- Checkpoint sha256 recorded for use by Plan 06 promotion gate
</verification>

<success_criteria>
- `pytest tests/unit/test_vertex_submit_phase20.py tests/integration/test_vertex_dockerfile_copy.py -x -q` exits 0
- `test -f Dockerfile.vertex-base && test -f Dockerfile.vertex`
- `grep -c "build_env_vars_v7\|check_l4_quota" scripts/vertex_submit.py` >= 2
- After Task 2: `test -f models/efficientat_mn10_v7.pt` and torch.load succeeds
</success_criteria>

<output>
After completion, create `.planning/phases/20-retrain-v7-with-wide-gain-room-ir-augmentation-vertex-remote/20-05-SUMMARY.md`
including the recorded sha256 of the v7 checkpoint for Plan 06 to verify against.
</output>
