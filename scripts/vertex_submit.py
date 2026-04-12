"""Submit a Vertex AI custom training job for Sky Fort Acoustic.

Builds the Docker image, pushes to GCR, and submits the training job.
Training data is loaded directly from HuggingFace — no GCS data upload needed.

Prerequisites:
  1. gcloud CLI authenticated: gcloud auth login
  2. Docker authenticated: gcloud auth configure-docker
  3. Vertex AI API enabled: gcloud services enable aiplatform.googleapis.com

Usage:
  # Train ResearchCNN (default, T4 GPU)
  python scripts/vertex_submit.py

  # Train EfficientAT with custom hyperparameters
  python scripts/vertex_submit.py \
    --model-type efficientat_mn10 \
    --epochs 50 \
    --batch-size 64 \
    --gpu-type NVIDIA_TESLA_T4

  # Use a bigger GPU
  python scripts/vertex_submit.py --gpu-type NVIDIA_L4 --machine-type g2-standard-8

  # Dry run (build + push image only)
  python scripts/vertex_submit.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone


# --- GCP Configuration ---
GCP_PROJECT = "interception-dashboard"
GCP_REGION = "us-central1"
GCP_REGION_V8 = "us-east1"  # Phase 22 (user-locked)
GCS_BUCKET = "sky-fort-acoustic"
GCR_IMAGE = f"us-central1-docker.pkg.dev/{GCP_PROJECT}/acoustic-training/acoustic-trainer"
GCR_BASE_IMAGE = (
    f"us-central1-docker.pkg.dev/{GCP_PROJECT}/acoustic-training/acoustic-trainer-base:v1"
)

# HuggingFace dataset (no GCS upload needed)
DEFAULT_HF_REPO = "geronimobasso/drone-audio-detection-samples"

# GCS paths for model output and optional pretrained weights
GCS_MODEL_OUTPUT = f"gs://{GCS_BUCKET}/models/vertex/"
GCS_PRETRAINED = f"gs://{GCS_BUCKET}/models/pretrained/mn10_as.pt"

# Phase 20 v7 output directory (D-25, locked by CONTEXT.md)
GCS_V7_MODEL_DIR = f"gs://{GCS_BUCKET}/models/vertex/efficientat_mn10_v7/"


def check_l4_quota(project: str, region: str = "us-central1") -> bool:
    """Pre-flight L4 GPU quota check (Phase 20 D-22, Research Pitfall 5).

    Returns True if NVIDIA_L4 quota in the region appears > 0, else False.
    Returns False if gcloud is unavailable so the caller falls back to T4
    (safer default than optimistically assuming L4 when we can't verify).
    """
    try:
        result = subprocess.run(
            [
                "gcloud", "compute", "regions", "describe", region,
                "--project", project,
                "--format=value(quotas)",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    stdout = result.stdout or ""
    if "NVIDIA_L4" not in stdout:
        return False
    # gcloud renders quotas as "limit: 0.0" when denied.
    return "limit: 0" not in stdout and "limit: 0.0" not in stdout


def build_env_vars_v7(
    hf_repo: str = DEFAULT_HF_REPO,
    gcs_pretrained: str = GCS_PRETRAINED,
) -> dict[str, str]:
    """Phase 20 v7 job payload (D-01..D-25).

    Returns a FLAT dict containing both job metadata (job_name / display_name /
    machine_type / accelerator_type / fallback_accelerator_type / base_output_dir)
    and all ACOUSTIC_TRAINING_* env vars with values LOCKED by CONTEXT.md.

    Per D-22, ``accelerator_type`` is always the PLANNED choice (NVIDIA_L4 on
    g2-standard-8). The T4 fallback is declared here as ``fallback_accelerator_type``
    and is selected at submission time by ``submit_v7_job`` if ``check_l4_quota``
    returns False.
    """
    payload: dict[str, str] = {
        # --- Job metadata (Phase 20 D-25) ---
        "job_name": "efficientat-mn10-v7-phase20",
        "display_name": "efficientat-mn10-v7-phase20",
        "machine_type": "g2-standard-8",
        "accelerator_type": "NVIDIA_L4",
        "fallback_accelerator_type": "NVIDIA_TESLA_T4",
        "base_output_dir": GCS_V7_MODEL_DIR,
        # --- Source + model ---
        "ACOUSTIC_TRAINING_DADS_HF_REPO": hf_repo,
        "ACOUSTIC_TRAINING_MODEL_TYPE": "efficientat_mn10",
        # --- Three-stage recipe (D-23) ---
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
        # --- Phase 20 augmentation knobs (D-01..D-08) ---
        "ACOUSTIC_TRAINING_WIDE_GAIN_DB": "40.0",
        "ACOUSTIC_TRAINING_RIR_ENABLED": "true",
        "ACOUSTIC_TRAINING_RIR_PROBABILITY": "0.7",
        # --- Phase 20 BG noise (D-10, D-18, D-20) ---
        "ACOUSTIC_TRAINING_NOISE_AUGMENTATION_ENABLED": "true",
        "ACOUSTIC_TRAINING_NOISE_DIRS": (
            '["/app/data/noise/esc50","/app/data/noise/urbansound8k",'
            '"/app/data/noise/fsd50k_subset","/app/data/field/uma16_ambient"]'
        ),
        "ACOUSTIC_TRAINING_UMA16_AMBIENT_DIR": "/app/data/field/uma16_ambient",
        "ACOUSTIC_TRAINING_UMA16_AMBIENT_SNR_LOW": "-5.0",
        "ACOUSTIC_TRAINING_UMA16_AMBIENT_SNR_HIGH": "15.0",
        "ACOUSTIC_TRAINING_UMA16_AMBIENT_PURE_NEGATIVE_RATIO": "0.10",
        # --- Sliding window (D-13) ---
        "ACOUSTIC_TRAINING_WINDOW_OVERLAP_RATIO": "0.6",
        # --- Output (D-25) ---
        "ACOUSTIC_TRAINING_CHECKPOINT_PATH": "/tmp/models/efficientat_mn10_v7.pt",
        "ACOUSTIC_TRAINING_PRETRAINED_GCS": gcs_pretrained,
    }
    return payload


def submit_v7_job(image_uri: str, *, dry_run: bool = False) -> None:
    """Submit the Phase 20 v7 Vertex AI job (D-21..D-25).

    Uses ``build_env_vars_v7`` + ``check_l4_quota`` to pick the accelerator, then
    runs a CustomContainerTrainingJob with the locked Phase 20 hyperparameters.
    """
    payload = build_env_vars_v7()
    planned_accelerator = payload["accelerator_type"]
    fallback = payload["fallback_accelerator_type"]
    machine_type = payload["machine_type"]
    display_name = payload["display_name"]
    base_output_dir = payload["base_output_dir"]

    # D-22: preflight L4 quota, fall back to T4 if denied.
    if check_l4_quota(GCP_PROJECT, GCP_REGION):
        accelerator_type = planned_accelerator
    else:
        accelerator_type = fallback

    # Strip non-env metadata keys before handing the dict to Vertex.
    meta_keys = {
        "job_name",
        "display_name",
        "machine_type",
        "accelerator_type",
        "fallback_accelerator_type",
        "base_output_dir",
    }
    env_vars = {k: v for k, v in payload.items() if k not in meta_keys}

    if accelerator_type == "NVIDIA_L4":
        print(f">>> [v7] L4 quota available — submitting with NVIDIA_L4 on {machine_type}")
    else:
        print(
            f">>> [v7] L4 quota unavailable or unverifiable — FALLING BACK to {fallback} "
            f"(original plan: NVIDIA_L4 on {machine_type})"
        )

    print(f"\n{'='*60}")
    print(f"  Job:           {display_name}")
    print(f"  Image:         {image_uri}")
    print(f"  Machine:       {machine_type}")
    print(f"  Accelerator:   {accelerator_type}")
    print(f"  Output dir:    {base_output_dir}")
    print(f"  Env vars:      {len(env_vars)} ACOUSTIC_TRAINING_* keys")
    print(f"{'='*60}\n")

    if dry_run:
        print(">>> Dry run — not submitting job.")
        return

    from google.cloud import aiplatform

    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        staging_bucket=f"gs://{GCS_BUCKET}",
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=image_uri,
    )

    job.run(
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,
        environment_variables=env_vars,
        base_output_dir=base_output_dir,
        sync=False,
    )

    print(f"\n>>> v7 job submitted. Monitor: "
          f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={GCP_PROJECT}")
    print(f">>> Download checkpoint when done: gsutil cp {base_output_dir}best_model.pt "
          f"models/efficientat_mn10_v7.pt")


def build_env_vars_v8(
    output_dir: str,
    *,
    pretrained_v6_path: str = "/app/models/efficientat_mn10_v6.pt",
) -> dict[str, str]:
    """Phase 22 v8 env vars. Additive to v7; does not mutate build_env_vars_v7."""
    env = build_env_vars_v7()
    # Strip v7 metadata keys before updating
    meta_keys = {
        "job_name",
        "display_name",
        "machine_type",
        "accelerator_type",
        "fallback_accelerator_type",
        "base_output_dir",
    }
    env = {k: v for k, v in env.items() if k not in meta_keys}
    env.update({
        "ACOUSTIC_TRAINING_MODEL_TYPE": "efficientat_mn10",
        "ACOUSTIC_TRAINING_FINETUNE_FROM_TRAINED": "true",
        "ACOUSTIC_TRAINING_PRETRAINED_WEIGHTS": pretrained_v6_path,
        "ACOUSTIC_TRAINING_INCLUDE_FIELD_RECORDINGS": "true",
        "ACOUSTIC_TRAINING_RUN_DATA_PREFLIGHT": "true",
        "ACOUSTIC_TRAINING_FIELD_DRONE_DIR": "/app/data/field/drone",
        "ACOUSTIC_TRAINING_FIELD_BACKGROUND_DIR": "/app/data/field/background",
        "ACOUSTIC_TRAINING_WINDOW_OVERLAP_RATIO": "0.5",
        "ACOUSTIC_TRAINING_OUTPUT_PATH": f"{output_dir}/efficientat_mn10_v8.pt",
        "ACOUSTIC_TRAINING_CHECKPOINT_PATH": "/tmp/models/efficientat_mn10_v8.pt",
    })
    return env


# Default v8 base image URI (Phase 22 Dockerfile.vertex-base:v2)
GCR_BASE_IMAGE_V2 = (
    f"us-central1-docker.pkg.dev/{GCP_PROJECT}/acoustic-training/acoustic-trainer-base:v2"
)


def submit_v8_job(image_uri: str, *, dry_run: bool = False) -> str | None:
    """Phase 22: submit Vertex job for v8 training in us-east1 L4.

    Additive path -- does not touch submit_v7_job. Rollback = re-run v7.
    Returns the job resource name (or None on dry run).
    """
    assert check_l4_quota(GCP_PROJECT, GCP_REGION_V8), (
        f"L4 quota unavailable in {GCP_REGION_V8} -- request quota increase or "
        f"fall back to us-central1"
    )

    output_dir = f"gs://{GCS_BUCKET}/training/efficientat_v8"
    env = build_env_vars_v8(output_dir)

    print(f"\n{'='*60}")
    print(f"  Job:           efficientat-mn10-v8-phase22")
    print(f"  Image:         {image_uri}")
    print(f"  Region:        {GCP_REGION_V8}")
    print(f"  Machine:       g2-standard-8")
    print(f"  Accelerator:   NVIDIA_L4")
    print(f"  Output dir:    {output_dir}")
    print(f"  Env vars:      {len(env)} keys")
    print(f"{'='*60}\n")

    if dry_run:
        print(">>> Dry run -- not submitting job.")
        return None

    from google.cloud import aiplatform as aip

    aip.init(project=GCP_PROJECT, location=GCP_REGION_V8)
    job = aip.CustomContainerTrainingJob(
        display_name="efficientat-mn10-v8-phase22",
        container_uri=image_uri,
    )
    resource_name = job.run(
        replica_count=1,
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        environment_variables=env,
        base_output_dir=output_dir,
        sync=False,  # do not block -- Plan 08 task will poll
    )

    print(f"\n>>> v8 job submitted. Monitor: "
          f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
          f"?project={GCP_PROJECT}&region={GCP_REGION_V8}")
    return resource_name


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def build_and_push_image(tag: str) -> str:
    """Build Docker image and push to GCR."""
    image_uri = f"{GCR_IMAGE}:{tag}"
    print(f"\n>>> Building {image_uri}")
    run(["docker", "build", "-f", "Dockerfile.vertex", "-t", image_uri, "."])
    print(f"\n>>> Pushing {image_uri}")
    run(["docker", "push", image_uri])
    return image_uri


def submit_job(
    image_uri: str,
    model_type: str,
    gpu_type: str,
    gpu_count: int,
    machine_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_function: str,
    hf_repo: str,
    job_name: str,
) -> None:
    """Submit Vertex AI custom training job via the Python SDK."""
    from google.cloud import aiplatform

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = job_name or f"acoustic-{model_type}-{timestamp}"
    model_output = f"{GCS_MODEL_OUTPUT}{display_name}/"

    env_vars = {
        "ACOUSTIC_TRAINING_DADS_HF_REPO": hf_repo,
        "ACOUSTIC_TRAINING_MODEL_TYPE": model_type,
        "ACOUSTIC_TRAINING_MAX_EPOCHS": str(epochs),
        "ACOUSTIC_TRAINING_BATCH_SIZE": str(batch_size),
        "ACOUSTIC_TRAINING_LEARNING_RATE": str(learning_rate),
        "ACOUSTIC_TRAINING_LOSS_FUNCTION": loss_function,
    }

    if model_type == "efficientat_mn10":
        env_vars["ACOUSTIC_TRAINING_PRETRAINED_GCS"] = GCS_PRETRAINED

    print(f"\n{'='*60}")
    print(f"  Job:         {display_name}")
    print(f"  Model type:  {model_type}")
    print(f"  GPU:         {gpu_count}x {gpu_type}")
    print(f"  Machine:     {machine_type}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {learning_rate}")
    print(f"  Loss:        {loss_function}")
    print(f"  Data:        HuggingFace {hf_repo}")
    print(f"  Output:      {model_output}")
    print(f"{'='*60}\n")

    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        staging_bucket=f"gs://{GCS_BUCKET}",
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=image_uri,
    )

    job.run(
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=gpu_type,
        accelerator_count=gpu_count,
        environment_variables=env_vars,
        base_output_dir=model_output,
        sync=False,  # Don't block — monitor in Cloud Console
    )

    print(f"\n>>> Job submitted! Monitor at:")
    print(f"    https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={GCP_PROJECT}")
    print(f"\n>>> View logs:")
    print(f"    gcloud ai custom-jobs list --project={GCP_PROJECT} --region={GCP_REGION}")
    print(f"    gcloud ai custom-jobs stream-logs <JOB_ID> --project={GCP_PROJECT} --region={GCP_REGION}")
    print(f"\n>>> Download model when done:")
    print(f"    gsutil cp {model_output}best_model.pt models/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit Vertex AI training job for Sky Fort Acoustic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick T4 training (cheapest GPU)
  python scripts/vertex_submit.py

  # EfficientAT transfer learning
  python scripts/vertex_submit.py --model-type efficientat_mn10

  # Higher performance GPU
  python scripts/vertex_submit.py --gpu-type NVIDIA_L4 --machine-type g2-standard-8
  python scripts/vertex_submit.py --gpu-type NVIDIA_A100_80GB --machine-type a2-ultragpu-1g
        """,
    )
    parser.add_argument("--version", default="", choices=["", "v7", "v8"],
                        help="Phase version preset (v7=Phase 20, v8=Phase 22 locked config)")
    parser.add_argument("--image", default="",
                        help="Pre-built image URI (skip docker build). Required with --version v7.")
    parser.add_argument("--model-type", default="research_cnn",
                        choices=["research_cnn", "efficientat_mn10"],
                        help="Model architecture to train (default: research_cnn)")
    parser.add_argument("--gpu-type", default="NVIDIA_TESLA_T4",
                        help="GPU type (NVIDIA_TESLA_T4, NVIDIA_L4, NVIDIA_TESLA_V100, NVIDIA_A100_80GB)")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--machine-type", default="n1-standard-8",
                        help="GCE machine type (default: n1-standard-8)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", default="focal",
                        choices=["focal", "bce", "bce_weighted"],
                        help="Loss function")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--job-name", default="", help="Custom job display name")
    parser.add_argument("--tag", default="latest", help="Docker image tag")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and push image but don't submit job")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Docker build (use existing image)")
    args = parser.parse_args()

    # Phase 20 v7 preset — locked config, distinct submission path
    if args.version == "v7":
        if not args.image:
            print("ERROR: --version v7 requires --image <phase20 image URI>", file=sys.stderr)
            sys.exit(2)
        submit_v7_job(args.image, dry_run=args.dry_run)
        return

    # Phase 22 v8 preset — locked config, us-east1, field recordings
    if args.version == "v8":
        image_uri = args.image or GCR_BASE_IMAGE_V2
        submit_v8_job(image_uri, dry_run=args.dry_run)
        return

    # Build and push Docker image
    if args.skip_build:
        image_uri = f"{GCR_IMAGE}:{args.tag}"
        print(f">>> Using existing image: {image_uri}")
    else:
        image_uri = build_and_push_image(args.tag)

    if args.dry_run:
        print("\n>>> Dry run complete — image built and pushed, no job submitted.")
        return

    submit_job(
        image_uri=image_uri,
        model_type=args.model_type,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        machine_type=args.machine_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        loss_function=args.loss,
        hf_repo=args.hf_repo,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    main()
