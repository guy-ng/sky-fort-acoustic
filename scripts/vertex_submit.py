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
GCS_BUCKET = "sky-fort-acoustic"
GCR_IMAGE = f"us-central1-docker.pkg.dev/{GCP_PROJECT}/acoustic-training/acoustic-trainer"

# HuggingFace dataset (no GCS upload needed)
DEFAULT_HF_REPO = "geronimobasso/drone-audio-detection-samples"

# GCS paths for model output and optional pretrained weights
GCS_MODEL_OUTPUT = f"gs://{GCS_BUCKET}/models/vertex/"
GCS_PRETRAINED = f"gs://{GCS_BUCKET}/models/pretrained/mn10_as.pt"


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
