"""Vertex AI custom training entry point for Sky Fort Acoustic models.

Loads training data directly from HuggingFace (DADS dataset), runs the
training loop (ResearchCNN or EfficientAT), and uploads the best checkpoint
to GCS.

Vertex AI sets these environment variables automatically:
  AIP_MODEL_DIR   — GCS path where the trained model should be saved
  CLOUD_ML_JOB_ID — unique job identifier

Usage (local testing):
  export ACOUSTIC_TRAINING_DADS_HF_REPO=geronimobasso/drone-audio-detection-samples
  python -m scripts.vertex_train

Usage (Vertex AI):
  Submitted via vertex_submit.py — see that file for details.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vertex_train")

# Default HF repo for DADS
DEFAULT_HF_REPO = "geronimobasso/drone-audio-detection-samples"


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse gs://bucket/blob into (bucket, blob)."""
    assert gcs_path.startswith("gs://"), f"Not a GCS path: {gcs_path}"
    parts = gcs_path[5:].split("/", 1)
    return parts[0], parts[1]


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local file to GCS using google-cloud-storage."""
    from google.cloud import storage

    logger.info("[GCS] Uploading %s -> %s", local_path, gcs_path)
    file_size = Path(local_path).stat().st_size / (1024 * 1024)
    logger.info("[GCS] File size: %.1f MB", file_size)
    bucket_name, blob_name = _parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("[GCS] Upload complete: %s", gcs_path)


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download a single file from GCS using google-cloud-storage."""
    from google.cloud import storage

    logger.info("[GCS] Downloading %s -> %s", gcs_path, local_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    bucket_name, blob_name = _parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    file_size = Path(local_path).stat().st_size / (1024 * 1024)
    logger.info("[GCS] Download complete: %.1f MB", file_size)


def log_gpu_info() -> None:
    """Log GPU information if available."""
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info("[GPU] Device %d: %s (%.1f GB)", i, name, mem)
    else:
        logger.info("[GPU] No CUDA devices available — training on CPU")


def log_system_info() -> None:
    """Log system resource information."""
    import torch
    logger.info("[SYS] PyTorch version: %s", torch.__version__)
    logger.info("[SYS] CUDA available: %s", torch.cuda.is_available())
    if hasattr(torch.backends, "mps"):
        logger.info("[SYS] MPS available: %s", torch.backends.mps.is_available())

    cpu_count = os.cpu_count() or 0
    logger.info("[SYS] CPU cores: %d", cpu_count)

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info("[SYS] RAM: %.1f GB total, %.1f GB available",
                    mem.total / (1024**3), mem.available / (1024**3))
    except ImportError:
        pass


def main() -> None:
    """Run training on Vertex AI."""
    start_time = time.time()

    # --- Environment ---
    model_dir = os.environ.get("AIP_MODEL_DIR", "")
    model_type = os.environ.get("ACOUSTIC_TRAINING_MODEL_TYPE", "research_cnn")
    hf_repo = os.environ.get("ACOUSTIC_TRAINING_DADS_HF_REPO", DEFAULT_HF_REPO)
    pretrained_gcs = os.environ.get("ACOUSTIC_TRAINING_PRETRAINED_GCS", "")
    job_id = os.environ.get("CLOUD_ML_JOB_ID", "local")

    logger.info("=" * 60)
    logger.info("[JOB] Vertex AI Training Job: %s", job_id)
    logger.info("[JOB] Model type: %s", model_type)
    logger.info("[JOB] Data source: HuggingFace %s", hf_repo)
    logger.info("[JOB] Model output: %s", model_dir or "(local only)")
    logger.info("=" * 60)

    # System info
    log_system_info()
    log_gpu_info()

    # Ensure HF repo is set for the training config
    os.environ["ACOUSTIC_TRAINING_DADS_HF_REPO"] = hf_repo
    os.environ["ACOUSTIC_TRAINING_MODEL_TYPE"] = model_type
    os.environ["ACOUSTIC_TRAINING_CHECKPOINT_PATH"] = "/tmp/models/best_model.pt"

    # Download pretrained weights from GCS if specified (for EfficientAT)
    if pretrained_gcs and model_type == "efficientat_mn10":
        logger.info("[SETUP] Downloading pretrained EfficientAT weights...")
        local_pretrained = "/tmp/pretrained/mn10_as.pt"
        download_from_gcs(pretrained_gcs, local_pretrained)
        os.environ["ACOUSTIC_TRAINING_PRETRAINED_WEIGHTS"] = local_pretrained

    # Import training modules after setting env vars
    logger.info("[SETUP] Loading training configuration...")
    from acoustic.classification.config import MelConfig
    from acoustic.training.config import TrainingConfig

    config = TrainingConfig()
    mel_config = MelConfig()

    logger.info("[CONFIG] learning_rate = %.1e", config.learning_rate)
    logger.info("[CONFIG] batch_size    = %d", config.batch_size)
    logger.info("[CONFIG] max_epochs    = %d", config.max_epochs)
    logger.info("[CONFIG] patience      = %d", config.patience)
    logger.info("[CONFIG] loss_function = %s", config.loss_function)
    logger.info("[CONFIG] augmentation  = %s", config.augmentation_enabled)
    logger.info("[CONFIG] model_type    = %s", config.model_type)
    if config.model_type == "efficientat_mn10":
        logger.info("[CONFIG] stage1: %d epochs @ lr=%.1e", config.stage1_epochs, config.stage1_lr)
        logger.info("[CONFIG] stage2: %d epochs @ lr=%.1e", config.stage2_epochs, config.stage2_lr)
        logger.info("[CONFIG] stage3: %d epochs @ lr=%.1e", config.stage3_epochs, config.stage3_lr)

    # --- Load HF dataset (will download and cache) ---
    logger.info("[DATA] Downloading HuggingFace dataset: %s ...", hf_repo)
    data_download_start = time.time()
    from datasets import load_dataset
    _ = load_dataset(hf_repo, split="train")
    data_elapsed = time.time() - data_download_start
    logger.info("[DATA] Dataset downloaded/cached in %.1f seconds", data_elapsed)

    # --- Run training ---
    stop_event = threading.Event()
    train_start = time.time()
    best_val_loss = float("inf")
    best_val_acc = 0.0

    def progress_cb(metrics: dict) -> None:
        nonlocal best_val_loss, best_val_acc

        epoch = metrics.get("epoch", "?")
        total = metrics.get("total_epochs", "?")
        train_loss = metrics.get("train_loss", 0)
        val_loss = metrics.get("val_loss", 0)
        val_acc = metrics.get("val_acc", 0)
        best_loss = metrics.get("best_val_loss", 0)
        stage = metrics.get("stage", "")
        batch = metrics.get("batch")
        total_batches = metrics.get("total_batches")

        elapsed = time.time() - train_start
        elapsed_min = elapsed / 60

        # Batch-level progress (during epoch)
        if batch and total_batches and val_loss == 0:
            pct = batch / total_batches * 100
            stage_str = f" [stage {stage}]" if stage else ""
            logger.info(
                "[TRAIN] Epoch %s/%s%s — batch %d/%d (%.0f%%) — train_loss=%.4f — %.1f min elapsed",
                epoch, total, stage_str, batch, total_batches, pct, train_loss, elapsed_min,
            )
            return

        # Epoch-level progress
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = " *** NEW BEST ***"
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        stage_str = f" [stage {stage}]" if stage else ""

        # Confusion matrix
        tp = metrics.get("tp", 0)
        fp = metrics.get("fp", 0)
        tn = metrics.get("tn", 0)
        fn = metrics.get("fn", 0)

        logger.info("-" * 60)
        logger.info(
            "[EPOCH] %s/%s%s — %.1f min elapsed",
            epoch, total, stage_str, elapsed_min,
        )
        logger.info(
            "[EPOCH] train_loss=%.4f | val_loss=%.4f | val_acc=%.3f%s",
            train_loss, val_loss, val_acc, improved,
        )
        logger.info(
            "[EPOCH] best_val_loss=%.4f | best_val_acc=%.3f",
            best_loss or best_val_loss, best_val_acc,
        )
        if tp + fp + tn + fn > 0:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            logger.info(
                "[EPOCH] TP=%d FP=%d TN=%d FN=%d | precision=%.3f recall=%.3f F1=%.3f",
                tp, fp, tn, fn, precision, recall, f1,
            )

        # Estimate remaining time
        if isinstance(epoch, int) and isinstance(total, int) and epoch > 0:
            time_per_epoch = elapsed / epoch
            remaining = (total - epoch) * time_per_epoch
            logger.info(
                "[EPOCH] ~%.1f min/epoch | ~%.1f min remaining (est.)",
                time_per_epoch / 60, remaining / 60,
            )

    logger.info("=" * 60)
    logger.info("[TRAIN] Starting training loop...")
    logger.info("=" * 60)

    if model_type == "efficientat_mn10":
        from acoustic.training.efficientat_trainer import EfficientATTrainingRunner
        runner = EfficientATTrainingRunner(config)
        checkpoint = runner.run(stop_event, progress_callback=progress_cb)
    else:
        from acoustic.training.trainer import TrainingRunner
        runner = TrainingRunner(config, mel_config)
        checkpoint = runner.run(stop_event, progress_callback=progress_cb)

    train_elapsed = time.time() - train_start
    total_elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("[DONE] Training completed in %.1f minutes", train_elapsed / 60)
    logger.info("[DONE] Total job time: %.1f minutes", total_elapsed / 60)
    logger.info("[DONE] Best val_loss: %.4f | Best val_acc: %.3f", best_val_loss, best_val_acc)

    if checkpoint is None:
        logger.warning("[DONE] No checkpoint was saved (model did not improve)")
        sys.exit(1)

    logger.info("[DONE] Best checkpoint: %s", checkpoint)
    ckpt_size = Path(str(checkpoint)).stat().st_size / (1024 * 1024)
    logger.info("[DONE] Checkpoint size: %.1f MB", ckpt_size)

    # --- Upload model to GCS ---
    if model_dir:
        logger.info("[UPLOAD] Uploading model artifacts to GCS...")
        upload_to_gcs(str(checkpoint), f"{model_dir}/best_model.pt")

        # Upload TorchScript version if it exists
        jit_path = Path(str(checkpoint) + ".jit")
        if jit_path.exists():
            upload_to_gcs(str(jit_path), f"{model_dir}/best_model.pt.jit")

        logger.info("[UPLOAD] All artifacts uploaded to %s", model_dir)
    else:
        logger.info("[UPLOAD] AIP_MODEL_DIR not set — skipping GCS upload")

    logger.info("=" * 60)
    logger.info("[JOB] Training job %s finished successfully", job_id)
    logger.info("[JOB] Total wall time: %.1f minutes", total_elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
