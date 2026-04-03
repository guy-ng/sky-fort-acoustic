#!/usr/bin/env python3
"""Download EfficientAT pretrained weights from GitHub releases.

Downloads mn10_as_mAP_471.pt (18MB) to models/pretrained/mn10_as.pt.
Skips download if the file already exists.

Usage:
    python scripts/download_pretrained.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt"
OUTPUT_DIR = Path("models/pretrained")
OUTPUT_PATH = OUTPUT_DIR / "mn10_as.pt"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  {mb_down:.1f} MB downloaded")
    sys.stdout.flush()


def download_pretrained() -> Path:
    """Download mn10 AudioSet-pretrained weights.

    Returns:
        Path to the downloaded weights file.
    """
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"Pretrained weights already exist: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        return OUTPUT_PATH

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EfficientAT mn10 pretrained weights...")
    print(f"  URL: {WEIGHTS_URL}")
    print(f"  Output: {OUTPUT_PATH}")

    urllib.request.urlretrieve(WEIGHTS_URL, OUTPUT_PATH, reporthook=_progress_hook)
    print()  # newline after progress

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Download complete: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    download_pretrained()
