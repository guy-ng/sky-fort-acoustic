"""Fetch the FSD50K 6-class subset directly from the Fhrozen/FSD50k HuggingFace mirror.

Bypasses soundata's Zenodo path because Zenodo throttles aggressively. The HF mirror
exposes individual files and is unthrottled, so we download only the ~1500-2000 WAVs
matching the 6 D-09 target classes (Wind, Rain, Traffic_noise_and_roadway_noise,
Mechanical_fan, Engine, Bird).

Layout produced (matches what soundata's acquire_fsd50k() would have produced):
  data/noise/fsd50k_subset/Wind/<fname>.wav
  data/noise/fsd50k_subset/Rain/<fname>.wav
  ...
  data/noise/fsd50k_subset/.acquired.json

Usage:
  python fetch_fsd50k_hf.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = "Fhrozen/FSD50k"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

OUT_DIR = Path("data/noise/fsd50k_subset")

# D-09: 6 dominant classes from FSD50K AudioSet vocabulary
TARGET_CLASSES: tuple[str, ...] = (
    "Wind",
    "Rain",
    "Traffic_noise_and_roadway_noise",
    "Mechanical_fan",
    "Engine",
    "Bird",
)

# Per-class minimum (from D-13). If primary-label pass falls short for any class,
# secondary-label fallback fires for that class.
MIN_PER_CLASS = 100

WORKERS = 16  # parallel WAV downloads


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_fsd50k_hf/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _save(url: str, dst: Path) -> int:
    """Download url to dst (atomic). Return size."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    data = _get(url)
    tmp.write_bytes(data)
    tmp.rename(dst)
    return len(data)


def fetch_csv(name: str) -> list[dict]:
    """Fetch labels/<name> from the HF repo and parse as CSV dicts."""
    url = f"{BASE}/labels/{name}"
    raw = _get(url).decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def collect_targets(rows: list[dict], split: str) -> dict[str, list[tuple[str, str]]]:
    """Pass 1 (primary-label rule): yield {class: [(fname, split)]}.

    `labels` column is comma-separated. The first label is the "primary" tag.
    """
    out: dict[str, list[tuple[str, str]]] = {c: [] for c in TARGET_CLASSES}
    for row in rows:
        labels = row.get("labels", "")
        if not labels:
            continue
        primary = labels.split(",", 1)[0].strip()
        if primary in out:
            out[primary].append((row["fname"], split))
    return out


def collect_secondary(
    rows: list[dict], split: str, target_class: str
) -> list[tuple[str, str]]:
    """Pass 2: any clip whose label list CONTAINS target_class."""
    out: list[tuple[str, str]] = []
    for row in rows:
        labels = [s.strip() for s in row.get("labels", "").split(",")]
        if target_class in labels:
            out.append((row["fname"], split))
    return out


def main() -> int:
    print(f">>> fetching FSD50K 6-class subset from {REPO}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch label CSVs
    print(">>> fetching label CSVs...")
    dev_rows = fetch_csv("dev.csv")
    eval_rows = fetch_csv("eval.csv")
    print(f"    dev.csv:  {len(dev_rows)} rows")
    print(f"    eval.csv: {len(eval_rows)} rows")

    # 2. Pass 1 — primary-label collect
    print(">>> Pass 1: primary-label filter...")
    targets: dict[str, list[tuple[str, str]]] = {c: [] for c in TARGET_CLASSES}
    for cls, items in collect_targets(dev_rows, "dev").items():
        targets[cls].extend(items)
    for cls, items in collect_targets(eval_rows, "eval").items():
        targets[cls].extend(items)
    for cls in TARGET_CLASSES:
        print(f"    {cls}: {len(targets[cls])} (primary)")

    # 3. Pass 2 — secondary fallback for under-populated classes
    for cls in TARGET_CLASSES:
        if len(targets[cls]) >= MIN_PER_CLASS:
            continue
        print(f">>> Pass 2: {cls} under {MIN_PER_CLASS}, secondary-label fallback...")
        existing = {fn for fn, _ in targets[cls]}
        for split, rows in (("dev", dev_rows), ("eval", eval_rows)):
            for fname, sp in collect_secondary(rows, split, cls):
                if fname in existing:
                    continue
                targets[cls].append((fname, sp))
                existing.add(fname)
        print(f"    {cls}: {len(targets[cls])} (after fallback)")

    # 4. Plan download list
    plan: list[tuple[str, str, str]] = []  # (fname, split, class)
    seen: set[tuple[str, str]] = set()  # (fname, class) — same file may appear once per class
    for cls, items in targets.items():
        for fname, split in items:
            key = (fname, cls)
            if key in seen:
                continue
            seen.add(key)
            plan.append((fname, split, cls))

    total = len(plan)
    print(f">>> downloading {total} WAVs in parallel ({WORKERS} workers)...")

    # 5. Parallel download
    t0 = time.time()
    done = 0
    bytes_total = 0
    failures = 0

    def _job(fname: str, split: str, cls: str) -> tuple[bool, int, str]:
        url = f"{BASE}/clips/{split}/{fname}.wav"
        dst = OUT_DIR / cls / f"{fname}.wav"
        if dst.exists() and dst.stat().st_size > 0:
            return True, dst.stat().st_size, fname
        try:
            sz = _save(url, dst)
            return True, sz, fname
        except Exception as e:
            print(f"  ! {url} -> {type(e).__name__}: {e}", file=sys.stderr)
            return False, 0, fname

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_job, fn, sp, cl) for fn, sp, cl in plan]
        for f in as_completed(futs):
            ok, sz, fname = f.result()
            done += 1
            if ok:
                bytes_total += sz
            else:
                failures += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = bytes_total / elapsed / 1024 / 1024 if elapsed > 0 else 0
                print(
                    f"    [{done:>4}/{total}] {bytes_total // 1024 // 1024} MB"
                    f" @ {rate:.1f} MB/s, {failures} failures"
                )

    elapsed = time.time() - t0
    print(f">>> done in {elapsed:.1f}s, {bytes_total // 1024 // 1024} MB, {failures} failures")

    # 6. Per-class WAV count audit
    print(">>> per-class file counts on disk:")
    counts = {}
    for cls in TARGET_CLASSES:
        n = sum(1 for _ in (OUT_DIR / cls).rglob("*.wav"))
        counts[cls] = n
        flag = "" if n >= MIN_PER_CLASS else " ** UNDER MIN **"
        print(f"    {cls}: {n}{flag}")

    # 7. Write marker
    total_wavs = sum(counts.values())
    total_bytes_disk = sum(p.stat().st_size for p in OUT_DIR.rglob("*.wav"))
    marker = {
        "schema_version": 1,
        "corpus": "fsd50k_subset",
        "source": f"huggingface:{REPO}@main:6class-primary-with-secondary-fallback",
        "soundata_version": "n/a (HF mirror, soundata bypassed due to Zenodo throttling)",
        "file_count": total_wavs,
        "total_bytes": total_bytes_disk,
        "per_class_count": counts,
        "acquired_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Acquired from Fhrozen/FSD50k HF mirror because Zenodo per-IP throttling "
            "made the soundata path infeasible. Same 6-class D-09 filter (primary tag "
            "with secondary-label fallback for under-populated classes). Fnames match "
            "FSD50K.{dev,eval}.csv exactly."
        ),
    }
    (OUT_DIR / ".acquired.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(f">>> marker written: {total_wavs} wavs, {total_bytes_disk // 1024 // 1024} MB")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
