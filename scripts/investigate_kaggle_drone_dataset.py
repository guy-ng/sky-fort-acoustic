#!/usr/bin/env python3
"""Investigate the Kaggle DroneAudioDataset for potential integration into v8 training.

Sources:
  - Kaggle notebook: https://www.kaggle.com/code/yehiellevi/audio-drone-sound-detection
  - Likely upstream: https://github.com/saraalemadi/DroneAudioDataset

This script probes metadata via GitHub API and README without downloading the
full dataset. If the repo is small enough, it shallow-clones to probe audio
properties (sample rate, channels, duration) on a random sample.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

REPO_OWNER = "saraalemadi"
REPO_NAME = "DroneAudioDataset"
GITHUB_API = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
README_RAW = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/master/README.md"
LICENSE_RAW = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/master/LICENSE"

MAX_FILES_FOR_CLONE = 200  # Clone only if repo has manageable file count
SAMPLE_SIZE = 5


def _fetch_url(url: str, *, accept: str = "application/json") -> str | None:
    """Fetch URL content, return None on failure."""
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "sky-fort-acoustic-probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  [WARN] Failed to fetch {url}: {exc}", file=sys.stderr)
        return None


def fetch_repo_metadata() -> dict | None:
    """Fetch repository metadata from GitHub API."""
    print("=" * 60)
    print("1. REPOSITORY METADATA (GitHub API)")
    print("=" * 60)
    raw = _fetch_url(GITHUB_API)
    if raw is None:
        print("  FAILED to fetch repo metadata.")
        return None
    data = json.loads(raw)
    print(f"  Description : {data.get('description', 'N/A')}")
    print(f"  License     : {data.get('license', {})}")
    print(f"  Size (KB)   : {data.get('size', 'N/A')}")
    print(f"  Stars       : {data.get('stargazers_count', 'N/A')}")
    print(f"  Forks       : {data.get('forks_count', 'N/A')}")
    print(f"  Created     : {data.get('created_at', 'N/A')}")
    print(f"  Updated     : {data.get('updated_at', 'N/A')}")
    print(f"  Default br. : {data.get('default_branch', 'N/A')}")
    return data


def fetch_readme() -> str | None:
    """Fetch and display README."""
    print()
    print("=" * 60)
    print("2. README (first 200 lines)")
    print("=" * 60)
    content = _fetch_url(README_RAW, accept="text/plain")
    if content is None:
        print("  FAILED to fetch README.")
        return None
    lines = content.splitlines()
    for line in lines[:200]:
        print(f"  {line}")
    if len(lines) > 200:
        print(f"  ... ({len(lines) - 200} more lines)")
    return content


def fetch_license() -> str | None:
    """Fetch LICENSE file if it exists."""
    print()
    print("=" * 60)
    print("3. LICENSE FILE")
    print("=" * 60)
    content = _fetch_url(LICENSE_RAW, accept="text/plain")
    if content is None:
        print("  No LICENSE file found at repo root (or fetch failed).")
        return None
    lines = content.splitlines()
    for line in lines[:50]:
        print(f"  {line}")
    if len(lines) > 50:
        print(f"  ... ({len(lines) - 50} more lines)")
    return content


def fetch_tree(default_branch: str) -> list[dict] | None:
    """Fetch the git tree to enumerate files."""
    print()
    print("=" * 60)
    print("4. FILE TREE ENUMERATION")
    print("=" * 60)
    url = f"{GITHUB_API}/git/trees/{default_branch}?recursive=1"
    raw = _fetch_url(url)
    if raw is None:
        print("  FAILED to fetch tree.")
        return None
    data = json.loads(raw)
    tree = data.get("tree", [])
    print(f"  Total entries: {len(tree)}")

    # Categorize
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    audio_files = [e for e in tree if e["type"] == "blob" and Path(e["path"]).suffix.lower() in audio_exts]
    dirs = [e for e in tree if e["type"] == "tree"]

    print(f"  Audio files : {len(audio_files)}")
    print(f"  Directories : {len(dirs)}")

    # Show directory structure
    print()
    print("  Directory structure:")
    for d in sorted(dirs, key=lambda x: x["path"]):
        print(f"    {d['path']}/")

    # Show sample of audio files
    print()
    print(f"  Sample audio files (up to 20):")
    for f in audio_files[:20]:
        size_str = f"  ({f.get('size', '?')} bytes)" if f.get("size") else ""
        print(f"    {f['path']}{size_str}")
    if len(audio_files) > 20:
        print(f"    ... and {len(audio_files) - 20} more")

    return audio_files


def probe_audio_samples(audio_files: list[dict], default_branch: str) -> list[dict]:
    """Shallow-clone and probe audio properties of a random sample."""
    print()
    print("=" * 60)
    print("5. AUDIO PROPERTY PROBE")
    print("=" * 60)

    if len(audio_files) > MAX_FILES_FOR_CLONE:
        print(f"  Too many audio files ({len(audio_files)}) -- skipping clone probe.")
        print("  Manual review required.")
        return []

    if len(audio_files) == 0:
        print("  No audio files found to probe.")
        return []

    # Try to import soundfile
    try:
        import soundfile as sf
    except ImportError:
        print("  soundfile not installed -- cannot probe audio properties.")
        print("  Install with: pip install soundfile")
        return []

    # Shallow clone
    tmpdir = tempfile.mkdtemp(prefix="drone_audio_probe_")
    clone_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
    print(f"  Cloning (shallow, no blobs) to {tmpdir} ...")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", clone_url, tmpdir],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"  Clone failed: {exc}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return []

    # Pick random sample
    sample = random.sample(audio_files, min(SAMPLE_SIZE, len(audio_files)))
    sample_paths = [e["path"] for e in sample]

    # Sparse checkout the sample files
    try:
        subprocess.run(
            ["git", "-C", tmpdir, "sparse-checkout", "set", "--no-cone", *sample_paths],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Fetch the blobs for the sample files
        subprocess.run(
            ["git", "-C", tmpdir, "checkout"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  Sparse checkout failed: {exc}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return []

    results = []
    for path in sample_paths:
        full = os.path.join(tmpdir, path)
        if not os.path.exists(full):
            print(f"  {path}: FILE NOT FOUND after checkout")
            continue
        try:
            info = sf.info(full)
            result = {
                "path": path,
                "samplerate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_s": round(info.frames / info.samplerate, 2),
                "format": info.format,
                "subtype": info.subtype,
            }
            results.append(result)
            print(f"  {path}:")
            print(f"    Sample rate : {info.samplerate} Hz")
            print(f"    Channels    : {info.channels}")
            print(f"    Duration    : {result['duration_s']}s ({info.frames} frames)")
            print(f"    Format      : {info.format} / {info.subtype}")
        except Exception as exc:
            print(f"  {path}: ERROR reading -- {exc}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def print_decision_template():
    """Print the decision criteria template."""
    print()
    print("=" * 60)
    print("6. DECISION CRITERIA TEMPLATE")
    print("=" * 60)
    print(
        textwrap.dedent("""\
    DECISION CRITERIA (fill in):
    License compatible with project use?      [ ]
    Sample rate 16 kHz mono WAV?              [ ]
    Labels match drone / background schema?   [ ]
    Overlap with DADS or field recordings?    [ ]
    Total size < 5 GB?                        [ ]
    Recording conditions defensible?          [ ]
    """)
    )


def main():
    print("DroneAudioDataset Investigation Probe")
    print(f"Date: {__import__('datetime').datetime.now(tz=__import__('datetime').timezone.utc).isoformat()}")
    print()

    # 1. Repo metadata
    meta = fetch_repo_metadata()

    # 2. README
    readme = fetch_readme()

    # 3. LICENSE
    license_text = fetch_license()

    # 4. File tree
    default_branch = meta.get("default_branch", "master") if meta else "master"
    audio_files = fetch_tree(default_branch)

    # 5. Audio probe
    probe_results = []
    if audio_files is not None:
        probe_results = probe_audio_samples(audio_files, default_branch)

    # 6. Decision template
    print_decision_template()

    # Summary
    print()
    print("=" * 60)
    print("INVESTIGATION SUMMARY")
    print("=" * 60)
    if meta:
        lic = meta.get("license")
        lic_name = lic.get("name", "N/A") if isinstance(lic, dict) else str(lic)
        print(f"  License (API)     : {lic_name}")
    if license_text:
        print(f"  LICENSE file      : Found ({len(license_text)} chars)")
    else:
        print("  LICENSE file      : NOT FOUND")
    if audio_files is not None:
        print(f"  Audio file count  : {len(audio_files)}")
    if probe_results:
        srs = set(r["samplerate"] for r in probe_results)
        chs = set(r["channels"] for r in probe_results)
        durs = [r["duration_s"] for r in probe_results]
        print(f"  Sample rates      : {srs}")
        print(f"  Channels          : {chs}")
        print(f"  Duration range    : {min(durs)}s - {max(durs)}s")
    print()
    print("Done. Review findings above and fill in decision criteria.")


if __name__ == "__main__":
    main()
