# UMA-16 Ambient Pool (D-09)

Single-channel ambient captures from the UMA-16v2 microphone array, used by
`BackgroundNoiseMixer` (Phase 20 plan 20-02) as the noise mixin pool during
training.

## Hard constraint: one mic per file, never sum

A "mono UMA-16 capture" MUST be a **single channel** of the array (e.g. mic01).
Summing or averaging multiple mics into one mono signal causes destructive
**comb-filtering** driven by the time-of-arrival differences across the 4×4
grid. The comb pattern lives in the spectrum, the trainer learns to use it
as a feature, and the model fails to generalize when deployed against any
other UMA-16 unit (or a single mic).

When recording new captures: use `arecord -D hw:2,0 -c 1 -f S16_LE -r 16000`
with the **single-channel device** path, OR record all 16 channels and split
to per-channel files via `sox in.wav out_ch01.wav remix 1`. Do **not** use
`remix 1,2,3,4,5,...,16` (that averages) or any tool that produces a single
mono channel from a multi-channel UMA-16 capture by mixdown.

## Subdirectories

| Subdir | Source | Mic | Files | Duration |
|---|---|---|---:|---:|
| `outdoor_quiet/` | Acoustic-UAV-Identification dataset (background, 5inch session @ d90m) | mic01 + mic03 (independent files, never summed) | 243 | 31.3 min |
| `indoor_quiet/`, `indoor_hvac/`, `outdoor_wind/` | _to be captured_ | — | 0 | 0 min |

The 31.3 min `outdoor_quiet/` pool exceeds the D-09 ≥30 min floor, so v7
training can begin. The other three condition buckets remain optional for
diversity expansion.

## Regenerating from source

```bash
python scripts/ingest_uav_uma16_dataset.py        # WAV form
python scripts/export_uma16_parquet.py            # Parquet form (smaller, single-file)
```

The Parquet export goes to `data/parquet/ambient/train-0.parquet` and is
loadable by the trainer's existing `ParquetDatasetBuilder`.
