# Phase 22 — v8 Training Run Log

## Preflight Results

| Check | Status | Detail |
|-------|--------|--------|
| Local data preflight | OK | drone=9 files (603.5s), bg=3 files (34.8s); holdout=4 drone + 1 bg |
| L4 quota us-east1 | OK | NVIDIA_L4_GPUS limit=1.0, usage=0.0 |
| Base image :v2 | OK | Pushed 2026-04-13, digest sha256:0890a8a1f80a05a98ad51ec2bff43d38c4c8ef065b53e1e87d34c013eb77cd67 |
| Cross-region pull | SKIPPED | Image in us-central1 Artifact Registry, Vertex us-east1 pulls cross-region (validated by v7 job success) |
| v6 sha256 | c8828b5d452c19c11f78a7cd5cb5caabc87339aa6c12656f3be9920587be21eb | models/efficientat_mn10_v6.pt |
| Dockerfile fix | g++ added | pyroomacoustics C++ build dependency was missing; added g++ to apt-get |
| GCS bucket fix | sky-fort-acoustic-east1 | Original script used us-central1 bucket; Vertex us-east1 requires co-located bucket |
| staging_bucket fix | Added to aip.init() | SDK requires explicit staging_bucket parameter |

## Submission

| Field | Value |
|-------|-------|
| Job resource name | projects/859551133057/locations/us-east1/customJobs/6134474782122442752 |
| Display name | efficientat-mn10-v8-phase22 |
| Region | us-east1 |
| Machine | g2-standard-8 |
| Accelerator | NVIDIA_L4 x1 |
| Image URI | us-central1-docker.pkg.dev/interception-dashboard/acoustic-training/acoustic-trainer-base:v2 |
| Output dir | gs://sky-fort-acoustic-east1/training/efficientat_v8 |
| Submit time | 2026-04-13 |
| Fine-tune from | models/efficientat_mn10_v6.pt (v6) |

## Results

<!-- Fill after job completes -->

| Field | Value |
|-------|-------|
| End time | |
| Final state | |
| Val accuracy | |
| Val loss | |
| v8 GCS path | |
| v8 sha256 | |
| Length-mismatch WARNs | |

## Notes

- Dockerfile.vertex-base required g++ for pyroomacoustics wheel build (cross-platform amd64 on ARM Mac)
- vertex_submit.py staging_bucket was missing — SDK requires it for CustomContainerTrainingJob
- GCS output bucket must be co-located with Vertex region (us-east1), switched to sky-fort-acoustic-east1
- Used CustomJob API directly instead of CustomContainerTrainingJob.run(sync=False) which silently failed
