---
phase: 21
plan: 04
subsystem: apps/rpi-edge
tags: [config, yaml, cli, dataclass, security, edge]
requires:
  - 21-01  # skyfort_edge package + Wave 0 RED stubs
provides:
  - EdgeConfig dataclass tree (thresholds/timing/hardware/model/detection_log/http)
  - load_config(yaml_path, cli_overrides) merge function with strict schema
  - Default apps/rpi-edge/config.yaml with all D-11 keys
affects:
  - apps/rpi-edge/tests/test_config_merge.py (RED stub -> GREEN)
tech-stack:
  added: [PyYAML (already in pyproject)]
  patterns: [dataclass config tree, strict unknown-key rejection, loopback bind validation]
key-files:
  created:
    - apps/rpi-edge/skyfort_edge/config.py
    - apps/rpi-edge/config.yaml
  modified:
    - apps/rpi-edge/tests/test_config_merge.py
decisions:
  - D-09 YAML + CLI override implemented with yaml.safe_load only
  - D-10 No hot reload; merge precedence CLI > YAML > defaults
  - D-11 Four param groups split into six dataclasses (thresholds, timing, hardware, model, detection_log, http) to keep detection_log + http separately typed
  - Default detection log path locked to /var/lib/skyfort-edge/detections.jsonl
  - Default config lookup order: /etc/skyfort-edge/config.yaml then apps/rpi-edge/config.yaml
  - HTTP bind restricted at load time to 127.0.0.1/localhost/::1 (T-21-01)
metrics:
  duration: ~10m
  tasks_completed: 2
  commits:
    - 5f3c469 feat(21-04): add EdgeConfig dataclass tree + default config.yaml
    - 8fdf519 test(21-04): GREEN config merge tests
  completed: 2026-04-08
---

# Phase 21 Plan 04: Edge Config Loader Summary

Typed YAML + CLI config loader for the Pi edge app covering all D-11 parameter groups, with yaml.safe_load enforcement and loopback/path-traversal guards wired to T-21-01/02/03.

## What Shipped

### `apps/rpi-edge/skyfort_edge/config.py`

Six dataclass groups plus a top-level `EdgeConfig`:

| Group           | Fields                                                                                                                                                 |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `thresholds`    | score_threshold, enter_threshold, exit_threshold, confirm_hits, release_hits, class_allowlist, per_class_thresholds                                    |
| `timing`        | window_seconds, hop_seconds, smoothing_window, ema_alpha, post_release_cooldown_seconds                                                                |
| `hardware`      | input_device, capture_sample_rate, led_gpio_pin, min_on_seconds, alarm_audio_device, alarm_enabled                                                     |
| `model`         | onnx_path, fallback_onnx_path, prefer_int8, num_threads, execution_provider, log_level                                                                 |
| `detection_log` | path, rotate_max_bytes, rotate_backup_count                                                                                                            |
| `http`          | bind_host, bind_port                                                                                                                                   |

`load_config(yaml_path, cli_overrides)`:

1. Reads YAML with `yaml.safe_load` (never `yaml.load` / unsafe loader).
2. Rejects unknown top-level groups with `ValueError`.
3. Merges each group: `defaults <- YAML <- CLI overrides` (later wins).
4. Rejects unknown per-group keys (per-dataclass `fields()` introspection).
5. Runs `_validate_http_bind` (T-21-01) and `_validate_detection_log_path` (T-21-02).

`DEFAULT_CONFIG_PATHS = [/etc/skyfort-edge/config.yaml, apps/rpi-edge/config.yaml]`.

### `apps/rpi-edge/config.yaml`

Default on-disk config with every D-11 key populated and inline comments citing decision IDs. Safe defaults:

- `http.bind_host: 127.0.0.1` / `bind_port: 8088`
- `detection_log.path: /var/lib/skyfort-edge/detections.jsonl`, 10 MB rotation, 5 backups
- `model.onnx_path: /opt/skyfort-edge/models/efficientat_mn10_v6_int8.onnx` (int8 preferred, FP32 fallback)
- `hardware.led_gpio_pin: 17`, `min_on_seconds: 2.0`, alarm disabled
- `timing.window_seconds: 1.0`, `hop_seconds: 0.5` (D-03)
- `thresholds` hysteresis defaults: 0.6 enter / 0.4 exit / 3 confirm / 5 release

### `apps/rpi-edge/tests/test_config_merge.py`

Wave 0 RED stub replaced with 8 GREEN tests:

| Test                                         | Covers               |
|----------------------------------------------|----------------------|
| `test_yaml_loads_with_safe_load_only`        | T-21-03 (source scan)|
| `test_default_paths_match_research`          | D-09 default paths   |
| `test_cli_overrides_yaml`                    | D-10 precedence      |
| `test_unknown_top_level_key_rejected`        | strict schema        |
| `test_unknown_field_in_group_rejected`       | strict schema        |
| `test_http_bind_host_must_be_loopback`       | T-21-01              |
| `test_detection_log_path_rejects_dotdot`     | T-21-02              |
| `test_all_four_param_groups_present`         | D-11 structural      |

All 8 pass: `pytest tests/test_config_merge.py -x -q` -> `........ [100%]`.

## Threat Model Coverage

| Threat ID | Status    | How                                                                          |
|-----------|-----------|------------------------------------------------------------------------------|
| T-21-01   | mitigated | `_validate_http_bind` + `test_http_bind_host_must_be_loopback`              |
| T-21-02   | mitigated | `_validate_detection_log_path` rejects `..` and non-absolute paths          |
| T-21-03   | mitigated | `yaml.safe_load` is the only loader; source-scan test enforces it           |

## Deviations from Plan

### Tightened to spec during implementation

**1. [Rule 2 - Hardening] Also require `detection_log.path` to be absolute**
- **Where:** `_validate_detection_log_path`
- **Why:** The plan called out absolute-path sanity in the action block but the acceptance criteria only mentioned `..`. Kept the absolute-path check because a relative log path on a systemd service would silently land in the CWD (`/`), which is worse than failing fast.
- **Impact:** Existing default `/var/lib/skyfort-edge/detections.jsonl` passes; any relative path in YAML now raises `ValueError`.

**2. [Rule 1 - Wording fix] Docstring rewrite to avoid triggering the safe-load grep**
- **Where:** Top-of-file module docstring
- **Why:** Original wording mentioned `yaml.unsafe_load` as a negative example, which made the `yaml.load(|unsafe_load` grep false-positive during verification. Rewrote to "never use the non-safe YAML loaders" — same meaning, no forbidden substring.
- **Impact:** Automated verify passes cleanly.

Otherwise the plan executed exactly as written.

## Authentication Gates

None.

## Known Stubs

None. All 8 tests are fully implemented and pass; no placeholder data in `config.py` or `config.yaml`.

## Self-Check

- Files exist:
  - `apps/rpi-edge/skyfort_edge/config.py` FOUND
  - `apps/rpi-edge/config.yaml` FOUND
  - `apps/rpi-edge/tests/test_config_merge.py` FOUND
- Commits exist on `main`:
  - `5f3c469` FOUND (Task 1)
  - `8fdf519` FOUND (Task 2)
- Tests: `pytest tests/test_config_merge.py -x -q` -> 8 passed

## Self-Check: PASSED
