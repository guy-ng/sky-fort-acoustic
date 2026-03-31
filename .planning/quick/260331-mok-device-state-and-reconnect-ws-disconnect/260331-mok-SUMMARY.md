---
type: quick
plan: 260331-mok
subsystem: web-ui
tags: [overlay, device-status, heatmap, ux]
key-files:
  modified:
    - web/src/components/layout/DashboardLayout.tsx
decisions: []
metrics:
  duration: ~2min
  completed: 2026-03-31T13:25:00Z
---

# Quick Task 260331-mok: Device Disconnected Overlay Summary

**One-liner:** White semi-transparent overlay on heatmap canvas when UMA-16v2 device is not detected, with pulsing amber indicator and "Device Disconnected" label.

## What Was Done

Added a device disconnected overlay to the beamforming heatmap panel in `DashboardLayout.tsx`. When `deviceStatus.detected` is `false` (device not found by the backend), a white 50% opacity overlay renders over the heatmap canvas with:

- A pulsing amber dot indicator (`animate-pulse`)
- "DEVICE DISCONNECTED" text label in dark gray for readability against the white overlay
- `z-20` stacking context, above the existing "No Signal" overlay (`z-10`)

The overlay coexists with the existing "No Signal" overlay -- they indicate different conditions:
- **No Signal** (z-10): WebSocket heatmap connection is down
- **Device Disconnected** (z-20): The UMA-16v2 mic array hardware is not detected

The `deviceStatus` state comes from the existing `useDeviceStatus()` hook which connects to `/ws/status` WebSocket endpoint.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | c020721 | Add device disconnected overlay on heatmap canvas |

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED
