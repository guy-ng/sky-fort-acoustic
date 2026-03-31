---
phase: quick
plan: 260331-mok
type: execute
wave: 1
depends_on: []
files_modified:
  - web/src/components/layout/DashboardLayout.tsx
autonomous: true
---

<objective>
Add a white semi-transparent overlay on the beamforming heatmap when the physical audio device is disconnected.

Purpose: Visual feedback that the heatmap data is stale/unavailable because no mic array is plugged in — distinct from the existing "No Signal" overlay which covers WebSocket disconnection.
Output: Updated DashboardLayout.tsx with device-disconnected overlay.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@web/src/components/layout/DashboardLayout.tsx
@web/src/hooks/useDeviceStatus.ts
@web/src/components/layout/Sidebar.tsx
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add device-disconnected overlay to heatmap panel</name>
  <files>web/src/components/layout/DashboardLayout.tsx</files>
  <action>
In `DashboardLayout.tsx`, inside the heatmap panel's `<div className="relative flex-1">` container (line 57), add a second conditional overlay **after** the existing `!heatmapConnected` overlay block (line 58-63) and **before** the `<HeatmapCanvas>` component (line 64).

The new overlay renders when `!deviceStatus.detected`:

```tsx
{!deviceStatus.detected && (
  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/50">
    <span className="text-hud-bg text-sm uppercase tracking-wider font-semibold">Device Disconnected</span>
  </div>
)}
```

Key details:
- `bg-white/50` gives white at 50% opacity per the requirement.
- `z-10` matches the existing "No Signal" overlay z-index so they layer correctly. If both are true (WS down AND device unplugged), the device overlay shows underneath the "No Signal" overlay — that is fine since both conditions mean no data.
- Text uses `text-hud-bg` (dark color) so it is readable on the white overlay.
- The `deviceStatus` variable is already available in the component (line 24).

No other files need changes — the backend DeviceMonitor, WS endpoint, useDeviceStatus hook, and Sidebar indicator are all already implemented and working.
  </action>
  <verify>
    <automated>cd /Users/guyelisha/Projects/sky-fort-acoustic/web && npx tsc --noEmit 2>&1 | head -20</automated>
  </verify>
  <done>When deviceStatus.detected is false, a white 50%-transparent overlay with "Device Disconnected" text appears over the beamforming heatmap canvas. When device reconnects (detected becomes true), overlay disappears. TypeScript compiles without errors.</done>
</task>

</tasks>

<verification>
- `npx tsc --noEmit` passes with no errors
- Visual: with device unplugged, heatmap shows white overlay with "Device Disconnected" text
- Visual: with device connected, no overlay appears
- Visual: sidebar still shows red/green/yellow dot independently
</verification>

<success_criteria>
- White 50% transparent overlay visible on heatmap when device not detected
- Overlay disappears when device reconnects
- Existing "No Signal" overlay for WS disconnect still works independently
- No TypeScript errors
</success_criteria>

<output>
After completion, create `.planning/quick/260331-mok-device-state-and-reconnect-ws-disconnect/260331-mok-SUMMARY.md`
</output>
