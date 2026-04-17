import { useCallback, useState } from 'react'

interface BeamformingControlsProps {
  onRecordStart: () => void
  onRecordStop: () => void
  recordingState: { status: string; elapsed_s?: number; remaining_s?: number }
  onTargetRecordStart: () => void
  onTargetRecordStop: () => void
  targetRecordingState: { status: string; elapsed_s?: number; samples?: number }
}

const API_BASE = ''

async function patchSettings(updates: Record<string, number | boolean>) {
  await fetch(`${API_BASE}/api/settings`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  })
}

export function BeamformingControls({ onRecordStart, onRecordStop, recordingState, onTargetRecordStart, onTargetRecordStop, targetRecordingState }: BeamformingControlsProps) {
  const [bfNu, setBfNu] = useState(4)
  const [massThreshold, setMassThreshold] = useState(10)

  const handleNuChange = useCallback((val: number) => {
    setBfNu(val)
    patchSettings({ bf_nu: val })
  }, [])

  const handleMassThresholdChange = useCallback((val: number) => {
    setMassThreshold(val)
    patchSettings({ bf_mass_threshold: val / 100 })
  }, [])

  const isRecording = recordingState.status === 'recording'
  const isTargetRecording = targetRecordingState.status === 'recording'

  return (
    <div className="space-y-2 text-xs">
      {/* Sharpness */}
      <div>
        <div className="flex justify-between text-hud-text-dim mb-0.5">
          <span>Sharpness (nu)</span>
          <span className="tabular-nums">{bfNu}</span>
        </div>
        <input
          type="range" min={1} max={50} step={1} value={bfNu}
          onChange={(e) => handleNuChange(Number(e.target.value))}
          className="w-full h-1.5 accent-hud-accent"
        />
      </div>

      {/* Mass threshold — cuts low noise from center-of-mass calc */}
      <div>
        <div className="flex justify-between text-hud-text-dim mb-0.5">
          <span>Focus cutoff</span>
          <span className="tabular-nums">{massThreshold}%</span>
        </div>
        <input
          type="range" min={1} max={90} step={1} value={massThreshold}
          onChange={(e) => handleMassThresholdChange(Number(e.target.value))}
          className="w-full h-1.5 accent-red-500"
        />
      </div>

      {/* Raw 16ch recording */}
      <div className="border-t border-hud-border/50 pt-2">
        {isRecording ? (
          <div className="space-y-1">
            <button
              onClick={onRecordStop}
              className="w-full py-1.5 rounded bg-hud-danger/80 hover:bg-hud-danger text-white text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1.5"
            >
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              Stop Recording
            </button>
            <div className="flex justify-between text-hud-text-dim tabular-nums">
              <span>{recordingState.elapsed_s?.toFixed(0) ?? 0}s elapsed</span>
              <span>{recordingState.remaining_s?.toFixed(0) ?? 60}s left</span>
            </div>
          </div>
        ) : (
          <button
            onClick={onRecordStart}
            className="w-full py-1.5 rounded bg-hud-border hover:bg-hud-accent/30 text-hud-text text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1.5"
          >
            <span className="material-symbols-outlined text-sm">fiber_manual_record</span>
            Record 16ch (60s)
          </button>
        )}
      </div>

      {/* Target location recording */}
      <div className="border-t border-hud-border/50 pt-2">
        {isTargetRecording ? (
          <div className="space-y-1">
            <button
              onClick={onTargetRecordStop}
              className="w-full py-1.5 rounded bg-amber-600/80 hover:bg-amber-600 text-white text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1.5"
            >
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              Stop Target Rec
            </button>
            <div className="flex justify-between text-hud-text-dim tabular-nums">
              <span>{targetRecordingState.elapsed_s?.toFixed(0) ?? 0}s</span>
              <span>{targetRecordingState.samples ?? 0} samples</span>
            </div>
          </div>
        ) : (
          <button
            onClick={onTargetRecordStart}
            className="w-full py-1.5 rounded bg-hud-border hover:bg-hud-accent/30 text-hud-text text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1.5"
          >
            <span className="material-symbols-outlined text-sm">track_changes</span>
            Record Targets
          </button>
        )}
      </div>
    </div>
  )
}
