import { useState } from 'react'
import { useRecordingSocket } from '../../hooks/useRecordingSocket'
import {
  useStartRecording,
  useStopRecording,
  useLabelRecording,
  useDeleteRecording,
} from '../../hooks/useRecordings'
import type { LabelBody } from '../../hooks/useRecordings'
import { AudioLevelMeter } from './AudioLevelMeter'

interface RecordingPanelProps {
  deviceDetected: boolean
}

const LABELS = ['drone', 'background', 'other'] as const

const INPUT_CLASS =
  'w-full bg-hud-bg border border-hud-border rounded px-2 py-1.5 text-sm text-hud-text focus:border-hud-accent focus:outline-none'

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function RecordingPanel({ deviceDetected }: RecordingPanelProps) {
  const recState = useRecordingSocket()
  const startMutation = useStartRecording()
  const stopMutation = useStopRecording()
  const labelMutation = useLabelRecording()
  const deleteMutation = useDeleteRecording()

  const [phase, setPhase] = useState<'idle' | 'recording' | 'labeling'>('idle')
  const [lastRecordingId, setLastRecordingId] = useState<string | null>(null)

  // Label form state
  const [label, setLabel] = useState('')
  const [subLabel, setSubLabel] = useState('')
  const [distanceM, setDistanceM] = useState('')
  const [altitudeM, setAltitudeM] = useState('')
  const [conditions, setConditions] = useState('')
  const [notes, setNotes] = useState('')

  function resetLabelForm() {
    setLabel('')
    setSubLabel('')
    setDistanceM('')
    setAltitudeM('')
    setConditions('')
    setNotes('')
  }

  function handleStart() {
    startMutation.mutate(undefined, {
      onSuccess: (data: { id: string }) => {
        setLastRecordingId(data.id)
        setPhase('recording')
      },
    })
  }

  function handleStop() {
    stopMutation.mutate(undefined, {
      onSuccess: () => {
        setPhase('labeling')
        resetLabelForm()
      },
    })
  }

  function handleSave() {
    if (!lastRecordingId || !label) return
    const body: LabelBody = { label }
    if (subLabel) body.sub_label = subLabel
    if (distanceM) body.distance_m = Number(distanceM)
    if (altitudeM) body.altitude_m = Number(altitudeM)
    if (conditions) body.conditions = conditions
    if (notes) body.notes = notes
    labelMutation.mutate(
      { id: lastRecordingId, body },
      {
        onSuccess: () => {
          setPhase('idle')
          setLastRecordingId(null)
        },
      },
    )
  }

  function handleDiscard() {
    if (!lastRecordingId) return
    deleteMutation.mutate(lastRecordingId, {
      onSuccess: () => {
        setPhase('idle')
        setLastRecordingId(null)
      },
    })
  }

  // Sync phase with WebSocket state
  const isRecording = recState.status === 'recording'
  if (phase === 'idle' && isRecording) {
    // External start detected
  }

  // Timer color based on remaining time
  const timerColor =
    recState.remaining_s > 0 && recState.remaining_s < 10
      ? 'text-hud-danger'
      : recState.remaining_s > 0 && recState.remaining_s < 30
        ? 'text-hud-warning'
        : 'text-hud-text'

  // --- Idle state ---
  if (phase === 'idle' && !isRecording) {
    return (
      <div className="flex flex-col gap-2">
        <button
          onClick={handleStart}
          disabled={!deviceDetected || startMutation.isPending}
          className={`w-full py-2 text-sm font-semibold rounded ${
            deviceDetected
              ? 'bg-hud-accent text-white hover:opacity-90'
              : 'bg-hud-border text-hud-text-dim cursor-not-allowed'
          }`}
        >
          {deviceDetected ? 'Start Recording' : 'No Device'}
        </button>
      </div>
    )
  }

  // --- Recording state ---
  if (phase === 'recording' || isRecording) {
    return (
      <div className="flex flex-col gap-3">
        <button
          onClick={handleStop}
          disabled={stopMutation.isPending}
          className="w-full py-2 text-sm font-semibold rounded bg-hud-danger text-white animate-pulse"
        >
          Stop Recording
        </button>
        <div className={`font-mono text-2xl font-semibold text-center ${timerColor}`}>
          {formatTime(recState.elapsed_s)}
          {recState.remaining_s > 0 && (
            <span className="text-hud-text-dim text-base ml-2">
              / {formatTime(recState.remaining_s)}
            </span>
          )}
        </div>
        <AudioLevelMeter level_db={recState.level_db} />
      </div>
    )
  }

  // --- Labeling state ---
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
        LABEL THIS RECORDING
      </span>
      <div className="flex gap-1.5">
        {LABELS.map(l => (
          <button
            key={l}
            onClick={() => setLabel(l)}
            className={`flex-1 px-2 py-1.5 text-xs rounded border ${
              label === l
                ? 'border-hud-accent text-hud-text'
                : 'border-hud-border text-hud-text-dim hover:text-hud-text'
            }`}
          >
            {l}
          </button>
        ))}
      </div>

      <span className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold mt-1">
        DETAILS (OPTIONAL)
      </span>

      <input
        type="text"
        placeholder="e.g., Mavic 3, traffic, wind"
        value={subLabel}
        onChange={e => setSubLabel(e.target.value)}
        className={INPUT_CLASS}
      />
      <div className="flex gap-2">
        <input
          type="number"
          placeholder="meters"
          min={0}
          value={distanceM}
          onChange={e => setDistanceM(e.target.value)}
          className={INPUT_CLASS}
        />
        <input
          type="number"
          placeholder="meters"
          min={0}
          value={altitudeM}
          onChange={e => setAltitudeM(e.target.value)}
          className={INPUT_CLASS}
        />
      </div>
      <input
        type="text"
        placeholder="e.g., light wind, rain"
        value={conditions}
        onChange={e => setConditions(e.target.value)}
        className={INPUT_CLASS}
      />
      <textarea
        rows={2}
        placeholder="Free-text notes"
        value={notes}
        onChange={e => setNotes(e.target.value)}
        className={INPUT_CLASS}
      />

      <div className="flex gap-2 mt-1">
        <button
          onClick={handleSave}
          disabled={!label || labelMutation.isPending}
          className={`flex-1 px-3 py-1.5 text-sm rounded bg-hud-accent text-white ${
            !label ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-90'
          }`}
        >
          Save Recording
        </button>
        <button
          onClick={handleDiscard}
          disabled={deleteMutation.isPending}
          className="px-3 py-1.5 text-sm text-hud-text-dim hover:text-hud-danger"
        >
          Discard Recording
        </button>
      </div>
    </div>
  )
}
