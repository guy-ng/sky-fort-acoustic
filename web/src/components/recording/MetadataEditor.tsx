import { useState } from 'react'
import type { Recording, UpdateBody, LabelBody } from '../../hooks/useRecordings'
import { useUpdateRecording, useLabelRecording } from '../../hooks/useRecordings'

interface MetadataEditorProps {
  recording: Recording
  onClose: () => void
}

const INPUT_CLASS =
  'w-full bg-hud-bg border border-hud-border rounded px-2 py-1.5 text-sm text-hud-text focus:border-hud-accent focus:outline-none'

const LABELS = ['drone', 'background', 'other'] as const

export function MetadataEditor({ recording, onClose }: MetadataEditorProps) {
  const [label, setLabel] = useState(recording.label || '')
  const [subLabel, setSubLabel] = useState(recording.sub_label ?? '')
  const [distanceM, setDistanceM] = useState(recording.distance_m?.toString() ?? '')
  const [altitudeM, setAltitudeM] = useState(recording.altitude_m?.toString() ?? '')
  const [conditions, setConditions] = useState(recording.conditions ?? '')
  const [notes, setNotes] = useState(recording.notes ?? '')

  const updateMutation = useUpdateRecording()
  const labelMutation = useLabelRecording()

  const needsLabel = !recording.labeled
  const canSave = needsLabel ? label !== '' : true

  function handleSave() {
    if (needsLabel && label) {
      const body: LabelBody = { label }
      if (subLabel) body.sub_label = subLabel
      if (distanceM) body.distance_m = Number(distanceM)
      if (altitudeM) body.altitude_m = Number(altitudeM)
      if (conditions) body.conditions = conditions
      if (notes) body.notes = notes
      labelMutation.mutate({ id: recording.id, body }, { onSuccess: onClose })
    } else {
      const body: UpdateBody = {}
      if (subLabel !== (recording.sub_label ?? '')) body.sub_label = subLabel
      if (distanceM !== (recording.distance_m?.toString() ?? ''))
        body.distance_m = distanceM ? Number(distanceM) : undefined
      if (altitudeM !== (recording.altitude_m?.toString() ?? ''))
        body.altitude_m = altitudeM ? Number(altitudeM) : undefined
      if (conditions !== (recording.conditions ?? '')) body.conditions = conditions
      if (notes !== (recording.notes ?? '')) body.notes = notes
      updateMutation.mutate({ id: recording.id, body }, { onSuccess: onClose })
    }
  }

  return (
    <div className="flex flex-col gap-2 py-2">
      {needsLabel && (
        <>
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
        </>
      )}

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
          disabled={!canSave}
          className={`flex-1 px-3 py-1.5 text-sm rounded bg-hud-accent text-white ${
            !canSave ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-90'
          }`}
        >
          Save
        </button>
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-sm text-hud-text-dim hover:text-hud-text"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
