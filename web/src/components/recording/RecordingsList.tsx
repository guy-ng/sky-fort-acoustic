import { useRef, useState } from 'react'
import { useRecordingsList, useDeleteRecording } from '../../hooks/useRecordings'
import type { Recording } from '../../hooks/useRecordings'
import { MetadataEditor } from './MetadataEditor'

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function LabelBadge({ recording }: { recording: Recording }) {
  if (!recording.labeled) {
    return (
      <span className="text-xs px-1.5 py-0.5 rounded bg-hud-warning/20 text-hud-warning italic">
        unlabeled
      </span>
    )
  }
  const colorMap: Record<string, string> = {
    drone: 'bg-hud-danger/20 text-hud-danger',
    background: 'bg-hud-success/20 text-hud-success',
    other: 'bg-hud-warning/20 text-hud-warning',
  }
  const cls = colorMap[recording.label] ?? 'bg-hud-warning/20 text-hud-warning'
  return <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>{recording.label}</span>
}

export function RecordingsList() {
  const { data: recordings, isLoading } = useRecordingsList()
  const deleteMutation = useDeleteRecording()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  function handlePlay(id: string) {
    if (playingId === id) {
      audioRef.current?.pause()
      setPlayingId(null)
      return
    }
    if (audioRef.current) {
      audioRef.current.pause()
    }
    const audio = new Audio(`/api/recordings/${id}/audio`)
    audio.onended = () => setPlayingId(null)
    audio.onerror = () => setPlayingId(null)
    audio.play()
    audioRef.current = audio
    setPlayingId(id)
  }

  if (isLoading) {
    return <div className="text-sm text-hud-text-dim text-center py-4">Loading...</div>
  }

  if (!recordings || recordings.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-6 gap-2">
        <span className="text-sm font-semibold text-hud-text">No Recordings</span>
        <span className="text-xs text-hud-text-dim text-center leading-relaxed">
          Start a recording to collect training data. Recordings are saved as labeled audio clips
          for the training pipeline.
        </span>
      </div>
    )
  }

  function handleDelete(id: string) {
    deleteMutation.mutate(id, {
      onSuccess: () => setConfirmDeleteId(null),
    })
  }

  return (
    <div className="flex flex-col">
      {recordings.map((rec: Recording) => {
        const isEditing = editingId === rec.id
        const isConfirmingDelete = confirmDeleteId === rec.id
        const borderLeft = isEditing
          ? 'border-l-2 border-hud-accent'
          : !rec.labeled
            ? 'border-l-2 border-hud-warning'
            : 'border-l-2 border-transparent'

        return (
          <div key={rec.id} className={`bg-hud-panel border-b border-hud-border py-2 px-3 ${borderLeft}`}>
            <div className="flex items-center gap-2">
              <LabelBadge recording={rec} />
              <span className="font-mono text-xs text-hud-text-dim">{formatDuration(rec.duration_s)}</span>
              <span className="font-mono text-xs text-hud-text-dim flex-1 text-right">
                {formatTimestamp(rec.recorded_at)}
              </span>
              <button
                onClick={() => handlePlay(rec.id)}
                className={`${playingId === rec.id ? 'text-hud-accent' : 'text-hud-text-dim hover:text-hud-accent'}`}
                aria-label={playingId === rec.id ? 'Stop playback' : 'Play recording'}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
                  {playingId === rec.id ? 'stop' : 'play_arrow'}
                </span>
              </button>
              <button
                onClick={() => setEditingId(isEditing ? null : rec.id)}
                className="text-hud-text-dim hover:text-hud-accent"
                aria-label="Edit recording metadata"
              >
                <span className="material-symbols-outlined" style={{ fontSize: 20 }}>edit</span>
              </button>
              <button
                onClick={() => setConfirmDeleteId(isConfirmingDelete ? null : rec.id)}
                className="text-hud-text-dim hover:text-hud-danger"
                aria-label="Delete recording"
              >
                <span className="material-symbols-outlined" style={{ fontSize: 20 }}>delete</span>
              </button>
            </div>

            {isConfirmingDelete && (
              <div className="mt-2 text-xs text-hud-text-dim">
                <p>This recording and its metadata will be permanently removed. Delete?</p>
                <div className="flex gap-2 mt-1.5">
                  <button
                    onClick={() => handleDelete(rec.id)}
                    disabled={deleteMutation.isPending}
                    className="px-2 py-1 text-xs rounded bg-hud-danger text-white hover:opacity-90"
                  >
                    Delete
                  </button>
                  <button
                    onClick={() => setConfirmDeleteId(null)}
                    className="px-2 py-1 text-xs text-hud-text-dim hover:text-hud-text"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {isEditing && (
              <MetadataEditor recording={rec} onClose={() => setEditingId(null)} />
            )}
          </div>
        )
      })}
    </div>
  )
}
