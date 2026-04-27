import { useCallback, useEffect, useState } from 'react'
import type { PlaybackState } from '../../hooks/useBfPeaksSocket'

interface RawRecording {
  id: string
  filename: string
  channels: number
  sample_rate: number
  duration_s: number
  size_bytes: number
}

interface RawRecordingsProps {
  playbackState: PlaybackState
}

const API_BASE = ''

export function RawRecordings({ playbackState }: RawRecordingsProps) {
  const [recordings, setRecordings] = useState<RawRecording[]>([])
  const [loading, setLoading] = useState(false)

  const fetchRecordings = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/raw-recordings`)
      if (res.ok) setRecordings(await res.json())
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => { fetchRecordings() }, [fetchRecordings])

  const handlePlay = useCallback(async (id: string) => {
    await fetch(`${API_BASE}/api/raw-recordings/${id}/playback`, { method: 'POST' })
  }, [])

  const handleStop = useCallback(async () => {
    await fetch(`${API_BASE}/api/raw-recordings/stop-playback`, { method: 'POST' })
  }, [])

  const isPlaying = playbackState.status === 'playing'

  if (loading) {
    return <div className="text-xs text-hud-text-dim">Loading...</div>
  }

  if (recordings.length === 0) {
    return (
      <div className="text-xs text-hud-text-dim/40 text-center py-2">
        No recordings yet
      </div>
    )
  }

  return (
    <div className="space-y-1 text-xs">
      {isPlaying && (
        <button
          onClick={handleStop}
          className="w-full py-1 rounded bg-hud-danger/80 hover:bg-hud-danger text-white text-xs font-medium uppercase tracking-wider flex items-center justify-center gap-1 mb-1"
        >
          <span className="material-symbols-outlined text-sm">stop</span>
          Stop Playback
        </button>
      )}
      {recordings.map((rec) => {
        const sizeMb = (rec.size_bytes / 1048576).toFixed(1)
        const isThisPlaying = isPlaying && playbackState.path?.includes(rec.id)
        return (
          <div
            key={rec.id}
            className={`flex items-center gap-2 px-2 py-1 rounded ${
              isThisPlaying ? 'bg-hud-accent/20 border border-hud-accent/40' : 'bg-hud-border/20 hover:bg-hud-border/40'
            }`}
          >
            <button
              onClick={() => isThisPlaying ? handleStop() : handlePlay(rec.id)}
              className="text-hud-accent hover:text-hud-text shrink-0"
              title={isThisPlaying ? 'Stop' : 'Play through pipeline'}
            >
              <span className="material-symbols-outlined text-base">
                {isThisPlaying ? 'stop_circle' : 'play_circle'}
              </span>
            </button>
            <div className="flex-1 min-w-0">
              <div className="truncate text-hud-text tabular-nums">{rec.id}</div>
              <div className="text-hud-text-dim tabular-nums">
                {rec.duration_s}s · {rec.channels}ch · {sizeMb}MB
              </div>
            </div>
          </div>
        )
      })}
      <button
        onClick={fetchRecordings}
        className="w-full text-center text-hud-text-dim hover:text-hud-text text-[10px] uppercase tracking-wider py-0.5"
      >
        Refresh
      </button>
    </div>
  )
}
