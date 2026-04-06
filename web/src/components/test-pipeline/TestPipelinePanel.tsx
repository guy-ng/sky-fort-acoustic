import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

interface TestSample {
  id: string
  label: 'drone' | 'background'
  duration_s: number
  filename: string
}

interface SamplesResponse {
  samples: TestSample[]
  count: number
}

function formatDuration(seconds: number): string {
  return `${seconds.toFixed(1)}s`
}

export function TestPipelinePanel() {
  const qc = useQueryClient()
  const [playingId, setPlayingId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [filter, setFilter] = useState<'all' | 'drone' | 'background'>('all')

  const { data, isLoading } = useQuery<SamplesResponse>({
    queryKey: ['test-samples'],
    queryFn: () => fetch('/api/test-pipeline/samples').then(r => r.json()),
  })

  const prepareMutation = useMutation({
    mutationFn: () =>
      fetch('/api/test-pipeline/prepare', { method: 'POST' }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['test-samples'] }),
  })

  function handlePlay(id: string) {
    if (playingId === id) {
      audioRef.current?.pause()
      setPlayingId(null)
      return
    }
    if (audioRef.current) {
      audioRef.current.pause()
    }
    const audio = new Audio(`/api/test-pipeline/samples/${id}/audio`)
    audio.onended = () => setPlayingId(null)
    audio.onerror = () => setPlayingId(null)
    audio.play()
    audioRef.current = audio
    setPlayingId(id)
  }

  const samples = data?.samples ?? []
  const hasSamples = samples.length > 0

  const filtered =
    filter === 'all' ? samples : samples.filter(s => s.label === filter)

  const droneCount = samples.filter(s => s.label === 'drone').length
  const bgCount = samples.filter(s => s.label === 'background').length

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <p className="text-xs text-hud-text-dim leading-relaxed">
          Play DADS dataset samples through speakers to test the detection pipeline with UMA-16.
        </p>
        <button
          onClick={() => prepareMutation.mutate()}
          disabled={prepareMutation.isPending}
          className="w-full px-3 py-2 text-xs font-semibold uppercase tracking-wider rounded bg-hud-accent text-hud-bg hover:opacity-90 disabled:opacity-50"
        >
          {prepareMutation.isPending
            ? 'Preparing samples...'
            : hasSamples
              ? 'Regenerate samples'
              : 'Prepare test samples'}
        </button>
        {prepareMutation.isError && (
          <p className="text-xs text-hud-danger">
            Failed to prepare samples. Check that DADS parquet files exist in data/.
          </p>
        )}
      </div>

      {isLoading && (
        <div className="text-sm text-hud-text-dim text-center py-4">Loading...</div>
      )}

      {!isLoading && !hasSamples && (
        <div className="flex flex-col items-center justify-center py-6 gap-2">
          <span className="text-sm font-semibold text-hud-text">No Test Samples</span>
          <span className="text-xs text-hud-text-dim text-center leading-relaxed">
            Click &quot;Prepare test samples&quot; to extract 20 drone + 20 background clips (3-5s each) from the DADS dataset.
          </span>
        </div>
      )}

      {hasSamples && (
        <>
          {/* Filter + stats */}
          <div className="flex items-center gap-2">
            <div className="flex gap-1 flex-1">
              {(['all', 'drone', 'background'] as const).map(f => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-2 py-1 text-xs rounded uppercase tracking-wider ${
                    filter === f
                      ? 'bg-hud-accent text-hud-bg font-semibold'
                      : 'text-hud-text-dim hover:text-hud-text'
                  }`}
                >
                  {f === 'all' ? `All (${samples.length})` : f === 'drone' ? `Drone (${droneCount})` : `BG (${bgCount})`}
                </button>
              ))}
            </div>
          </div>

          {/* Sample list */}
          <div className="flex flex-col overflow-y-auto flex-1">
            {filtered.map(sample => {
              const isDrone = sample.label === 'drone'
              const isPlaying = playingId === sample.id
              return (
                <div
                  key={sample.id}
                  className={`flex items-center gap-2 py-2 px-3 border-b border-hud-border border-l-2 ${
                    isDrone ? 'border-l-hud-danger' : 'border-l-hud-success'
                  }`}
                >
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      isDrone
                        ? 'bg-hud-danger/20 text-hud-danger'
                        : 'bg-hud-success/20 text-hud-success'
                    }`}
                  >
                    {sample.label}
                  </span>
                  <span className="font-mono text-xs text-hud-text-dim flex-1">
                    {formatDuration(sample.duration_s)}
                  </span>
                  <button
                    onClick={() => handlePlay(sample.id)}
                    className={`${
                      isPlaying
                        ? 'text-hud-accent'
                        : 'text-hud-text-dim hover:text-hud-accent'
                    }`}
                    aria-label={isPlaying ? 'Stop playback' : 'Play sample'}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 22 }}>
                      {isPlaying ? 'stop_circle' : 'play_circle'}
                    </span>
                  </button>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
