import { useEffect, useRef, useState } from 'react'
import { useModels } from '../../hooks/useModels'
import {
  usePipelineSocket,
  useStartDetection,
  useStopDetection,
} from '../../hooks/usePipeline'

function probColor(p: number): string {
  if (p > 0.8) return 'text-red-400'
  if (p > 0.5) return 'text-yellow-400'
  return 'text-hud-success'
}

function probBarColor(p: number): string {
  if (p > 0.8) return 'bg-red-500'
  if (p > 0.5) return 'bg-yellow-500'
  return 'bg-hud-success'
}

function DroneIndicator({ state, probability }: { state: string | null; probability: number | null }) {
  const isConfirmed = state === 'DRONE_CONFIRMED'
  const isCandidate = state === 'DRONE_CANDIDATE'

  return (
    <div
      className={`flex items-center gap-3 rounded-lg px-4 py-3 border ${
        isConfirmed
          ? 'bg-red-900/40 border-red-500 animate-pulse'
          : isCandidate
          ? 'bg-yellow-900/30 border-yellow-500'
          : 'bg-hud-bg border-hud-border'
      }`}
    >
      <div
        className={`w-4 h-4 rounded-full shrink-0 ${
          isConfirmed
            ? 'bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.7)]'
            : isCandidate
            ? 'bg-yellow-500'
            : 'bg-hud-text-dim/30'
        }`}
      />
      <div className="flex flex-col flex-1 min-w-0">
        <span
          className={`text-sm font-bold uppercase tracking-wider ${
            isConfirmed ? 'text-red-400' : isCandidate ? 'text-yellow-400' : 'text-hud-text-dim'
          }`}
        >
          {isConfirmed ? 'DRONE DETECTED' : isCandidate ? 'CANDIDATE' : 'NO DRONE'}
        </span>
      </div>
    </div>
  )
}

function CnnProbDisplay({ probability }: { probability: number | null }) {
  const prob = probability ?? 0
  const pct = Math.round(prob * 100)

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-hud-border bg-hud-bg px-3 py-2.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-hud-text-dim uppercase tracking-wider">CNN Prob</span>
        <span className={`text-2xl font-bold font-mono tabular-nums ${probability != null ? probColor(prob) : 'text-hud-text-dim/40'}`}>
          {probability != null ? `${pct}%` : '--'}
        </span>
      </div>
      <div className="w-full h-2.5 bg-hud-border/50 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 rounded-full ${probBarColor(prob)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function PipelinePanel() {
  const { data: modelsData } = useModels()
  const startMutation = useStartDetection()
  const stopMutation = useStopDetection()
  const live = usePipelineSocket()
  const logEndRef = useRef<HTMLDivElement>(null)

  // Form state
  const [modelPath, setModelPath] = useState('')
  const [confidence, setConfidence] = useState(90)
  const [timeFrame, setTimeFrame] = useState(2)
  const [positiveDetections, setPositiveDetections] = useState(2)
  const [gain, setGain] = useState(3)

  // Auto-select first model
  const models = modelsData?.models ?? []
  useEffect(() => {
    if (models.length > 0 && !modelPath) {
      const efficientat = models.find(
        (m) => m.filename.toLowerCase().includes('efficientat') || m.filename.toLowerCase().includes('mn10')
      )
      setModelPath(efficientat?.path ?? models[0].path)
    }
  }, [models, modelPath])

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [live.log.length])

  function handleStart() {
    if (!modelPath) return
    startMutation.mutate({
      model_path: modelPath,
      confidence: confidence / 100,
      time_frame: timeFrame,
      positive_detections: positiveDetections,
      gain,
    })
  }

  function handleStop() {
    stopMutation.mutate()
  }

  const isRunning = live.running
  const error = startMutation.error?.message || stopMutation.error?.message

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto">
      {/* Drone indicator */}
      <DroneIndicator state={live.detectionState} probability={live.droneProbability} />

      {/* CNN Probability - always visible when running */}
      {isRunning && <CnnProbDisplay probability={live.droneProbability} />}

      {/* Controls */}
      {!isRunning ? (
        <div className="flex flex-col gap-2">
          {/* Model selection */}
          <label className="text-xs text-hud-text-dim uppercase tracking-wider">Model</label>
          <select
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="bg-hud-bg border border-hud-border rounded px-2 py-1.5 text-sm text-hud-text font-mono"
          >
            {models.length === 0 && <option value="">No models found</option>}
            {models.map((m) => (
              <option key={m.path} value={m.path}>
                {m.filename}
              </option>
            ))}
          </select>

          {/* Confidence */}
          <label className="text-xs text-hud-text-dim uppercase tracking-wider">
            Confidence: {confidence}%
          </label>
          <input
            type="range"
            min={50}
            max={99}
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
            className="w-full accent-hud-accent"
          />

          {/* Time window */}
          <label className="text-xs text-hud-text-dim uppercase tracking-wider">
            Time window: {timeFrame}s
          </label>
          <input
            type="range"
            min={0.5}
            max={10}
            step={0.5}
            value={timeFrame}
            onChange={(e) => setTimeFrame(Number(e.target.value))}
            className="w-full accent-hud-accent"
          />

          {/* Positive detections */}
          <label className="text-xs text-hud-text-dim uppercase tracking-wider">
            Positive detections: {positiveDetections}
          </label>
          <input
            type="range"
            min={1}
            max={10}
            value={positiveDetections}
            onChange={(e) => setPositiveDetections(Number(e.target.value))}
            className="w-full accent-hud-accent"
          />

          {/* Gain */}
          <label className="text-xs text-hud-text-dim uppercase tracking-wider">
            Amplify gain: {gain}x
          </label>
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={gain}
            onChange={(e) => setGain(Number(e.target.value))}
            className="w-full accent-hud-accent"
          />

          {/* Start button */}
          <button
            onClick={handleStart}
            disabled={!modelPath || startMutation.isPending}
            className="mt-1 bg-hud-accent hover:bg-hud-accent/80 disabled:opacity-40 disabled:cursor-not-allowed text-hud-bg font-semibold text-sm py-2 px-4 rounded uppercase tracking-wider"
          >
            {startMutation.isPending ? 'Starting...' : 'Start Detection'}
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {/* Running status */}
          <div className="flex items-center gap-1.5 text-xs text-hud-text-dim">
            <span className="inline-block w-2 h-2 rounded-full bg-hud-success animate-pulse" />
            Detection active
          </div>

          {/* Stop button */}
          <button
            onClick={handleStop}
            disabled={stopMutation.isPending}
            className="bg-red-600 hover:bg-red-700 disabled:opacity-40 text-white font-semibold text-sm py-2 px-4 rounded uppercase tracking-wider"
          >
            {stopMutation.isPending ? 'Stopping...' : 'Stop Detection'}
          </button>
        </div>
      )}

      {error && (
        <p className="text-xs text-red-400 bg-red-900/20 rounded px-2 py-1">{error}</p>
      )}

      {/* Detection log */}
      {isRunning && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <span className="text-xs text-hud-text-dim uppercase tracking-wider">Detection Log</span>
            {live.log.length > 0 && (
              <span className="text-[10px] text-hud-text-dim/50">{live.log.length} entries</span>
            )}
          </div>
          <div className="max-h-56 overflow-y-auto bg-hud-bg border border-hud-border rounded p-2 font-mono text-[11px] leading-relaxed space-y-px">
            {live.log.length === 0 ? (
              <div className="text-hud-text-dim/40 text-center py-2">Waiting for CNN results...</div>
            ) : (
              live.log.map((entry, i) => {
                const time = new Date(entry.timestamp * 1000).toLocaleTimeString()
                const isConfirmed = entry.detection_state === 'DRONE_CONFIRMED'
                const isCandidate = entry.detection_state === 'DRONE_CANDIDATE'
                const isTransition = entry.message.includes('\u2192')
                const pct = Math.round(entry.drone_probability * 100)
                return (
                  <div
                    key={i}
                    className={`flex items-start gap-1.5 ${
                      isConfirmed
                        ? 'text-red-400'
                        : isCandidate
                        ? 'text-yellow-400'
                        : 'text-hud-text-dim/70'
                    } ${isTransition ? 'font-semibold' : ''}`}
                  >
                    <span className="text-hud-text-dim/40 shrink-0">{time}</span>
                    <span className={`shrink-0 w-8 text-right tabular-nums ${probColor(entry.drone_probability)}`}>
                      {pct}%
                    </span>
                    <span className="truncate">{entry.message}</span>
                  </div>
                )
              })
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  )
}
