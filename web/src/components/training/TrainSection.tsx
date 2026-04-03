import { useState, useRef, useEffect } from 'react'
import { useStartTraining, useCancelTraining } from '../../hooks/useTraining'
import { useTrainingSocket } from '../../hooks/useTrainingSocket'
import { useRunEvaluation } from '../../hooks/useEvaluation'
import { useQueryClient } from '@tanstack/react-query'
import { TrainingProgress } from './TrainingProgress'
import { Tooltip } from './Tooltip'
import type { TrainingStatus } from '../../utils/types'

const INPUT_CLASS =
  'w-full bg-hud-bg border border-hud-border rounded px-2 py-1.5 text-sm text-hud-text focus:border-hud-accent focus:outline-none'

const STATUS_COLORS: Record<TrainingStatus, { dot: string; text: string }> = {
  idle: { dot: 'bg-gray-500', text: 'Idle' },
  running: { dot: 'bg-hud-success animate-pulse', text: 'Training...' },
  completed: { dot: 'bg-hud-success', text: 'Completed' },
  failed: { dot: 'bg-hud-danger', text: 'Failed' },
  cancelled: { dot: 'bg-hud-warning', text: 'Cancelled' },
}

export function TrainSection() {
  const startMutation = useStartTraining()
  const cancelMutation = useCancelTraining()
  const evalMutation = useRunEvaluation()
  const { state: wsState, lossHistory } = useTrainingSocket()
  const qc = useQueryClient()

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [lr, setLr] = useState('0.001')
  const [batchSize, setBatchSize] = useState('32')
  const [epochs, setEpochs] = useState('50')
  const [patience, setPatience] = useState('5')
  const [augEnabled, setAugEnabled] = useState(true)
  const [dataRoot, setDataRoot] = useState('data/field')

  // Auto-eval tracking (per D-07)
  const autoEvalFired = useRef(false)
  const prevStatus = useRef<TrainingStatus>(wsState.status)

  useEffect(() => {
    if (prevStatus.current === 'running' && wsState.status === 'completed' && !autoEvalFired.current) {
      autoEvalFired.current = true
      evalMutation.mutate({})
      qc.invalidateQueries({ queryKey: ['models'] })
    }
    if (wsState.status === 'running') {
      autoEvalFired.current = false
    }
    prevStatus.current = wsState.status
  }, [wsState.status, evalMutation, qc])

  function handleStart() {
    startMutation.mutate({
      learning_rate: Number(lr),
      batch_size: Number(batchSize),
      max_epochs: Number(epochs),
      patience: Number(patience),
      augmentation_enabled: augEnabled,
      data_root: dataRoot,
    })
  }

  const status = wsState.status
  const statusStyle = STATUS_COLORS[status] ?? STATUS_COLORS.idle

  return (
    <div className="flex flex-col gap-2">
      {/* Status indicator -- always visible (per D-05) */}
      <div className="flex items-center gap-2">
        <span className={`inline-block w-2 h-2 rounded-full ${statusStyle.dot}`} />
        <span className="text-xs text-hud-text-dim">{statusStyle.text}</span>
        {wsState.error && (
          <span className="text-xs text-hud-danger truncate ml-auto">{wsState.error}</span>
        )}
      </div>

      {/* Start Training button */}
      <button
        onClick={handleStart}
        disabled={status === 'running' || startMutation.isPending}
        className={`w-full py-2 text-sm font-semibold rounded ${
          status === 'running'
            ? 'bg-hud-border text-hud-text-dim cursor-not-allowed'
            : 'bg-hud-accent text-white hover:opacity-90'
        }`}
      >
        {startMutation.isPending ? 'Starting...' : 'Start Training'}
      </button>

      {/* Cancel button -- visible only when running (per D-05) */}
      {status === 'running' && (
        <button
          onClick={() => cancelMutation.mutate()}
          disabled={cancelMutation.isPending}
          className="bg-hud-danger text-white text-sm rounded px-3 py-1.5"
        >
          {cancelMutation.isPending ? 'Cancelling...' : 'Cancel'}
        </button>
      )}

      {/* Advanced hyperparameters toggle (per D-03) */}
      <button
        type="button"
        onClick={() => setShowAdvanced(v => !v)}
        className="text-xs text-hud-text-dim hover:text-hud-text cursor-pointer text-left"
      >
        {showAdvanced ? '\u25BC Advanced' : '\u25B6 Advanced'}
      </button>

      {showAdvanced && (
        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Tooltip text="Step size for gradient descent (e.g. 0.001)">
                <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
                  Learning Rate
                </label>
              </Tooltip>
              <input
                type="text"
                value={lr}
                onChange={e => setLr(e.target.value)}
                className={INPUT_CLASS}
              />
            </div>
            <div>
              <Tooltip text="Number of samples per training step">
                <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
                  Batch Size
                </label>
              </Tooltip>
              <input
                type="text"
                value={batchSize}
                onChange={e => setBatchSize(e.target.value)}
                className={INPUT_CLASS}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Tooltip text="Maximum training passes over the dataset">
                <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
                  Max Epochs
                </label>
              </Tooltip>
              <input
                type="text"
                value={epochs}
                onChange={e => setEpochs(e.target.value)}
                className={INPUT_CLASS}
              />
            </div>
            <div>
              <Tooltip text="Epochs without improvement before early stop">
                <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
                  Patience
                </label>
              </Tooltip>
              <input
                type="text"
                value={patience}
                onChange={e => setPatience(e.target.value)}
                className={INPUT_CLASS}
              />
            </div>
          </div>
          <div>
            <Tooltip text="Path to labeled audio data directory">
              <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
                Data Root
              </label>
            </Tooltip>
            <input
              type="text"
              value={dataRoot}
              onChange={e => setDataRoot(e.target.value)}
              className={INPUT_CLASS}
            />
          </div>
          <label className="flex items-center gap-2 text-xs text-hud-text-dim cursor-pointer">
            <input
              type="checkbox"
              checked={augEnabled}
              onChange={e => setAugEnabled(e.target.checked)}
              className="accent-hud-accent"
            />
            <Tooltip text="Apply random gain and noise to training data">
              <span className="uppercase tracking-wider font-semibold">Augmentation</span>
            </Tooltip>
          </label>
        </div>
      )}

      {/* Training progress -- shown when running or completed (per D-04) */}
      {(status === 'running' || status === 'completed') && (
        <TrainingProgress state={wsState} lossHistory={lossHistory} />
      )}
    </div>
  )
}
