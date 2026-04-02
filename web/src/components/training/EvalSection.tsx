import { useState } from 'react'
import { useRunEvaluation } from '../../hooks/useEvaluation'
import { useModels } from '../../hooks/useModels'
import { EvaluationResults } from './EvaluationResults'

const INPUT_CLASS =
  'w-full bg-hud-bg border border-hud-border rounded px-2 py-1.5 text-sm text-hud-text focus:border-hud-accent focus:outline-none'

export function EvalSection() {
  const evalMutation = useRunEvaluation()
  const { data: modelsData } = useModels()

  const [selectedModel, setSelectedModel] = useState('')
  const [showDataDir, setShowDataDir] = useState(false)
  const [dataDir, setDataDir] = useState('')

  function handleRun() {
    evalMutation.mutate({
      model_path: selectedModel || undefined,
      data_dir: dataDir || undefined,
    })
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Model dropdown (per D-08) */}
      <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
        Model
      </label>
      <select
        value={selectedModel}
        onChange={e => setSelectedModel(e.target.value)}
        className="w-full bg-hud-bg border border-hud-border rounded px-2 py-1.5 text-sm text-hud-text"
      >
        <option value="">Default model</option>
        {modelsData?.models.map(m => (
          <option key={m.path} value={m.path}>
            {m.filename}
          </option>
        ))}
      </select>

      {/* Run Evaluation button */}
      <button
        onClick={handleRun}
        disabled={evalMutation.isPending}
        className={`w-full py-2 text-sm font-semibold rounded ${
          evalMutation.isPending
            ? 'bg-hud-border text-hud-text-dim cursor-not-allowed'
            : 'bg-hud-accent text-white hover:opacity-90'
        }`}
      >
        {evalMutation.isPending ? 'Evaluating...' : 'Run Evaluation'}
      </button>

      {/* Advanced toggle for data_dir */}
      <button
        type="button"
        onClick={() => setShowDataDir(v => !v)}
        className="text-xs text-hud-text-dim hover:text-hud-text cursor-pointer text-left"
      >
        {showDataDir ? '\u25BC Advanced' : '\u25B6 Advanced'}
      </button>

      {showDataDir && (
        <div>
          <label className="text-xs uppercase tracking-wider text-hud-text-dim font-semibold">
            Data Directory
          </label>
          <input
            type="text"
            placeholder="audio-data/data/"
            value={dataDir}
            onChange={e => setDataDir(e.target.value)}
            className={INPUT_CLASS}
          />
        </div>
      )}

      {/* Loading state */}
      {evalMutation.isPending && (
        <div className="flex items-center gap-2 text-xs text-hud-text-dim">
          <span className="inline-block w-2 h-2 rounded-full bg-hud-accent animate-pulse" />
          Evaluating...
        </div>
      )}

      {/* Error state */}
      {evalMutation.isError && (
        <div className="text-hud-danger text-xs">
          {evalMutation.error?.message ?? 'Evaluation failed'}
        </div>
      )}

      {/* Results */}
      {evalMutation.data && <EvaluationResults result={evalMutation.data} />}
    </div>
  )
}
