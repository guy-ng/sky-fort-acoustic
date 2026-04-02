import { useState } from 'react'
import type { EvalResultResponse, ConfusionMatrixData } from '../../utils/types'

function ConfusionMatrix({ cm }: { cm: ConfusionMatrixData }) {
  return (
    <div className="grid grid-cols-2 gap-1 text-center text-xs">
      <div className="bg-hud-success/20 text-hud-success p-2 rounded">
        <span className="font-mono text-lg">{cm.tp}</span>
        <div>TP</div>
      </div>
      <div className="bg-hud-danger/20 text-hud-danger p-2 rounded">
        <span className="font-mono text-lg">{cm.fp}</span>
        <div>FP</div>
      </div>
      <div className="bg-hud-danger/20 text-hud-danger p-2 rounded">
        <span className="font-mono text-lg">{cm.fn}</span>
        <div>FN</div>
      </div>
      <div className="bg-hud-success/20 text-hud-success p-2 rounded">
        <span className="font-mono text-lg">{cm.tn}</span>
        <div>TN</div>
      </div>
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-hud-bg/50 border border-hud-border rounded p-2 text-center">
      <div className="font-mono text-lg text-hud-text">{value}</div>
      <div className="text-xs text-hud-text-dim uppercase">{label}</div>
    </div>
  )
}

export function EvaluationResults({ result }: { result: EvalResultResponse }) {
  const [showFiles, setShowFiles] = useState(false)
  const { summary, distribution, per_file } = result

  return (
    <div className="flex flex-col gap-3">
      {/* Summary message */}
      <div className="text-xs text-hud-text-dim">{result.message}</div>

      {/* Metrics cards (per D-06) */}
      <div className="grid grid-cols-2 gap-2">
        <MetricCard label="Accuracy" value={`${(summary.accuracy * 100).toFixed(1)}%`} />
        <MetricCard label="Precision" value={`${(summary.precision * 100).toFixed(1)}%`} />
        <MetricCard label="Recall" value={`${(summary.recall * 100).toFixed(1)}%`} />
        <MetricCard label="F1" value={`${(summary.f1 * 100).toFixed(1)}%`} />
      </div>

      {/* Confusion matrix */}
      <ConfusionMatrix cm={summary.confusion_matrix} />

      {/* Distribution stats */}
      {Object.entries(distribution).map(([cls, dist]) => (
        <div key={cls} className="text-xs">
          <span className="uppercase tracking-wider text-hud-text-dim font-semibold">{cls}</span>
          <div className="flex gap-3 mt-0.5 text-hud-text">
            <span>p50: {dist.p_agg.p50.toFixed(3)}</span>
            <span>p95: {dist.p_agg.p95.toFixed(3)}</span>
          </div>
        </div>
      ))}

      {/* Per-file results (collapsible) */}
      <button
        type="button"
        onClick={() => setShowFiles(v => !v)}
        className="text-xs text-hud-text-dim hover:text-hud-text cursor-pointer text-left"
      >
        {showFiles ? `Hide file results` : `Show ${per_file.length} file results`}
      </button>

      {showFiles && (
        <table className="text-xs w-full">
          <thead>
            <tr className="text-hud-text-dim uppercase tracking-wider border-b border-hud-border">
              <th className="text-left py-1">File</th>
              <th className="text-left py-1">True</th>
              <th className="text-left py-1">Pred</th>
              <th className="text-right py-1">p_agg</th>
              <th className="text-center py-1"></th>
            </tr>
          </thead>
          <tbody>
            {per_file.map((f, i) => (
              <tr
                key={i}
                className={`border-b border-hud-border/50 ${
                  f.correct ? '' : 'text-hud-danger'
                }`}
              >
                <td className="py-1 font-mono truncate max-w-[120px]">{f.filename}</td>
                <td className="py-1">{f.true_label}</td>
                <td className="py-1">{f.predicted_label}</td>
                <td className="py-1 text-right font-mono">{f.p_agg.toFixed(3)}</td>
                <td className="py-1 text-center">{f.correct ? '\u2713' : '\u2717'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
