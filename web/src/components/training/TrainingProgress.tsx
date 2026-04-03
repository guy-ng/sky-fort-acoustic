import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { TrainingWsMessage, LossDataPoint, ConfusionMatrixData } from '../../utils/types'

interface TrainingProgressProps {
  state: TrainingWsMessage
  lossHistory: LossDataPoint[]
}

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

export function TrainingProgress({ state, lossHistory }: TrainingProgressProps) {
  const bestValLoss = lossHistory.length > 0
    ? Math.min(...lossHistory.map(d => d.val_loss))
    : undefined

  return (
    <div className="flex flex-col gap-2">
      {/* Loss chart (per D-04) */}
      {lossHistory.length > 0 && (
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={lossHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="epoch"
              tick={{ fontSize: 10, fill: '#9ca3af' }}
            />
            <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#111827',
                border: '1px solid #1f2937',
                fontSize: 12,
              }}
              labelStyle={{ color: '#e5e7eb' }}
              itemStyle={{ color: '#e5e7eb' }}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Line
              type="monotone"
              dataKey="train_loss"
              stroke="#3b82f6"
              dot={false}
              name="Train"
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="#f59e0b"
              dot={false}
              name="Val"
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Epoch + batch progress */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between text-xs text-hud-text-dim">
          <span>
            Epoch {state.epoch ?? '-'}/{state.total_epochs ?? '-'}
          </span>
          <span>
            Val Acc: {state.val_acc != null ? `${(state.val_acc * 100).toFixed(1)}%` : '-'}
          </span>
          <span>
            Best Loss: {bestValLoss != null ? bestValLoss.toFixed(4) : '-'}
          </span>
        </div>
        {/* Batch progress bar (intra-epoch) */}
        {state.status === 'running' && state.total_batches != null && state.total_batches > 0 && (
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-hud-border rounded-full overflow-hidden">
              <div
                className="h-full bg-hud-accent rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, ((state.batch ?? 0) / state.total_batches) * 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-hud-text-dim font-mono whitespace-nowrap">
              {state.batch ?? 0}/{state.total_batches}
            </span>
          </div>
        )}
      </div>

      {/* Confusion matrix */}
      {state.confusion_matrix && (
        <ConfusionMatrix cm={state.confusion_matrix} />
      )}
    </div>
  )
}
