import { Panel } from '../layout/Panel'
import type { TargetState } from '../../utils/types'

interface TargetCardProps {
  target: TargetState
}

function confidenceBarColor(confidence: number): string {
  if (confidence >= 0.7) return 'bg-hud-success'
  if (confidence >= 0.4) return 'bg-hud-warning'
  return 'bg-hud-danger'
}

export function TargetCard({ target }: TargetCardProps) {
  return (
    <Panel className="shrink-0">
      <div className="flex flex-col gap-1.5">
        {/* Target ID */}
        <div className="font-mono text-sm text-hud-accent font-semibold">
          {target.id.slice(0, 8)}
        </div>

        {/* Class label */}
        <div className="text-xs text-hud-text-dim">
          {target.class_label}
        </div>

        {/* Drone probability */}
        <div className="flex justify-between text-xs">
          <span className="text-hud-text-dim">Probability</span>
          <span className={`font-mono font-semibold ${target.confidence >= 0.7 ? 'text-hud-success' : target.confidence >= 0.4 ? 'text-hud-warning' : 'text-hud-danger'}`}>
            {(target.confidence * 100).toFixed(1)}%
          </span>
        </div>

        {/* Confidence bar */}
        <div>
          <div className="h-1.5 w-full bg-hud-border rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${confidenceBarColor(target.confidence)}`}
              style={{ width: `${Math.round(target.confidence * 100)}%` }}
            />
          </div>
        </div>

        {/* Speed */}
        <div className="flex justify-between text-xs">
          <span className="text-hud-text-dim">Speed</span>
          <span className="font-mono text-hud-text">
            {target.speed_mps !== null ? `${target.speed_mps.toFixed(1)} m/s` : '--'}
          </span>
        </div>

        {/* Bearing */}
        <div className="flex justify-between text-xs">
          <span className="text-hud-text-dim">Bearing</span>
          <span className="font-mono text-hud-text">
            Az: {target.az_deg.toFixed(1)} El: {target.el_deg.toFixed(1)}
          </span>
        </div>
      </div>
    </Panel>
  )
}
