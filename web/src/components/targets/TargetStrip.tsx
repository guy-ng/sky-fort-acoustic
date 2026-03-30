import { TargetCard } from './TargetCard'
import type { TargetState } from '../../utils/types'

interface TargetStripProps {
  targets: TargetState[]
}

export function TargetStrip({ targets }: TargetStripProps) {
  if (targets.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="text-sm text-hud-text-dim font-mono">No targets detected</span>
      </div>
    )
  }

  return (
    <div className="flex gap-2 items-start overflow-x-auto h-full p-1">
      {targets.map((target) => (
        <TargetCard key={target.id} target={target} />
      ))}
    </div>
  )
}
