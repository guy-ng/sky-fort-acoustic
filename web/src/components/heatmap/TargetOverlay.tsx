import type { TargetState, HeatmapHandshake } from '../../utils/types'

interface TargetOverlayProps {
  targets: TargetState[]
  gridInfo: HeatmapHandshake | null
}

function targetToPercent(
  target: TargetState,
  gridInfo: HeatmapHandshake,
): { left: number; top: number } {
  // Map azimuth to horizontal position (left to right: az_min to az_max)
  const left = ((target.az_deg - gridInfo.az_min) / (gridInfo.az_max - gridInfo.az_min)) * 100
  // Map elevation to vertical position (top to bottom: el_max to el_min — top is high elevation)
  const top = ((gridInfo.el_max - target.el_deg) / (gridInfo.el_max - gridInfo.el_min)) * 100
  return { left, top }
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.7) return 'border-hud-success'
  if (confidence >= 0.4) return 'border-hud-warning'
  return 'border-hud-danger'
}

export function TargetOverlay({ targets, gridInfo }: TargetOverlayProps) {
  if (!gridInfo) return null

  return (
    <div className="absolute inset-0 pointer-events-none">
      {targets.map((target) => {
        const { left, top } = targetToPercent(target, gridInfo)
        return (
          <div
            key={target.id}
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${left}%`, top: `${top}%` }}
          >
            {/* Pulsing marker circle */}
            <div
              className={`w-3 h-3 rounded-full border-2 ${confidenceColor(target.confidence)} bg-transparent animate-ping`}
              style={{ animationDuration: '2s' }}
            />
            <div
              className={`absolute inset-0 w-3 h-3 rounded-full border-2 ${confidenceColor(target.confidence)} bg-transparent`}
            />
            {/* Target ID label */}
            <span className="absolute left-4 -top-1 text-[10px] font-mono text-hud-accent whitespace-nowrap">
              {target.id.slice(0, 8)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
