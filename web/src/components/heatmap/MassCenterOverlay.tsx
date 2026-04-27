import type { MassCenter } from '../../hooks/useBfPeaksSocket'
import type { HeatmapHandshake } from '../../utils/types'

interface MassCenterOverlayProps {
  massCenter: MassCenter | null
  gridInfo: HeatmapHandshake | null
}

export function MassCenterOverlay({ massCenter, gridInfo }: MassCenterOverlayProps) {
  if (!massCenter || !gridInfo) return null

  const azRange = gridInfo.az_max - gridInfo.az_min
  const elRange = gridInfo.el_max - gridInfo.el_min

  // Center point as percentage
  const cx = ((massCenter.az_deg - gridInfo.az_min) / azRange) * 100
  const cy = ((gridInfo.el_max - massCenter.el_deg) / elRange) * 100

  // Extent box as percentage
  const boxLeft = ((massCenter.az_min - gridInfo.az_min) / azRange) * 100
  const boxRight = ((massCenter.az_max - gridInfo.az_min) / azRange) * 100
  const boxTop = ((gridInfo.el_max - massCenter.el_max) / elRange) * 100
  const boxBottom = ((gridInfo.el_max - massCenter.el_min) / elRange) * 100

  const boxW = boxRight - boxLeft
  const boxH = boxBottom - boxTop

  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Extent bounding box */}
      {boxW > 0 && boxH > 0 && (
        <div
          className="absolute border border-cyan-400/40 rounded-sm"
          style={{
            left: `${boxLeft}%`,
            top: `${boxTop}%`,
            width: `${boxW}%`,
            height: `${boxH}%`,
          }}
        />
      )}

      {/* Crosshair lines */}
      <div
        className="absolute w-px bg-cyan-400/60"
        style={{ left: `${cx}%`, top: `${Math.max(cy - 6, 0)}%`, height: '12%' }}
      />
      <div
        className="absolute h-px bg-cyan-400/60"
        style={{ top: `${cy}%`, left: `${Math.max(cx - 6, 0)}%`, width: '12%' }}
      />

      {/* Center dot */}
      <div
        className="absolute w-2 h-2 rounded-full bg-cyan-400 -translate-x-1/2 -translate-y-1/2"
        style={{ left: `${cx}%`, top: `${cy}%` }}
      />

      {/* Label */}
      <div
        className="absolute text-[10px] font-mono text-cyan-400 whitespace-nowrap"
        style={{ left: `${cx + 1.5}%`, top: `${cy - 3}%` }}
      >
        {massCenter.az_deg > 0 ? '+' : ''}{massCenter.az_deg.toFixed(1)}° / {massCenter.el_deg.toFixed(1)}°
      </div>
    </div>
  )
}
