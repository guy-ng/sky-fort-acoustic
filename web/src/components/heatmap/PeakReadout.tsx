import type { BfPeaksData } from '../../hooks/useBfPeaksSocket'

interface PeakReadoutProps {
  data: BfPeaksData | null
}

function Val({ v, suffix = '°' }: { v: number | undefined | null; suffix?: string }) {
  if (v == null) return <span className="text-hud-text-dim/30 tabular-nums">---{suffix}</span>
  return (
    <span className="tabular-nums">
      {v > 0 ? '+' : ''}{v.toFixed(1)}{suffix}
    </span>
  )
}

export function PeakReadout({ data }: PeakReadoutProps) {
  const mc = data?.mass_center ?? null

  return (
    <div className="space-y-1.5 text-xs font-mono">
      <div className="text-hud-text-dim uppercase text-[10px] mb-1">Sound Center</div>

      {/* Az */}
      <div className="flex items-baseline gap-2">
        <span className="text-hud-text-dim w-6">Az</span>
        <span className="text-red-400 text-sm font-bold">
          <Val v={mc?.az_deg} />
        </span>
        <span className="text-hud-text-dim text-[10px] ml-auto">
          [<Val v={mc?.az_min} /> .. <Val v={mc?.az_max} />]
        </span>
      </div>

      {/* El */}
      <div className="flex items-baseline gap-2">
        <span className="text-hud-text-dim w-6">El</span>
        <span className="text-red-400 text-sm font-bold">
          <Val v={mc?.el_deg} />
        </span>
        <span className="text-hud-text-dim text-[10px] ml-auto">
          [<Val v={mc?.el_min} /> .. <Val v={mc?.el_max} />]
        </span>
      </div>

      {/* Spread */}
      <div className="flex items-baseline gap-2 text-hud-text-dim text-[10px]">
        <span>Spread</span>
        <span className="tabular-nums">
          Az {mc ? (mc.az_max - mc.az_min).toFixed(0) : '---'}° × El {mc ? (mc.el_max - mc.el_min).toFixed(0) : '---'}°
        </span>
      </div>
    </div>
  )
}
