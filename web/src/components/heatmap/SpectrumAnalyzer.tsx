import type { SpectrumData } from '../../hooks/useSpectrumSocket'

interface SpectrumAnalyzerProps {
  spectrum: SpectrumData | null
}

const DB_MIN = -100
const DB_MAX = 0

export function SpectrumAnalyzer({ spectrum }: SpectrumAnalyzerProps) {
  if (!spectrum) {
    return (
      <div className="flex items-center justify-center text-xs text-hud-text-dim/40" style={{ height: 100 }}>
        No signal
      </div>
    )
  }

  return (
    <div className="flex items-end gap-1" style={{ height: 100 }}>
      {spectrum.bands.map((band) => {
        const clamped = Math.max(DB_MIN, Math.min(DB_MAX, band.db))
        const pct = ((clamped - DB_MIN) / (DB_MAX - DB_MIN)) * 100
        const hue = pct > 60 ? 0 : pct > 30 ? 40 : 120
        return (
          <div key={band.name} className="flex-1 flex flex-col items-center gap-0.5 min-w-0 h-full">
            <span className="text-[8px] text-hud-text-dim tabular-nums leading-none">
              {Math.round(band.db)}
            </span>
            <div className="w-full bg-hud-border/30 rounded-sm relative flex-1">
              <div
                className="absolute bottom-0 left-0 right-0 rounded-sm transition-all duration-75"
                style={{
                  height: `${pct}%`,
                  backgroundColor: `hsl(${hue}, 80%, 50%)`,
                  opacity: 0.85,
                }}
              />
            </div>
            <span className="text-[9px] text-hud-text-dim truncate w-full text-center leading-none">
              {band.fmax >= 1000 ? `${(band.fmax / 1000).toFixed(0)}k` : band.fmax}
            </span>
          </div>
        )
      })}
    </div>
  )
}
