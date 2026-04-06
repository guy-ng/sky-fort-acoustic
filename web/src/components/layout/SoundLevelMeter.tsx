import { useSoundLevelSocket } from '../../hooks/useSoundLevelSocket'

// Map dBFS to 0..1 for the bar fill. -80 dB = empty, 0 dB = full.
function dbToFill(db: number | null): number {
  if (db === null || !Number.isFinite(db)) return 0
  const min = -80
  const max = 0
  return Math.max(0, Math.min(1, (db - min) / (max - min)))
}

function fillColor(fill: number): string {
  if (fill > 0.85) return 'bg-hud-danger'
  if (fill > 0.6) return 'bg-hud-warning'
  return 'bg-hud-success'
}

export function SoundLevelMeter() {
  const { level_db, connected } = useSoundLevelSocket()
  const fill = dbToFill(level_db)
  const color = fillColor(fill)
  const label = level_db === null ? '—' : `${level_db.toFixed(0)} dB`

  return (
    <div className="flex items-center gap-2 min-w-[140px]">
      <span className="text-[10px] text-hud-text-dim uppercase tracking-wider">Level</span>
      <div className="relative flex-1 h-2 bg-hud-bg border border-hud-border rounded-sm overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 ${color} transition-[width] duration-75`}
          style={{ width: `${fill * 100}%` }}
        />
      </div>
      <span className={`text-[10px] font-mono tabular-nums w-12 text-right ${connected ? 'text-hud-text-dim' : 'text-hud-text-dim/40'}`}>
        {label}
      </span>
    </div>
  )
}
