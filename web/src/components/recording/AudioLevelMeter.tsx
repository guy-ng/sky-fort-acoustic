interface AudioLevelMeterProps {
  level_db: number
}

export function AudioLevelMeter({ level_db }: AudioLevelMeterProps) {
  // Map -60dB..0dB to 0%..100% width
  const pct = Math.max(0, Math.min(100, ((level_db + 60) / 60) * 100))
  // Color: hud-success (low) -> hud-warning (mid) -> hud-danger (high)
  const color =
    level_db > -6
      ? 'bg-hud-danger'
      : level_db > -20
        ? 'bg-hud-warning'
        : 'bg-hud-success'

  return (
    <div className="w-full h-1.5 bg-hud-border rounded-full overflow-hidden">
      <div
        className={`h-full ${color} transition-all duration-100`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
