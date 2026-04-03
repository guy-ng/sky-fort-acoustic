interface HeaderProps {
  status: string
  pipelineRunning: boolean
}

const STATUS_DOT: Record<string, string> = {
  LIVE: 'bg-hud-success animate-pulse',
  'NO DEVICE': 'bg-amber-500',
  'CONNECTING...': 'bg-amber-500 animate-pulse',
  'TARGETS OFFLINE': 'bg-hud-warning',
}

export function Header({ status, pipelineRunning }: HeaderProps) {
  const dotClass = pipelineRunning ? 'bg-hud-success animate-pulse' : (STATUS_DOT[status] ?? 'bg-hud-danger')

  return (
    <header className="flex items-center justify-between px-4 py-2 bg-hud-panel border-b border-hud-border">
      <h1 className="font-mono font-semibold text-hud-accent text-lg tracking-wide">
        SKY FORT ACOUSTIC
      </h1>
      <div className="flex items-center gap-2">
        <span className={`inline-block w-2.5 h-2.5 rounded-full ${dotClass}`} />
        <span className="text-sm text-hud-text-dim font-mono">{status}</span>
      </div>
    </header>
  )
}
