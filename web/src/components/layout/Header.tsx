interface HeaderProps {
  status: string
  pipelineRunning: boolean
}

export function Header({ status, pipelineRunning }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 py-2 bg-hud-panel border-b border-hud-border">
      <h1 className="font-mono font-semibold text-hud-accent text-lg tracking-wide">
        SKY FORT ACOUSTIC
      </h1>
      <div className="flex items-center gap-2">
        <span
          className={`inline-block w-2.5 h-2.5 rounded-full ${
            pipelineRunning ? 'bg-hud-success animate-pulse' : 'bg-hud-danger'
          }`}
        />
        <span className="text-sm text-hud-text-dim font-mono">{status}</span>
      </div>
    </header>
  )
}
