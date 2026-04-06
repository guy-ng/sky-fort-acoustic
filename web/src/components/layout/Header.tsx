import { SoundLevelMeter } from './SoundLevelMeter'

export type PageKey = 'monitor' | 'tools'

interface HeaderProps {
  status: string
  pipelineRunning: boolean
  page: PageKey
  onPageChange: (p: PageKey) => void
}

const STATUS_DOT: Record<string, string> = {
  LIVE: 'bg-hud-success animate-pulse',
  'NO DEVICE': 'bg-amber-500',
  'CONNECTING...': 'bg-amber-500 animate-pulse',
  'TARGETS OFFLINE': 'bg-hud-warning',
}

function NavTab({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-xs uppercase tracking-wider font-semibold border-b-2 transition-colors ${
        active
          ? 'text-hud-text border-hud-accent'
          : 'text-hud-text-dim border-transparent hover:text-hud-text'
      }`}
    >
      {label}
    </button>
  )
}

export function Header({ status, pipelineRunning, page, onPageChange }: HeaderProps) {
  const dotClass = pipelineRunning ? 'bg-hud-success animate-pulse' : (STATUS_DOT[status] ?? 'bg-hud-danger')

  return (
    <header className="flex items-center justify-between px-4 py-2 bg-hud-panel border-b border-hud-border">
      <div className="flex items-center gap-6">
        <h1 className="font-mono font-semibold text-hud-accent text-lg tracking-wide">
          SKY FORT ACOUSTIC
        </h1>
        <nav className="flex items-center gap-1">
          <NavTab active={page === 'monitor'} label="Monitor" onClick={() => onPageChange('monitor')} />
          <NavTab active={page === 'tools'} label="Tools" onClick={() => onPageChange('tools')} />
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <SoundLevelMeter />
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2.5 h-2.5 rounded-full ${dotClass}`} />
          <span className="text-sm text-hud-text-dim font-mono">{status}</span>
        </div>
      </div>
    </header>
  )
}
