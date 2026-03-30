import { Panel } from './Panel'
import type { HealthStatus } from '../../utils/types'

interface SidebarProps {
  health: HealthStatus | undefined
}

function StatusDot({ active }: { active: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${
        active ? 'bg-hud-success' : 'bg-hud-danger'
      }`}
    />
  )
}

function StatRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-hud-border last:border-b-0">
      <span className="text-xs text-hud-text-dim uppercase tracking-wider">{label}</span>
      <span className="text-sm font-mono text-hud-text">{children}</span>
    </div>
  )
}

function formatFrameAge(lastFrameTime: number | null | undefined): string {
  if (lastFrameTime == null) return 'N/A'
  const now = performance.now() / 1000
  // lastFrameTime is monotonic seconds from the backend
  // We can't directly compare, so display the raw value or "active"
  const age = Math.max(0, now - lastFrameTime)
  if (age > 86400) return 'N/A'
  if (age < 1) return '<1s ago'
  return `${Math.round(age)}s ago`
}

export function Sidebar({ health }: SidebarProps) {
  const pipelineRunning = health?.pipeline_running ?? false
  const deviceDetected = health?.device_detected ?? false

  return (
    <Panel title="SYSTEM" className="h-full">
      <div className="flex flex-col gap-0">
        <StatRow label="Pipeline">
          <span className="flex items-center gap-1.5">
            <StatusDot active={pipelineRunning} />
            {pipelineRunning ? 'Running' : 'Stopped'}
          </span>
        </StatRow>
        <StatRow label="Device">
          <span className="flex items-center gap-1.5">
            <StatusDot active={deviceDetected} />
            {deviceDetected ? 'Connected' : 'Not found'}
          </span>
        </StatRow>
        <StatRow label="Overflows">
          {health?.overflow_count ?? 0}
        </StatRow>
        <StatRow label="Last Frame">
          {formatFrameAge(health?.last_frame_time)}
        </StatRow>
      </div>
    </Panel>
  )
}
