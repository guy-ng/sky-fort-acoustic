import { useState } from 'react'
import { Panel } from './Panel'
import type { HealthStatus } from '../../utils/types'
import type { DeviceStatusState } from '../../hooks/useDeviceStatus'
import { RecordingPanel } from '../recording/RecordingPanel'
import { RecordingsList } from '../recording/RecordingsList'
import { useRecordingsList } from '../../hooks/useRecordings'

interface SidebarProps {
  health: HealthStatus | undefined
  deviceStatus: DeviceStatusState
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
  const age = Math.max(0, now - lastFrameTime)
  if (age > 86400) return 'N/A'
  if (age < 1) return '<1s ago'
  return `${Math.round(age)}s ago`
}

function deviceLabel(ds: DeviceStatusState): string {
  if (ds.detected) return ds.name ?? 'Connected'
  if (ds.scanning) return 'Scanning...'
  return 'No device'
}

export function Sidebar({ health, deviceStatus }: SidebarProps) {
  const [tab, setTab] = useState<'system' | 'recordings'>('system')
  const { data: recordings } = useRecordingsList()
  const pipelineRunning = health?.pipeline_running ?? false

  const unlabeledCount = recordings?.filter(r => !r.labeled).length ?? 0

  return (
    <Panel className="h-full">
      {/* Tab header */}
      <div className="flex border-b border-hud-border -mx-3 -mt-3 mb-3">
        <button
          onClick={() => setTab('system')}
          className={`flex-1 px-3 py-2 text-xs uppercase tracking-wider font-semibold ${
            tab === 'system'
              ? 'text-hud-text border-b-2 border-hud-accent'
              : 'text-hud-text-dim hover:text-hud-text'
          }`}
        >
          SYSTEM
        </button>
        <button
          onClick={() => setTab('recordings')}
          className={`flex-1 px-3 py-2 text-xs uppercase tracking-wider font-semibold relative ${
            tab === 'recordings'
              ? 'text-hud-text border-b-2 border-hud-accent'
              : 'text-hud-text-dim hover:text-hud-text'
          }`}
          title={unlabeledCount > 0 ? `${unlabeledCount} unlabeled` : undefined}
        >
          RECORDINGS
          {unlabeledCount > 0 && (
            <span className="ml-1.5 inline-flex items-center justify-center bg-hud-warning text-hud-bg text-xs rounded-full px-1.5 min-w-[1.25rem] text-center">
              {unlabeledCount}
            </span>
          )}
        </button>
      </div>

      {tab === 'system' ? (
        <div className="flex flex-col gap-0">
          <StatRow label="Pipeline">
            <span className="flex items-center gap-1.5">
              <StatusDot active={pipelineRunning} />
              {pipelineRunning ? 'Running' : 'Stopped'}
            </span>
          </StatRow>
          <StatRow label="Device">
            <span className="flex items-center gap-1.5">
              {deviceStatus.scanning && !deviceStatus.detected ? (
                <span className="inline-block w-2 h-2 rounded-full bg-hud-warning animate-pulse" />
              ) : (
                <StatusDot active={deviceStatus.detected} />
              )}
              {deviceLabel(deviceStatus)}
            </span>
          </StatRow>
          <StatRow label="Overflows">
            {health?.overflow_count ?? 0}
          </StatRow>
          <StatRow label="Last Frame">
            {formatFrameAge(health?.last_frame_time)}
          </StatRow>
        </div>
      ) : (
        <div className="flex flex-col gap-3 h-full overflow-y-auto">
          <RecordingPanel deviceDetected={deviceStatus.detected} />
          <RecordingsList />
        </div>
      )}
    </Panel>
  )
}
