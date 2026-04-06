import { Panel } from '../components/layout/Panel'
import { PipelinePanel } from '../components/pipeline/PipelinePanel'
import { RecordingPanel } from '../components/recording/RecordingPanel'
import { RecordingsList } from '../components/recording/RecordingsList'
import { TrainingPanel } from '../components/training/TrainingPanel'
import { TestPipelinePanel } from '../components/test-pipeline/TestPipelinePanel'
import { useHealth } from '../hooks/useHealth'
import { useDeviceStatus } from '../hooks/useDeviceStatus'
import { useRecordingsList } from '../hooks/useRecordings'
import { usePipelineSocket } from '../hooks/usePipeline'
import type { HealthStatus } from '../utils/types'
import type { DeviceStatusState } from '../hooks/useDeviceStatus'

function DroneStatus({ state, probability }: { state: string | null; probability: number | null }) {
  const isConfirmed = state === 'DRONE_CONFIRMED'
  const isCandidate = state === 'DRONE_CANDIDATE'
  const pct = probability != null ? Math.round(probability * 100) : null

  return (
    <div
      className={`flex items-center gap-4 rounded-xl px-6 py-5 border-2 ${
        isConfirmed
          ? 'bg-red-900/40 border-red-500 animate-pulse'
          : isCandidate
          ? 'bg-yellow-900/30 border-yellow-500'
          : 'bg-hud-bg border-hud-border'
      }`}
    >
      <div
        className={`w-6 h-6 rounded-full shrink-0 ${
          isConfirmed
            ? 'bg-red-500 shadow-[0_0_20px_rgba(239,68,68,0.7)]'
            : isCandidate
            ? 'bg-yellow-500 shadow-[0_0_12px_rgba(234,179,8,0.5)]'
            : 'bg-hud-text-dim/30'
        }`}
      />
      <div className="flex flex-col flex-1 min-w-0">
        <span
          className={`text-xl font-bold uppercase tracking-wider ${
            isConfirmed ? 'text-red-400' : isCandidate ? 'text-yellow-400' : 'text-hud-text-dim'
          }`}
        >
          {isConfirmed ? 'DRONE DETECTED' : isCandidate ? 'CANDIDATE' : 'NO DRONE'}
        </span>
        {pct != null && (
          <span className={`text-sm font-mono ${
            isConfirmed ? 'text-red-400/70' : isCandidate ? 'text-yellow-400/70' : 'text-hud-text-dim/50'
          }`}>
            CNN {pct}%
          </span>
        )}
      </div>
    </div>
  )
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

function SystemPanel({ health, deviceStatus }: { health: HealthStatus | undefined; deviceStatus: DeviceStatusState }) {
  const pipelineRunning = health?.pipeline_running ?? false

  return (
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
  )
}

export function ToolsPage() {
  const { data: health, isError: healthError } = useHealth()
  const deviceStatus = useDeviceStatus()
  const { data: recordings } = useRecordingsList()
  const live = usePipelineSocket()

  const effectiveHealth = healthError ? undefined : health
  const unlabeledCount = recordings?.filter(r => !r.labeled).length ?? 0

  return (
    <div className="flex-1 overflow-y-auto p-2">
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-2 max-w-[1600px] mx-auto">

        {/* Drone status — top priority */}
        <Panel title="DETECTION STATUS" className="xl:col-span-3">
          <DroneStatus state={live.detectionState} probability={live.droneProbability} />
          {!live.running && (
            <p className="text-xs text-hud-text-dim/50 mt-2 text-center">
              Pipeline not running — start a detection session below
            </p>
          )}
        </Panel>

        {/* Pipeline / Detection */}
        <Panel title="DETECTION PIPELINE" className="xl:row-span-2">
          <PipelinePanel />
        </Panel>

        {/* System Health */}
        <Panel title="SYSTEM">
          <SystemPanel health={effectiveHealth} deviceStatus={deviceStatus} />
        </Panel>

        {/* Recordings */}
        <Panel title={unlabeledCount > 0 ? `RECORDINGS (${unlabeledCount} unlabeled)` : 'RECORDINGS'} className="xl:row-span-2">
          <div className="flex flex-col gap-3 h-full overflow-y-auto max-h-[600px]">
            <RecordingPanel deviceDetected={deviceStatus.detected} />
            <RecordingsList />
          </div>
        </Panel>

        {/* Training */}
        <Panel title="TRAINING" className="lg:col-span-1">
          <div className="flex flex-col gap-1 max-h-[600px] overflow-y-auto">
            <TrainingPanel />
          </div>
        </Panel>

        {/* Test Pipeline */}
        <Panel title="TEST PIPELINE">
          <div className="flex flex-col gap-1 max-h-[500px] overflow-y-auto">
            <TestPipelinePanel />
          </div>
        </Panel>
      </div>
    </div>
  )
}
