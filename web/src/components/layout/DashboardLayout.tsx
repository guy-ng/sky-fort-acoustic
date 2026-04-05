import { useCallback, useRef } from 'react'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { Panel } from './Panel'
import { HeatmapCanvas, type HeatmapCanvasHandle } from '../heatmap/HeatmapCanvas'
import { TargetOverlay } from '../heatmap/TargetOverlay'
import { ColorScale } from '../heatmap/ColorScale'
import { TargetStrip } from '../targets/TargetStrip'
import { PipelinePanel } from '../pipeline/PipelinePanel'
import { useHeatmapSocket } from '../../hooks/useHeatmapSocket'
import { useTargetSocket } from '../../hooks/useTargetSocket'
import { useHealth } from '../../hooks/useHealth'
import { useDeviceStatus } from '../../hooks/useDeviceStatus'

export function DashboardLayout() {
  const heatmapRef = useRef<HeatmapCanvasHandle>(null)

  const onFrame = useCallback((buffer: ArrayBuffer) => {
    heatmapRef.current?.renderFrame(buffer)
  }, [])

  const { connected: heatmapConnected, gridInfo } = useHeatmapSocket({ onFrame })
  const { targets, connected: targetsConnected, droneProbability, detectionState } = useTargetSocket()
  const { data: health, isError: healthError } = useHealth()
  const deviceStatus = useDeviceStatus()

  const effectiveHealth = healthError ? undefined : health
  const pipelineRunning = effectiveHealth?.pipeline_running ?? false
  const deviceDetected = deviceStatus.detected
  const statusText = !heatmapConnected && !targetsConnected
    ? 'CONNECTING...'
    : !deviceDetected
      ? 'NO DEVICE'
      : heatmapConnected && targetsConnected
        ? 'LIVE'
        : 'TARGETS OFFLINE'

  return (
    <div
      className="h-screen w-screen bg-hud-bg p-1 gap-1"
      style={{
        display: 'grid',
        gridTemplateAreas: `
          "header  header  header"
          "targets heatmap sidebar"
          "targets heatmap sidebar"
        `,
        gridTemplateColumns: 'minmax(200px, 280px) 1fr 300px',
        gridTemplateRows: 'auto 1fr 0px',
      }}
    >
      {/* Header */}
      <div style={{ gridArea: 'header' }}>
        <Header status={statusText} pipelineRunning={pipelineRunning} />
      </div>

      {/* Heatmap */}
      <div style={{ gridArea: 'heatmap' }}>
        <Panel title="BEAMFORMING MAP" className="h-full">
          <div className="flex h-full gap-2">
            <div className="relative flex-1">
              {!deviceDetected && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-hud-panel/80">
                  <span className="material-symbols-outlined text-3xl text-hud-text-dim mb-2">mic_off</span>
                  <span className="text-hud-text-dim text-sm uppercase tracking-wider">No Audio Device</span>
                  <span className="text-hud-text-dim text-xs mt-1">Connect UMA-16v2 to enable beamforming</span>
                </div>
              )}
              {deviceDetected && !heatmapConnected && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-hud-panel/80">
                  <div className="w-3 h-3 rounded-full bg-hud-danger mb-2" />
                  <span className="text-hud-text-dim text-sm uppercase tracking-wider">No Signal</span>
                </div>
              )}
              <HeatmapCanvas ref={heatmapRef} gridInfo={gridInfo} />
              <TargetOverlay targets={targets} gridInfo={gridInfo} />
            </div>
            <ColorScale />
          </div>
        </Panel>
      </div>

      {/* Sidebar */}
      <div style={{ gridArea: 'sidebar' }}>
        <Sidebar health={effectiveHealth} deviceStatus={deviceStatus} />
      </div>

      {/* Pipeline + Targets — left column */}
      <div style={{ gridArea: 'targets' }}>
        <Panel title="PIPELINE" className="h-full">
          <div className="flex flex-col h-full gap-2 overflow-y-auto">
            <PipelinePanel />
            {/* Target cards */}
            {targets.length > 0 && (
              <div className="border-t border-hud-border pt-2">
                <span className="text-[10px] text-hud-text-dim uppercase tracking-wider mb-1 block">Targets</span>
                <TargetStrip targets={targets} />
              </div>
            )}
          </div>
        </Panel>
      </div>
    </div>
  )
}
