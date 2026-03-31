import { useCallback, useRef } from 'react'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { Panel } from './Panel'
import { HeatmapCanvas, type HeatmapCanvasHandle } from '../heatmap/HeatmapCanvas'
import { TargetOverlay } from '../heatmap/TargetOverlay'
import { ColorScale } from '../heatmap/ColorScale'
import { TargetStrip } from '../targets/TargetStrip'
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
  const { targets, connected: targetsConnected } = useTargetSocket()
  const { data: health, isError: healthError } = useHealth()
  const deviceStatus = useDeviceStatus()

  const effectiveHealth = healthError ? undefined : health
  const pipelineRunning = effectiveHealth?.pipeline_running ?? false
  const statusText = heatmapConnected && targetsConnected
    ? 'LIVE'
    : heatmapConnected
      ? 'TARGETS OFFLINE'
      : 'CONNECTING...'

  return (
    <div
      className="h-screen w-screen bg-hud-bg p-1 gap-1"
      style={{
        display: 'grid',
        gridTemplateAreas: `
          "header  header  header"
          "heatmap heatmap sidebar"
          "targets targets sidebar"
        `,
        gridTemplateColumns: '3fr 2fr 300px',
        gridTemplateRows: 'auto 1fr minmax(80px, 12vh)',
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
              {!heatmapConnected && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-hud-panel/80">
                  <div className="w-3 h-3 rounded-full bg-hud-danger mb-2" />
                  <span className="text-hud-text-dim text-sm uppercase tracking-wider">No Signal</span>
                </div>
              )}
              {!deviceStatus.detected && (
                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/50">
                  <div className="w-3 h-3 rounded-full bg-amber-500 mb-2 animate-pulse" />
                  <span className="text-gray-800 text-sm font-semibold uppercase tracking-wider">Device Disconnected</span>
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

      {/* Targets */}
      <div style={{ gridArea: 'targets' }}>
        <Panel title="TARGETS" className="h-full">
          <TargetStrip targets={targets} />
        </Panel>
      </div>
    </div>
  )
}
