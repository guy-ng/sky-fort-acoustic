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

export function DashboardLayout() {
  const heatmapRef = useRef<HeatmapCanvasHandle>(null)

  const onFrame = useCallback((buffer: ArrayBuffer) => {
    heatmapRef.current?.renderFrame(buffer)
  }, [])

  const { connected: heatmapConnected, gridInfo } = useHeatmapSocket({ onFrame })
  const { targets, connected: targetsConnected } = useTargetSocket()
  const { data: health } = useHealth()

  const pipelineRunning = health?.pipeline_running ?? false
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
        gridTemplateColumns: '1fr 1fr 300px',
        gridTemplateRows: 'auto 1fr minmax(120px, 20vh)',
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
              <HeatmapCanvas ref={heatmapRef} gridInfo={gridInfo} />
              <TargetOverlay targets={targets} gridInfo={gridInfo} />
            </div>
            <ColorScale />
          </div>
        </Panel>
      </div>

      {/* Sidebar */}
      <div style={{ gridArea: 'sidebar' }}>
        <Sidebar health={health} />
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
