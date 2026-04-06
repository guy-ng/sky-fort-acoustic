import { useCallback, useRef } from 'react'
import { Panel } from '../components/layout/Panel'
import { HeatmapCanvas, type HeatmapCanvasHandle } from '../components/heatmap/HeatmapCanvas'
import { TargetOverlay } from '../components/heatmap/TargetOverlay'
import { ColorScale } from '../components/heatmap/ColorScale'
import { TargetStrip } from '../components/targets/TargetStrip'
import { useHeatmapSocket } from '../hooks/useHeatmapSocket'
import { useTargetSocket } from '../hooks/useTargetSocket'
import { useDeviceStatus } from '../hooks/useDeviceStatus'

export function MonitorPage() {
  const heatmapRef = useRef<HeatmapCanvasHandle>(null)

  const onFrame = useCallback((buffer: ArrayBuffer) => {
    heatmapRef.current?.renderFrame(buffer)
  }, [])

  const { connected: heatmapConnected, gridInfo } = useHeatmapSocket({ onFrame })
  const { targets } = useTargetSocket()
  const deviceStatus = useDeviceStatus()

  const deviceDetected = deviceStatus.detected

  return (
    <div
      className="flex-1 p-1 gap-1 min-h-0"
      style={{
        display: 'grid',
        gridTemplateAreas: `"targets heatmap"`,
        gridTemplateColumns: 'minmax(200px, 280px) 1fr',
        gridTemplateRows: '1fr',
      }}
    >
      {/* Targets — left column */}
      <div style={{ gridArea: 'targets' }} className="min-h-0">
        <Panel title="TARGETS" className="h-full">
          {targets.length > 0 ? (
            <TargetStrip targets={targets} />
          ) : (
            <div className="flex items-center justify-center h-full text-xs text-hud-text-dim/40">
              No targets
            </div>
          )}
        </Panel>
      </div>

      {/* Heatmap — right, full height */}
      <div style={{ gridArea: 'heatmap' }} className="min-h-0">
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
    </div>
  )
}
