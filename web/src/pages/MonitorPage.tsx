import { useCallback } from 'react'
import { Panel } from '../components/layout/Panel'
import { SoundSourceLocator } from '../components/heatmap/SoundSourceLocator'
import { TargetOverlay } from '../components/heatmap/TargetOverlay'
import { SpectrumAnalyzer } from '../components/heatmap/SpectrumAnalyzer'
import { PeakReadout } from '../components/heatmap/PeakReadout'
import { BeamformingControls } from '../components/heatmap/BeamformingControls'
import { RawRecordings } from '../components/heatmap/RawRecordings'
import { TargetStrip } from '../components/targets/TargetStrip'
import { useHeatmapSocket } from '../hooks/useHeatmapSocket'
import { useTargetSocket } from '../hooks/useTargetSocket'
import { useDeviceStatus } from '../hooks/useDeviceStatus'
import { useSpectrumSocket } from '../hooks/useSpectrumSocket'
import { useBfPeaksSocket } from '../hooks/useBfPeaksSocket'

const API_BASE = ''

export function MonitorPage() {
  const { gridInfo } = useHeatmapSocket({})
  const { targets } = useTargetSocket()
  const deviceStatus = useDeviceStatus()
  const spectrum = useSpectrumSocket()
  const bfPeaks = useBfPeaksSocket()

  const deviceDetected = deviceStatus.detected
  const recordingState = bfPeaks?.raw_recording ?? { status: 'idle' as const }
  const playbackState = bfPeaks?.playback ?? { status: 'idle' as const }
  const targetRecordingState = bfPeaks?.target_recording ?? { status: 'idle' as const }

  const handleRecordStart = useCallback(async () => {
    await fetch(`${API_BASE}/api/raw-recording/start`, { method: 'POST' })
  }, [])

  const handleRecordStop = useCallback(async () => {
    await fetch(`${API_BASE}/api/raw-recording/stop`, { method: 'POST' })
  }, [])

  const handleTargetRecordStart = useCallback(async () => {
    await fetch(`${API_BASE}/api/target-recording/start`, { method: 'POST' })
  }, [])

  const handleTargetRecordStop = useCallback(async () => {
    await fetch(`${API_BASE}/api/target-recording/stop`, { method: 'POST' })
  }, [])

  return (
    <div
      className="flex-1 p-1 gap-1 min-h-0"
      style={{
        display: 'grid',
        gridTemplateAreas: `"left heatmap"`,
        gridTemplateColumns: 'minmax(220px, 300px) 1fr',
        gridTemplateRows: '1fr',
      }}
    >
      {/* Left column — stacked panels */}
      <div style={{ gridArea: 'left' }} className="min-h-0 flex flex-col gap-1 overflow-y-auto">
        {/* Peak bearing readout */}
        <Panel title="BEARING">
          <PeakReadout data={bfPeaks} />
        </Panel>

        {/* Spectrum analyzer */}
        <Panel title="SPECTRUM" className="min-h-25">
          <SpectrumAnalyzer spectrum={spectrum} />
        </Panel>

        {/* Beamforming controls */}
        <Panel title="CONTROLS">
          <BeamformingControls
            onRecordStart={handleRecordStart}
            onRecordStop={handleRecordStop}
            recordingState={recordingState}
            onTargetRecordStart={handleTargetRecordStart}
            onTargetRecordStop={handleTargetRecordStop}
            targetRecordingState={targetRecordingState}
          />
        </Panel>

        {/* Raw recordings / playback */}
        <Panel title="RECORDINGS">
          <RawRecordings playbackState={playbackState} />
        </Panel>

        {/* Targets */}
        <Panel title="TARGETS" className="flex-1 min-h-20">
          {targets.length > 0 ? (
            <TargetStrip targets={targets} />
          ) : (
            <div className="flex items-center justify-center h-full text-xs text-hud-text-dim/40">
              No targets
            </div>
          )}
        </Panel>
      </div>

      {/* Sound source locator — right, full height */}
      <div style={{ gridArea: 'heatmap' }} className="min-h-0">
        <Panel title="SOUND SOURCE LOCATOR" className="h-full">
          <div className="relative h-full">
            {!deviceDetected && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-hud-panel/80">
                <span className="material-symbols-outlined text-3xl text-hud-text-dim mb-2">mic_off</span>
                <span className="text-hud-text-dim text-sm uppercase tracking-wider">No Audio Device</span>
                <span className="text-hud-text-dim text-xs mt-1">Connect UMA-16v2 to enable beamforming</span>
              </div>
            )}
            <SoundSourceLocator massCenter={bfPeaks?.mass_center ?? null} gridInfo={gridInfo} />
            <TargetOverlay targets={targets} gridInfo={gridInfo} />
          </div>
        </Panel>
      </div>
    </div>
  )
}
