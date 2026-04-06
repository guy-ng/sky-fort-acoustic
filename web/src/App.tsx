import { useState } from 'react'
import { Header, type PageKey } from './components/layout/Header'
import { MonitorPage } from './pages/MonitorPage'
import { ToolsPage } from './pages/ToolsPage'
import { useHealth } from './hooks/useHealth'
import { useDeviceStatus } from './hooks/useDeviceStatus'

export default function App() {
  const [page, setPage] = useState<PageKey>('monitor')
  const { data: health, isError: healthError } = useHealth()
  const deviceStatus = useDeviceStatus()

  const effectiveHealth = healthError ? undefined : health
  const pipelineRunning = effectiveHealth?.pipeline_running ?? false
  const deviceDetected = deviceStatus.detected
  const statusText = !deviceDetected
    ? 'NO DEVICE'
    : pipelineRunning
      ? 'LIVE'
      : 'CONNECTING...'

  return (
    <div className="h-screen w-screen bg-hud-bg flex flex-col">
      <Header
        status={statusText}
        pipelineRunning={pipelineRunning}
        page={page}
        onPageChange={setPage}
      />
      {page === 'monitor' ? <MonitorPage /> : <ToolsPage />}
    </div>
  )
}
