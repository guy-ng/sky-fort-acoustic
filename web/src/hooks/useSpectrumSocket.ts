import { useEffect, useRef, useState } from 'react'

export interface SpectrumBand {
  name: string
  fmin: number
  fmax: number
  db: number
}

export interface SpectrumData {
  bands: SpectrumBand[]
  sample_rate: number
}

export function useSpectrumSocket(): SpectrumData | null {
  const [spectrum, setSpectrum] = useState<SpectrumData | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    function connect() {
      const state = wsRef.current?.readyState
      if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/spectrum`)
      wsRef.current = ws

      ws.onmessage = (event: MessageEvent) => {
        try {
          setSpectrum(JSON.parse(event.data))
        } catch { /* ignore */ }
      }

      ws.onclose = () => {
        wsRef.current = null
        reconnectTimer.current = setTimeout(connect, 3000)
      }

      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
      }
    }
  }, [])

  return spectrum
}
