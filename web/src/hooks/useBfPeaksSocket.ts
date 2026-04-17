import { useEffect, useRef, useState } from 'react'

export interface BfPeak {
  az_deg: number
  el_deg: number
  power: number
  threshold: number
}

export interface RawRecordingState {
  status: 'idle' | 'recording'
  id?: string
  elapsed_s?: number
  remaining_s?: number
}

export interface PlaybackState {
  status: 'idle' | 'playing'
  path?: string
}

export interface TargetRecordingState {
  status: 'idle' | 'recording'
  id?: string
  elapsed_s?: number
  samples?: number
}

export interface MassCenter {
  az_deg: number
  el_deg: number
  az_min: number
  az_max: number
  el_min: number
  el_max: number
}

export interface BfPeaksData {
  peaks: BfPeak[]
  primary: { az_deg: number; el_deg: number } | null
  mass_center: MassCenter | null
  raw_recording: RawRecordingState
  playback: PlaybackState
  target_recording: TargetRecordingState
}

export function useBfPeaksSocket(): BfPeaksData | null {
  const [data, setData] = useState<BfPeaksData | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    function connect() {
      const state = wsRef.current?.readyState
      if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/bf-peaks`)
      wsRef.current = ws

      ws.onmessage = (event: MessageEvent) => {
        try {
          setData(JSON.parse(event.data))
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

  return data
}
