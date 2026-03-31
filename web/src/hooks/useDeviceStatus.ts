import { useCallback, useEffect, useRef, useState } from 'react'

export interface DeviceStatusState {
  detected: boolean
  name: string | null
  scanning: boolean
}

const INITIAL: DeviceStatusState = { detected: false, name: null, scanning: true }

export function useDeviceStatus(): DeviceStatusState {
  const [status, setStatus] = useState<DeviceStatusState>(INITIAL)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectDelay = useRef(2000)

  const connect = useCallback(() => {
    const state = wsRef.current?.readyState
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return

    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current = null
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/status`)
    wsRef.current = ws

    ws.onopen = () => {
      reconnectDelay.current = 2000
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data) as {
            device_detected: boolean
            device_name: string | null
            scanning: boolean
          }
          setStatus({
            detected: msg.device_detected,
            name: msg.device_name,
            scanning: msg.scanning,
          })
        } catch {
          // ignore
        }
      }
    }

    ws.onclose = () => {
      wsRef.current = null
      reconnectTimer.current = setTimeout(() => {
        connect()
      }, reconnectDelay.current)
      reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, 10000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
      }
    }
  }, [connect])

  return status
}
