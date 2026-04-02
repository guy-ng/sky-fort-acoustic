import { useCallback, useEffect, useRef, useState } from 'react'

export interface RecordingState {
  status: 'idle' | 'recording'
  elapsed_s: number
  remaining_s: number
  level_db: number
}

const INITIAL: RecordingState = { status: 'idle', elapsed_s: 0, remaining_s: 0, level_db: -100 }

export function useRecordingSocket(): RecordingState {
  const [state, setState] = useState<RecordingState>(INITIAL)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectDelay = useRef(2000)

  const connect = useCallback(() => {
    const readyState = wsRef.current?.readyState
    if (readyState === WebSocket.OPEN || readyState === WebSocket.CONNECTING) return

    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current = null
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/recording`)
    wsRef.current = ws

    ws.onopen = () => {
      reconnectDelay.current = 2000
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data) as RecordingState
          setState(msg)
        } catch {
          // ignore parse errors
        }
      }
    }

    ws.onclose = () => {
      wsRef.current = null
      setState(INITIAL)
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

  return state
}
