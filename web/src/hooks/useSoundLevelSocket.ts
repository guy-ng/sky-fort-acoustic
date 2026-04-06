import { useCallback, useEffect, useRef, useState } from 'react'

export interface SoundLevelState {
  level_db: number | null
  connected: boolean
}

const INITIAL: SoundLevelState = { level_db: null, connected: false }

export function useSoundLevelSocket(): SoundLevelState {
  const [state, setState] = useState<SoundLevelState>(INITIAL)
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
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/sound-level`)
    wsRef.current = ws

    ws.onopen = () => {
      reconnectDelay.current = 2000
      setState((prev) => ({ ...prev, connected: true }))
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data) as { level_db: number | null }
          setState({ level_db: msg.level_db, connected: true })
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
