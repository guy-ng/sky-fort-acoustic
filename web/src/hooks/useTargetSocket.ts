import { useCallback, useEffect, useRef, useState } from 'react'
import type { TargetState } from '../utils/types'

interface UseTargetSocketResult {
  targets: TargetState[]
  connected: boolean
}

export function useTargetSocket(): UseTargetSocketResult {
  const [connected, setConnected] = useState(false)
  const [targets, setTargets] = useState<TargetState[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectDelay = useRef(2000)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/targets`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      reconnectDelay.current = 2000
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const parsed = JSON.parse(event.data) as TargetState[]
          setTargets(parsed)
        } catch {
          // ignore parse errors
        }
      }
    }

    ws.onclose = () => {
      setConnected(false)
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

  return { targets, connected }
}
