import { useCallback, useEffect, useRef, useState } from 'react'
import type { TargetState } from '../utils/types'

interface UseTargetSocketResult {
  targets: TargetState[]
  connected: boolean
  deviceOk: boolean
  droneProbability: number | null
  detectionState: string | null
}

export function useTargetSocket(): UseTargetSocketResult {
  const [connected, setConnected] = useState(false)
  const [targets, setTargets] = useState<TargetState[]>([])
  const [deviceOk, setDeviceOk] = useState(true)
  const [droneProbability, setDroneProbability] = useState<number | null>(null)
  const [detectionState, setDetectionState] = useState<string | null>(null)
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
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/targets`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      reconnectDelay.current = 2000
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'device_disconnected') {
            setDeviceOk(false)
            setTargets([])
            setDroneProbability(null)
            setDetectionState(null)
            return
          }
          if (msg.type === 'device_reconnected') {
            setDeviceOk(true)
            return
          }
          // Structured target message with probability
          if (msg.targets !== undefined) {
            setTargets(msg.targets as TargetState[])
            setDroneProbability(msg.drone_probability ?? null)
            setDetectionState(msg.detection_state ?? null)
          } else if (Array.isArray(msg)) {
            // Legacy format fallback
            setTargets(msg as TargetState[])
          }
        } catch {
          // ignore parse errors
        }
      }
    }

    ws.onclose = () => {
      setConnected(false)
      setTargets([])
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

  return { targets, connected, deviceOk, droneProbability, detectionState }
}
