import { useCallback, useEffect, useRef, useState } from 'react'
import type { HeatmapHandshake } from '../utils/types'

interface UseHeatmapSocketOptions {
  onFrame: (buffer: ArrayBuffer) => void
}

interface UseHeatmapSocketResult {
  connected: boolean
  gridInfo: HeatmapHandshake | null
  deviceOk: boolean
}

export function useHeatmapSocket({ onFrame }: UseHeatmapSocketOptions): UseHeatmapSocketResult {
  const [connected, setConnected] = useState(false)
  const [gridInfo, setGridInfo] = useState<HeatmapHandshake | null>(null)
  const [deviceOk, setDeviceOk] = useState(true)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectDelay = useRef(2000)
  const onFrameRef = useRef(onFrame)
  onFrameRef.current = onFrame

  const connect = useCallback(() => {
    // Guard: skip if already open or still connecting
    const state = wsRef.current?.readyState
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return

    // Clean up any lingering socket in CLOSING state
    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current = null
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/heatmap`)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    let handshakeDone = false

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
            return
          }
          if (msg.type === 'device_reconnected') {
            setDeviceOk(true)
            return
          }
          if (!handshakeDone && msg.type === 'handshake') {
            setGridInfo(msg as HeatmapHandshake)
            handshakeDone = true
          }
        } catch {
          // ignore parse errors
        }
        return
      }

      if (event.data instanceof ArrayBuffer) {
        handshakeDone = true
        onFrameRef.current(event.data)
      }
    }

    ws.onclose = () => {
      setConnected(false)
      setGridInfo(null)
      wsRef.current = null
      // Auto-reconnect with exponential backoff up to 10s
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

  return { connected, gridInfo, deviceOk }
}
