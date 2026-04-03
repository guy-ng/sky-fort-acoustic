import { useCallback, useEffect, useRef, useState } from 'react'
import type { TrainingWsMessage, LossDataPoint } from '../utils/types'

const INITIAL: TrainingWsMessage = { status: 'idle' }

export interface TrainingWsState {
  state: TrainingWsMessage
  lossHistory: LossDataPoint[]
}

export function useTrainingSocket(): TrainingWsState {
  const [state, setState] = useState<TrainingWsMessage>(INITIAL)
  const [lossHistory, setLossHistory] = useState<LossDataPoint[]>([])
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
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training`)
    wsRef.current = ws

    ws.onopen = () => {
      reconnectDelay.current = 2000
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data) as TrainingWsMessage
          setState(msg)

          if (
            msg.status === 'running' &&
            msg.epoch !== undefined &&
            msg.train_loss !== undefined &&
            msg.val_loss !== undefined &&
            msg.val_loss > 0  // Only add to history on epoch completion (val_loss=0 means batch-level update)
          ) {
            // Clear history when a new training run starts (epoch 1)
            if (msg.epoch === 1) {
              setLossHistory([
                { epoch: msg.epoch, train_loss: msg.train_loss, val_loss: msg.val_loss },
              ])
            } else {
              setLossHistory(prev => {
                // Avoid duplicate epoch entries
                if (prev.length > 0 && prev[prev.length - 1].epoch === msg.epoch) return prev
                return [
                  ...prev,
                  { epoch: msg.epoch!, train_loss: msg.train_loss!, val_loss: msg.val_loss! },
                ]
              })
            }
          }
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

  return { state, lossHistory }
}
