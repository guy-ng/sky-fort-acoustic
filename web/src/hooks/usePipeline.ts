import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  DetectionLogEntry,
  PipelineStartParams,
  PipelineStatusResponse,
  PipelineWsMessage,
} from '../utils/types'

interface ActivateModelParams {
  model_path: string
}

interface ActivateModelResponse {
  message: string
  model_path: string
  active: boolean
}

export function useActivateModel() {
  return useMutation<ActivateModelResponse, Error, ActivateModelParams>({
    mutationFn: async (params: ActivateModelParams) => {
      const res = await fetch('/api/pipeline/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: `Activation failed (${res.status})` }))
        throw new Error(body.message ?? `Activation failed (${res.status})`)
      }
      return res.json()
    },
  })
}

export function usePipelineStatus() {
  return useQuery<PipelineStatusResponse>({
    queryKey: ['pipeline-status'],
    queryFn: () => fetch('/api/pipeline/status').then(r => r.json()),
    refetchInterval: 5000,
  })
}

export function useStartDetection() {
  const qc = useQueryClient()
  return useMutation<unknown, Error, PipelineStartParams>({
    mutationFn: async (params) => {
      const res = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: `Start failed (${res.status})` }))
        throw new Error(body.message ?? `Start failed (${res.status})`)
      }
      return res.json()
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['pipeline-status'] }),
  })
}

export function useStopDetection() {
  const qc = useQueryClient()
  return useMutation<unknown, Error, void>({
    mutationFn: async () => {
      const res = await fetch('/api/pipeline/stop', { method: 'POST' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ message: `Stop failed (${res.status})` }))
        throw new Error(body.message ?? `Stop failed (${res.status})`)
      }
      return res.json()
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['pipeline-status'] }),
  })
}

export interface PipelineLiveState {
  running: boolean
  detectionState: string | null
  droneProbability: number | null
  log: DetectionLogEntry[]
}

export function usePipelineSocket(): PipelineLiveState {
  const [state, setState] = useState<PipelineLiveState>({
    running: false,
    detectionState: null,
    droneProbability: null,
    log: [],
  })
  const logRef = useRef<DetectionLogEntry[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/pipeline`)

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data) as PipelineWsMessage
      if (!msg.running) {
        logRef.current = []
      }
      if (msg.new_log_entries.length > 0) {
        logRef.current = [...logRef.current, ...msg.new_log_entries].slice(-200)
      }
      setState({
        running: msg.running,
        detectionState: msg.detection_state,
        droneProbability: msg.drone_probability,
        log: logRef.current,
      })
    }

    ws.onclose = () => {
      wsRef.current = null
      setTimeout(connect, 2000)
    }

    ws.onerror = () => ws.close()
    wsRef.current = ws
  }, [])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [connect])

  return state
}
