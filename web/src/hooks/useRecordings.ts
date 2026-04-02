import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

export interface Recording {
  id: string
  label: string
  labeled: boolean
  duration_s: number
  recorded_at: string
  sub_label?: string
  distance_m?: number
  altitude_m?: number
  conditions?: string
  notes?: string
  filename?: string
}

export interface LabelBody {
  label: string
  sub_label?: string
  distance_m?: number
  altitude_m?: number
  conditions?: string
  notes?: string
}

export interface UpdateBody {
  sub_label?: string
  distance_m?: number
  altitude_m?: number
  conditions?: string
  notes?: string
}

export function useRecordingsList() {
  return useQuery<Recording[]>({
    queryKey: ['recordings'],
    queryFn: () => fetch('/api/recordings').then(r => r.json()),
    refetchInterval: 10000,
  })
}

export function useStartRecording() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      fetch('/api/recordings/start', { method: 'POST' }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['recordings'] }),
  })
}

export function useStopRecording() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      fetch('/api/recordings/stop', { method: 'POST' }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['recordings'] }),
  })
}

export function useLabelRecording() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: LabelBody }) =>
      fetch(`/api/recordings/${id}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['recordings'] }),
  })
}

export function useUpdateRecording() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateBody }) =>
      fetch(`/api/recordings/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['recordings'] }),
  })
}

export function useDeleteRecording() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      fetch(`/api/recordings/${id}`, { method: 'DELETE' }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['recordings'] }),
  })
}
