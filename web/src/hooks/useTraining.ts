import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { TrainingStartParams, TrainingProgressResponse } from '../utils/types'

export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation<{ message: string }, Error, TrainingStartParams>({
    mutationFn: (params: TrainingStartParams) =>
      fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training-progress'] }),
  })
}

export function useCancelTraining() {
  const qc = useQueryClient()
  return useMutation<{ message: string }>({
    mutationFn: () =>
      fetch('/api/training/cancel', { method: 'POST' }).then(r => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training-progress'] }),
  })
}

export function useTrainingProgress() {
  return useQuery<TrainingProgressResponse>({
    queryKey: ['training-progress'],
    queryFn: () => fetch('/api/training/progress').then(r => r.json()),
    refetchInterval: 5000,
  })
}
