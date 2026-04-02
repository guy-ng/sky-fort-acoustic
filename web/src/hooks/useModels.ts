import { useQuery } from '@tanstack/react-query'
import type { ModelListResponse } from '../utils/types'

export function useModels() {
  return useQuery<ModelListResponse>({
    queryKey: ['models'],
    queryFn: () => fetch('/api/models').then(r => r.json()),
    refetchInterval: 30000,
  })
}
