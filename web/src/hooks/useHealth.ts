import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from '../api/client'
import type { HealthStatus } from '../utils/types'

export function useHealth() {
  return useQuery<HealthStatus>({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 5000,
  })
}
