import type { BeamformingMapResponse, TargetState, HealthStatus } from '../utils/types'

const BASE = ''  // relative URLs -- works with both Vite proxy and production

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

export async function fetchMap(): Promise<BeamformingMapResponse> {
  const res = await fetch(`${BASE}/api/map`)
  if (!res.ok) throw new Error(`Map fetch failed: ${res.status}`)
  return res.json()
}

export async function fetchTargets(): Promise<TargetState[]> {
  const res = await fetch(`${BASE}/api/targets`)
  if (!res.ok) throw new Error(`Targets fetch failed: ${res.status}`)
  return res.json()
}
