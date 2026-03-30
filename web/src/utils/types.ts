export interface TargetState {
  id: string
  class_label: string
  speed_mps: number | null
  az_deg: number
  el_deg: number
  confidence: number
}

export interface HeatmapHandshake {
  type: 'handshake'
  width: number
  height: number
  az_min: number
  az_max: number
  el_min: number
  el_max: number
}

export interface HealthStatus {
  status: string
  device_detected: boolean
  pipeline_running: boolean
  overflow_count: number
  last_frame_time: number | null
}

export interface BeamformingMapResponse {
  az_min: number
  az_max: number
  el_min: number
  el_max: number
  az_resolution: number
  el_resolution: number
  width: number
  height: number
  data: number[][]
  peak: { az_deg: number; el_deg: number; power: number } | null
}
