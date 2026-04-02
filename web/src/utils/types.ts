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
  device_name: string | null
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

// --- Training types (Phase 12) ---

export interface TrainingStartParams {
  learning_rate?: number
  batch_size?: number
  max_epochs?: number
  patience?: number
  augmentation_enabled?: boolean
  data_root?: string
}

export interface ConfusionMatrixData {
  tp: number
  fp: number
  tn: number
  fn: number
}

export interface TrainingProgressResponse {
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'failed'
  epoch: number
  total_epochs: number
  train_loss: number
  val_loss: number
  val_acc: number
  best_val_loss: number
  best_epoch: number
  confusion_matrix: ConfusionMatrixData
  error: string | null
}

export type TrainingStatus = TrainingProgressResponse['status']

// --- Evaluation types (Phase 12) ---

export interface EvalRunParams {
  model_path?: string
  data_dir?: string
  ensemble_config_path?: string
}

export interface DistributionStats {
  p25: number
  p50: number
  p75: number
  p95: number
}

export interface ClassDistribution {
  p_agg: DistributionStats
  p_max: DistributionStats
  p_mean: DistributionStats
}

export interface FileResult {
  filename: string
  true_label: string
  predicted_label: string
  p_agg: number
  correct: boolean
}

export interface EvalSummary {
  total: number
  correct: number
  incorrect: number
  accuracy: number
  precision: number
  recall: number
  f1: number
  confusion_matrix: ConfusionMatrixData
}

export interface PerModelResult {
  model_type: string
  model_path: string
  weight: number
  accuracy: number
  precision: number
  recall: number
  f1: number
}

export interface EvalResultResponse {
  summary: EvalSummary
  distribution: Record<string, ClassDistribution>
  per_file: FileResult[]
  model_path: string
  data_dir: string
  message: string
  per_model_results?: PerModelResult[] | null
}

// --- Model types (Phase 12) ---

export interface ModelInfo {
  filename: string
  path: string
  size_bytes: number
  modified: string
}

export interface ModelListResponse {
  models: ModelInfo[]
}

// --- WebSocket types (Phase 12) ---

export interface TrainingWsMessage {
  status: TrainingStatus
  epoch?: number
  total_epochs?: number
  train_loss?: number
  val_loss?: number
  val_acc?: number
  confusion_matrix?: ConfusionMatrixData
  error?: string
}

export interface LossDataPoint {
  epoch: number
  train_loss: number
  val_loss: number
}
