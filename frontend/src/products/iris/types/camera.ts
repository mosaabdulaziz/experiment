export interface CameraStatus {
  running: boolean
  paused: boolean
  source: string
  is_file: boolean
}

export interface CameraInfo {
  index: number
  label: string
}

/** One stream discovered from the DeepStream sidecar (multi-stream mode). */
export interface StreamInfo {
  id: string
  running: boolean
  is_file: boolean
  source: string
  offset: number | null
}

export interface CameraListResponse {
  cameras: CameraInfo[]
  current: number | string | null
}

export interface CameraActionResponse {
  status: string
  source?: string
  is_file?: boolean
  error?: string
}

export interface CameraTimeResponse {
  offset: number
  is_file: boolean
}
