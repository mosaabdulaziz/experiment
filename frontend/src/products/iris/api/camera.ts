import { apiGet, apiPost } from './client'
import type {
  CameraActionResponse,
  CameraListResponse,
  CameraStatus,
  CameraTimeResponse,
} from '@/types/camera'

export const cameraApi = {
  status: () => apiGet<CameraStatus>('/api/camera/status'),
  start: () => apiPost<CameraActionResponse>('/api/camera/start'),
  stop: () => apiPost<CameraActionResponse>('/api/camera/stop'),
  pause: () => apiPost<CameraActionResponse>('/api/camera/pause'),
  resume: () => apiPost<CameraActionResponse>('/api/camera/resume'),
  useWebcam: () => apiPost<CameraActionResponse>('/api/camera/use-camera'),
  switchTo: (index: number) => apiPost<CameraActionResponse>(`/api/camera/switch/${index}`),
  list: () => apiGet<CameraListResponse>('/api/cameras'),
  time: () => apiGet<CameraTimeResponse>('/api/camera/time'),
  uploadVideo: (file: File, onProgress?: (percent: number) => void) => {
    const form = new FormData()
    form.append('file', file)
    return new Promise<CameraActionResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/upload-video')
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }
      xhr.onload = () => {
        const ct = xhr.getResponseHeader('content-type') ?? ''
        const body = ct.includes('application/json')
          ? (() => { try { return JSON.parse(xhr.responseText) } catch { return null } })()
          : xhr.responseText
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body as CameraActionResponse)
        } else {
          const msg = body && typeof body === 'object' && 'error' in body && typeof body.error === 'string'
            ? body.error
            : `Upload failed: ${xhr.status}`
          reject(new Error(msg))
        }
      }
      xhr.onerror = () => reject(new Error('Network error during upload'))
      xhr.onabort = () => reject(new Error('Upload cancelled'))
      xhr.send(form)
    })
  },
}
