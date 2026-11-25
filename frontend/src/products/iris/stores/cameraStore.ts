import { create } from 'zustand'

interface CameraState {
  running: boolean
  paused: boolean
  source: string
  isFile: boolean
  cameras: { index: number; label: string }[]
  currentCamera: number | null

  setStatus: (s: Partial<Omit<CameraState, 'setStatus' | 'setCameras'>>) => void
  setCameras: (cameras: { index: number; label: string }[], current: number | null) => void
}

export const useCameraStore = create<CameraState>(set => ({
  running: false,
  paused: false,
  source: 'CAM 0',
  isFile: false,
  cameras: [],
  currentCamera: null,

  setStatus: s => set(prev => ({ ...prev, ...s })),
  setCameras: (cameras, current) => set({ cameras, currentCamera: current }),
}))
