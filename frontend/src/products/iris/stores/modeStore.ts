import { create } from 'zustand'
import type { Mode } from '@/types/config'

interface ModeState {
  mode: Mode
  setMode: (mode: Mode) => void
}

export const useModeStore = create<ModeState>(set => ({
  mode: 'proactive',
  setMode: mode => set({ mode }),
}))
