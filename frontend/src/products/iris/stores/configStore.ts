import { create } from 'zustand'
import type { Language, Persona } from '@/types/config'
import { DEFAULT_CLIP_TTL_SECONDS } from '@/lib/clips'

interface ConfigState {
  persona: string
  personas: Persona[]
  language: Language
  prompt: string
  clipTtlSeconds: number
  /** the VLM runs (proactive analysis, chat); off by default, as shipped */
  vlmEnabled: boolean
  /** 'deepstream' (the sidecar's streams) or 'local' (the in-process camera) */
  perceptionBackend: string

  setPersona: (id: string) => void
  setPersonas: (personas: Persona[], active: string) => void
  setLanguage: (language: Language) => void
  setPrompt: (prompt: string) => void
  setClipTtlSeconds: (seconds: number) => void
  setVlmEnabled: (enabled: boolean) => void
  setPerceptionBackend: (backend: string) => void
}

export const useConfigStore = create<ConfigState>(set => ({
  persona: '',
  personas: [],
  language: 'english',
  prompt: '',
  clipTtlSeconds: DEFAULT_CLIP_TTL_SECONDS,
  vlmEnabled: false,
  perceptionBackend: 'deepstream',

  setPersona: persona => set({ persona }),
  setPersonas: (personas, active) => set({ personas, persona: active }),
  setLanguage: language => set({ language }),
  setPrompt: prompt => set({ prompt }),
  setClipTtlSeconds: clipTtlSeconds => set({ clipTtlSeconds }),
  setVlmEnabled: vlmEnabled => set({ vlmEnabled }),
  setPerceptionBackend: perceptionBackend => set({ perceptionBackend }),
}))
