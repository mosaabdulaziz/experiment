import { create } from 'zustand'
import type { StreamInfo } from '@/types/camera'
import type { ZonesStatus } from '@/types/zone'

export type PipelineDesired = 'playing' | 'paused' | 'unknown'

// Persisted so the operator's "show zone outlines on the wall" choice survives reloads.
const SHOW_ZONES_KEY = 'iris:showZones'
const loadShowZones = (): boolean => {
  try {
    return localStorage.getItem(SHOW_ZONES_KEY) === '1'
  } catch {
    return false
  }
}

interface StreamsState {
  streams: StreamInfo[]
  /** id of the stream shown in the expanded overlay, or null for the grid view. */
  expandedId: string | null
  /** id of the stream whose zone editor is open, or null. */
  zoneEditId: string | null
  /** Desired pipeline state reported by the sidecar control key. */
  pipelineDesired: PipelineDesired
  /** Zone-config apply handshake, or null when the backend doesn't report it (legacy). */
  zonesStatus: ZonesStatus | null
  /** Whether saved zone outlines are drawn over each stream tile on the dashboard. */
  showZones: boolean

  setStreams: (streams: StreamInfo[]) => void
  setExpanded: (id: string | null) => void
  setZoneEdit: (id: string | null) => void
  setPipelineDesired: (d: PipelineDesired) => void
  setZonesStatus: (z: ZonesStatus | null) => void
  setShowZones: (v: boolean) => void
}

export const useStreamsStore = create<StreamsState>(set => ({
  streams: [],
  expandedId: null,
  zoneEditId: null,
  pipelineDesired: 'unknown',
  zonesStatus: null,
  showZones: loadShowZones(),

  setStreams: streams => set({ streams }),
  setExpanded: id => set({ expandedId: id }),
  setZoneEdit: id => set({ zoneEditId: id }),
  setPipelineDesired: d => set({ pipelineDesired: d }),
  setZonesStatus: z => set({ zonesStatus: z }),
  setShowZones: v => {
    try {
      localStorage.setItem(SHOW_ZONES_KEY, v ? '1' : '0')
    } catch {
      /* ignore storage failures (private mode, quota) */
    }
    set({ showZones: v })
  },
}))
