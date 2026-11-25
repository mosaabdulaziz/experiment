import { useEffect, useState } from 'react'
import { streamsApi } from '@/api/streams'
import { useStreamsStore } from '@/stores/streamsStore'
import type { PipelineDesired } from '@/stores/streamsStore'
import type { StreamInfo } from '@/types/camera'

export type StreamsMode = 'loading' | 'deepstream' | 'legacy'

/**
 * Polls `/api/cameras` and keeps the streams store in sync. Returns the backend
 * mode so the dashboard can pick the multi-stream grid vs. the legacy single feed.
 * Reconciles the expanded overlay: if its stream disappears, the overlay closes.
 */
export function useStreams(intervalMs = 2000): StreamsMode {
  const [mode, setMode] = useState<StreamsMode>('loading')

  useEffect(() => {
    let cancelled = false

    const tick = async () => {
      try {
        const res = await streamsApi.list()
        if (cancelled) return
        if ('current' in res) {
          setMode('legacy')
          return
        }
        setMode('deepstream')
        const streams = res.cameras as StreamInfo[]
        const store = useStreamsStore.getState()
        store.setStreams(streams)
        if (store.expandedId && !streams.some(s => s.id === store.expandedId)) {
          store.setExpanded(null)
        }
        const p = res.pipeline
        if (p === 'playing' || p === 'paused') {
          store.setPipelineDesired(p as PipelineDesired)
        } else if (p === undefined) {
          store.setPipelineDesired('unknown')
        }
        store.setZonesStatus(res.zones
          ? { enabled: res.zones.enabled, version: res.zones.version, appliedVersion: res.zones.applied_version,
              appliedByCamera: res.zones.applied_by_camera,
              proximity: res.zones.proximity
                ? { version: res.zones.proximity.version, appliedVersion: res.zones.proximity.applied_version }
                : null }
          : null)
      } catch {
        /* sidecar/API blip — keep last known state and retry next tick */
      }
    }

    tick()
    const id = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs])

  return mode
}
