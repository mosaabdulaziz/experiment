import { useEffect, useState } from 'react'
import { cameraApi } from '@/api/camera'

export function useStreamTime(enabled: boolean, intervalMs = 1000) {
  const [offset, setOffset] = useState<number | null>(null)

  useEffect(() => {
    if (!enabled) {
      setOffset(null)
      return
    }
    let cancelled = false
    const tick = async () => {
      try {
        const d = await cameraApi.time()
        if (cancelled) return
        setOffset(d.is_file ? d.offset : null)
      } catch {
        /* non-fatal */
      }
    }
    tick()
    const id = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [enabled, intervalMs])

  return offset
}
