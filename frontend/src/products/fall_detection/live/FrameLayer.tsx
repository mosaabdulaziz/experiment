import { useEffect, useRef, useState } from 'react'
import { frameUrl } from './frameUrl'

type Props = {
  camera: string
  /** Gap between a frame finishing and the next request, so a slow frame never stacks requests. */
  intervalMs?: number
  /** The house's own box (.feed) -- the layer draws the picture and nothing else. */
  className?: string
}

/** The camera's latest frame, refreshed only after the previous request finished.
 *
 * Requesting on a blind timer makes the browser abort frames in flight once a few tabs share
 * the origin, and the visible result is stutter; waiting on onLoad/onError keeps exactly one
 * request open per layer.
 *
 * This module is the only place that knows the frame endpoint (tab.test.ts pins that).
 */
export function FrameLayer({ camera, intervalMs = 250, className }: Props) {
  const [tick, setTick] = useState(0)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    setTick(0)
    return () => {
      if (timer.current !== null) clearTimeout(timer.current)
    }
  }, [camera])

  const scheduleNext = () => {
    if (timer.current !== null) clearTimeout(timer.current)
    timer.current = setTimeout(() => setTick(n => n + 1), intervalMs)
  }

  return (
    <img
      alt={`camera ${camera}`}
      className={className}
      src={`${frameUrl(camera)}?t=${tick}`}
      onLoad={scheduleNext}
      onError={scheduleNext}
    />
  )
}
