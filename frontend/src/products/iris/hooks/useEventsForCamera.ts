import { useMemo } from 'react'
import { useEventStore } from '@/stores/eventStore'
import type { IrisEvent } from '@/types/event'

/** Events for one camera, or all events when `cameraId` is null. */
export function useEventsForCamera(cameraId: string | null): IrisEvent[] {
  const events = useEventStore(s => s.events)
  return useMemo(
    () => (cameraId == null ? events : events.filter(e => e.camera === cameraId)),
    [events, cameraId],
  )
}
