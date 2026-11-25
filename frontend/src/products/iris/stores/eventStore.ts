import { create } from 'zustand'
import type { IrisEvent } from '@/types/event'
import type { WsStatus } from '@/types/ws'

const RANK: Record<string, number> = { info: 0, warning: 1, critical: 2 }
const rank = (e: IrisEvent | undefined) => RANK[e?.severity ?? 'warning'] ?? 1

/** Whether an alert flashes the screen: a new warning or critical alert, or an update that raised one to warning or
 *  critical. Unconfirmed (info) alerts never flash. */
export function flashes(prev: IrisEvent | undefined, next: IrisEvent): boolean {
  if (next.event_type !== 'alert' || rank(next) < 1) return false
  return prev === undefined || rank(next) > rank(prev)
}

interface EventState {
  events: IrisEvent[]
  status: WsStatus
  lastAlertId: number | null
  lastAlertIdByCamera: Record<string, number>
  /** Bumped on every flash (a new warning/critical alert, or an incident raised to one): the flash watches this, not
   *  an id, so an incident escalating in place (same id) flashes again. */
  flashSeq: number
  flashSeqByCamera: Record<string, number>
  /** Updates for events the live list does not hold -- the Incidents page also shows older ones it fetched over
   *  REST: kept by id and applied over those (useAlertIncidents), so their cards do not wait for a reload. */
  outside: Record<number, Partial<IrisEvent>>

  addEvent: (e: IrisEvent) => void
  updateEvent: (id: number, patch: Partial<IrisEvent>) => void
  /** An incident's whole record, raised in place (it flashes when it rose to warning or critical). */
  replaceEvent: (e: IrisEvent) => void
  setHistory: (events: IrisEvent[]) => void
  setStatus: (status: WsStatus) => void
  clear: () => void
}

/** The flash counters after an event that does (or does not) flash. */
function bumped(prev: EventState, flash: boolean, camera: string | null) {
  if (!flash) return { flashSeq: prev.flashSeq, flashSeqByCamera: prev.flashSeqByCamera }
  return {
    flashSeq: prev.flashSeq + 1,
    flashSeqByCamera: camera != null
      ? { ...prev.flashSeqByCamera, [camera]: (prev.flashSeqByCamera[camera] ?? 0) + 1 } : prev.flashSeqByCamera,
  }
}

export const useEventStore = create<EventState>(set => ({
  events: [],
  status: 'idle',
  lastAlertId: null,
  lastAlertIdByCamera: {},
  flashSeq: 0,
  flashSeqByCamera: {},
  outside: {},

  addEvent: e =>
    set(prev => {
      const flash = flashes(undefined, e)
      return {
        events: [...prev.events, e],
        lastAlertId: flash ? e.id : prev.lastAlertId,
        lastAlertIdByCamera:
          flash && e.camera != null
            ? { ...prev.lastAlertIdByCamera, [e.camera]: e.id }
            : prev.lastAlertIdByCamera,
        ...bumped(prev, flash, e.camera),
      }
    }),
  updateEvent: (id, patch) =>
    set(prev => {
      const idx = prev.events.findIndex(e => e.id === id)
      // Not in the live list (an older event the Incidents page fetched, or a bootstrap race): kept aside.
      if (idx === -1) return { outside: { ...prev.outside, [id]: { ...prev.outside[id], ...patch } } }
      const events = prev.events.slice()
      events[idx] = { ...events[idx], ...patch }
      return { events }
    }),
  replaceEvent: e =>
    set(prev => {
      const idx = prev.events.findIndex(x => x.id === e.id)
      if (idx === -1) return { outside: { ...prev.outside, [e.id]: { ...prev.outside[e.id], ...e } } }   // kept aside
      const flash = flashes(prev.events[idx], e)
      const events = prev.events.slice()
      events[idx] = { ...events[idx], ...e }
      return {
        events,
        lastAlertId: flash ? e.id : prev.lastAlertId,
        lastAlertIdByCamera: flash && e.camera != null
          ? { ...prev.lastAlertIdByCamera, [e.camera]: e.id } : prev.lastAlertIdByCamera,
        ...bumped(prev, flash, e.camera),
      }
    }),
  setHistory: events => set({ events }),
  setStatus: status => set({ status }),
  clear: () => set({ events: [], lastAlertId: null, lastAlertIdByCamera: {}, outside: {} }),
}))
