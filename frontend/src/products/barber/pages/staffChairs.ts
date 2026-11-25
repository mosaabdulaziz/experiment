/** Staff & chairs' pure bits: the barber choices for a chair's dropdown, and
 * a guard against out-of-order async responses. */
import type { Staff } from '../api/types'

export interface ChairOption { id: string; name: string; disabled: boolean }

/** Every active barber, in staff-list order, plus the chair's current
 * barber if they have since been made inactive -- shown disabled, so the
 * select still has an option to display as selected instead of falling
 * back to nothing. */
export function chairOptions(staff: Staff[], currentId: string | null): ChairOption[] {
  const options = staff.filter(s => s.active).map(s => ({ id: s.id, name: s.name, disabled: false }))
  if (currentId && !options.some(o => o.id === currentId)) {
    const current = staff.find(s => s.id === currentId)
    if (current) options.push({ id: current.id, name: current.name, disabled: true })
  }
  return options
}

/** Reassigning several chairs quickly fires several assign-then-reload
 * requests at once; they can settle in any order, so a slow one resolving
 * last would overwrite the roster with stale data. `start()` hands out a
 * token per request; `isCurrent(token)` is true only for the most recently
 * started one, so a caller can drop every earlier request's result. */
export interface LatestGuard { start: () => number; isCurrent: (token: number) => boolean }

export function createLatestGuard(): LatestGuard {
  let token = 0
  return {
    start: () => ++token,
    isCurrent: t => t === token,
  }
}
