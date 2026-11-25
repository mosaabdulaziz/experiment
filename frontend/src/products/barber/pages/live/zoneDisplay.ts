import type { Zone } from '../../api/types'

/** The zones the camera views draw: a type switched off in Setup (policy zone_show_<type>, every screen) is left
 * out; a type with no switch is drawn. */
export function shownZones(zones: Zone[], display: Record<string, boolean> | undefined): Zone[] {
  return display ? zones.filter(z => display[z.type] !== false) : zones
}
