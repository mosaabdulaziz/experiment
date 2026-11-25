import type { Key } from '../i18n'

const NAMES: Record<string, Key> = { 'barber-c': 'cam.centre', 'barber-l': 'cam.left', 'barber-r': 'cam.right' }
const configured = new Map<string, string>()

/** The service's camera names (barber yaml `name`, 2026-10-14); called when the camera list arrives. */
export function rememberCameraNames(cams: { id: string; name?: string | null }[]): void {
  configured.clear()
  for (const c of cams) if (c.name) configured.set(c.id, c.name)
}

/** A camera's name for people: its configured name, else Centre / Left / Right, else its id. */
export function cameraLabel(id: string, t: (key: Key) => string): string {
  const named = configured.get(id)
  if (named) return named
  const key = NAMES[id]
  return key ? t(key) : id
}
