/** What pressing Apply will change: per camera, the features saved on that no sidecar runs yet (to start) and those a
 *  sidecar runs but are saved off (to stop); overall, the features whose sidecar the controller restarts. A stream
 *  counts once its sidecar has the camera, sending or not: a silent camera is Health's to report, not Apply's. */
import type { CameraFeature, SafetyCamera } from './types'

export interface PendingChanges {
  cameras: { id: string; start: CameraFeature[]; stop: CameraFeature[] }[]
  features: CameraFeature[]
}

export function pendingChanges(cameras: SafetyCamera[]): PendingChanges {
  const out: PendingChanges = { cameras: [], features: [] }
  const touched = new Set<CameraFeature>()
  for (const cam of cameras) {
    const running = new Set(cam.streams.filter(s => s.feature).map(s => s.feature as CameraFeature))
    const start = cam.features.filter(f => !running.has(f))
    const stop = [...running].filter(f => !cam.features.includes(f))
    if (start.length || stop.length) {
      out.cameras.push({ id: cam.id, start, stop })
      for (const f of [...start, ...stop]) touched.add(f)
    }
  }
  out.features = [...touched].sort()
  return out
}
