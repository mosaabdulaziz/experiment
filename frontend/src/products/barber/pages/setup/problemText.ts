/** A structured zone problem (floor/areas.ts zoneProblems, or the server's
 * zone_set_problems) as one sentence for people: areas by their name,
 * cameras by their label (Centre / Left / Right). */
import type { ZoneProblem } from '../../api/types'
import type { AreaTable } from '../../floor/areas'
import type { Key } from '../../i18n'
import { cameraLabel } from '../../ui/cameraLabel'

type T = (key: Key, vars?: Record<string, string | number>) => string

export function problemText(p: ZoneProblem, t: T, areas: AreaTable): string {
  const pairName = (pair: string) => areas.byId[pair]?.name ?? pair
  switch (p.code) {
    case 'pair_single_camera':
      return t('pr.pair_single_camera', { pair: pairName(p.pair), camera: cameraLabel(p.camera, t) })
    case 'pair_twice_in_camera':
      return t('pr.pair_twice_in_camera', { pair: pairName(p.pair) })
    case 'pair_owner_count':
      return t('pr.pair_owner_count', { pair: pairName(p.pair), count: p.count })
    case 'pair_kind_mismatch':
      return t('pr.pair_kind_mismatch', { pair: pairName(p.pair) })
    case 'chair_without_seat':
      return t('pr.chair_without_seat', { zone: p.zone, camera: cameraLabel(p.camera, t) })
    case 'door_area_ignored':
      return t('pr.door_area_ignored', { camera: cameraLabel(p.camera, t) })
    case 'face_area_unknown_chair':
      return t('pr.face_area_unknown_chair', { zone: p.zone, camera: cameraLabel(p.camera, t) })
  }
}
