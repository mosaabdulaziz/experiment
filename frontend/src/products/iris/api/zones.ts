import { apiGet, apiPost } from './client'
import type { ZonesPayload, Zone } from '@/types/zone'

/** Zones sent to the backend on save: everything but the client-only `id` (the backend
 *  assigns canonical positional ids). */
export type ZoneInput = Omit<Zone, 'id'>

export const zonesApi = {
  list: (streamId: string) => apiGet<ZonesPayload>(`/api/streams/${streamId}/zones`),
  /** Bulk-replace a stream's whole zone list in one POST (empty array deletes all). */
  saveAll: (streamId: string, zones: ZoneInput[]) =>
    apiPost<ZonesPayload>(`/api/streams/${streamId}/zones`, { zones }),
  /** Object classes the pipeline can detect — the choices offered in the zone editor. */
  detectionClasses: () => apiGet<{ classes: string[] }>(`/api/detection-classes`),
}
