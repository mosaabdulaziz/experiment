import { apiDelete, apiGet } from './client'
import type { IrisEvent } from '@/types/event'

export const eventsApi = {
  list: (limit = 100) => apiGet<IrisEvent[]>(`/api/events?limit=${limit}`),
  clear: () => apiDelete<{ status: string; deleted: number }>('/api/events'),
}
