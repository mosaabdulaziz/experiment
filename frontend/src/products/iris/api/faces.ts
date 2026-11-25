import { apiGet, apiPost, apiPostForm, apiDelete } from './client'

export interface FaceEntry {
  id: string
  name: string
  photo_path: string
  created_at: number
}

export interface FaceCapture {
  id: string
  source_id: string
  track_id: number
  pts: number
  captured_at: number
  assigned_to: string | null
}

export const facesApi = {
  /** List all known registered faces. */
  listKnown: () => apiGet<FaceEntry[]>('/api/faces'),

  /** Register a new face from an uploaded photo. */
  register: async (name: string, file: File) => {
    const form = new FormData()
    form.append('name', name)
    form.append('file', file)
    return apiPostForm<FaceEntry>('/api/faces/register', form)
  },

  /** Delete a known face from the gallery. */
  deleteKnown: (faceId: string) => apiDelete<{ message: string }>(`/api/faces/${faceId}`),

  /** List all CCTV face captures (unknown faces from live footage). */
  listCaptures: () => apiGet<FaceCapture[]>('/api/faces/captures'),

  /** Get the thumbnail image URL for a capture. */
  captureImageUrl: (captureId: string) => `/api/faces/captures/${captureId}/image`,

  /** Assign a capture to a known person. */
  assignCapture: (captureId: string, name: string) =>
    apiPost<{ message: string; face_id: string | null; name: string }>(
      `/api/faces/captures/${captureId}/assign`,
      { name }
    ),

  /** Upload an additional photo for an existing person. */
  addImage: async (faceId: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return apiPostForm<FaceCapture>(`/api/faces/${faceId}/images`, form)
  },

  /** Delete/reject a face capture. */
  deleteCapture: (captureId: string) =>
    apiDelete<{ message: string }>(`/api/faces/captures/${captureId}`),
}
