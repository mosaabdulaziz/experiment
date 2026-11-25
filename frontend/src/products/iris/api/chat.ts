import { apiDelete } from './client'
import type { ChatStreamHandlers } from '@/types/chat'

export const chatApi = {
  async stream(
    message: string,
    sessionId: string | null,
    h: ChatStreamHandlers,
    mode: 'normal' | 'search' = 'normal',
  ): Promise<void> {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, mode }),
    })
    if (res.status === 503) {
      const body = await res.json().catch(() => ({}))
      h.onError(body.error ?? 'No camera frame available.')
      return
    }
    if (!res.ok || !res.body) {
      h.onError(`Request failed: ${res.status}`)
      return
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const frames = buf.split('\n\n')
      buf = frames.pop() ?? ''
      for (const frame of frames) {
        const lines = frame.split('\n')
        const evLine = lines.find(l => l.startsWith('event: '))
        const dataLine = lines.find(l => l.startsWith('data: '))
        if (!evLine || !dataLine) continue
        const ev = evLine.slice(7).trim()
        const data = JSON.parse(dataLine.slice(6))
        if (ev === 'status') h.onStatus?.(data.stage, data.detail)
        else if (ev === 'token') h.onToken(data.delta)
        else if (ev === 'done') h.onDone(data.session_id, data.cameras, data.clip)
        else if (ev === 'error') h.onError(data.message)
      }
    }
  },
  deleteSession: (sessionId: string) =>
    apiDelete<{ status: string }>(`/api/chat/session/${sessionId}`),
}
