import { create } from 'zustand'
import type { ChatMessage, ChatRole, ClipInfo } from '@/types/chat'

interface ChatState {
  messages: ChatMessage[]
  sessionId: string | null
  typing: boolean

  searchMode: boolean

  appendMessage: (role: ChatRole, text: string, cameras?: string[], streaming?: boolean) => void
  appendToLast: (delta: string) => void
  finalizeLast: (cameras?: string[], clip?: ClipInfo | null, failed?: boolean) => void
  setSessionId: (id: string | null) => void
  setTyping: (typing: boolean) => void
  setSearchMode: (v: boolean) => void
  reset: () => void
}

let _counter = 0
const nextId = () => `m${++_counter}-${Date.now()}`

export const useChatStore = create<ChatState>(set => ({
  messages: [],
  sessionId: null,
  typing: false,
  searchMode: false,

  appendMessage: (role, text, cameras, streaming) =>
    set(prev => ({
      messages: [
        ...prev.messages,
        { id: nextId(), role, text, cameras, streaming, timestamp: new Date().toISOString() },
      ],
    })),
  appendToLast: delta =>
    set(prev => {
      const msgs = prev.messages.slice()
      const last = msgs[msgs.length - 1]
      if (last && last.role === 'ai' && last.streaming) {
        msgs[msgs.length - 1] = { ...last, text: last.text + delta }
      }
      return { messages: msgs }
    }),
  finalizeLast: (cameras, clip, failed) =>
    set(prev => {
      const msgs = prev.messages.slice()
      const last = msgs[msgs.length - 1]
      if (last && last.role === 'ai') {
        msgs[msgs.length - 1] = { ...last, streaming: false, cameras, clip: clip ?? undefined, failed }
      }
      return { messages: msgs }
    }),
  setSessionId: id => set({ sessionId: id }),
  setTyping: typing => set({ typing }),
  setSearchMode: v => set({ searchMode: v }),
  reset: () => set({ messages: [], sessionId: null, typing: false }),
}))
