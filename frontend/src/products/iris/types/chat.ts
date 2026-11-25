export type ChatRole = 'user' | 'ai'

export interface ClipInfo {
  id: string
  cameras: string[]
  range?: [number, number] | null
}

export interface ChatMessage {
  id: string
  role: ChatRole
  text: string
  timestamp: string
  cameras?: string[]
  clip?: ClipInfo
  streaming?: boolean
  failed?: boolean
}

export interface ChatRequestBody {
  message: string
  session_id: string | null
  mode: 'normal' | 'search'
}

export interface ChatStreamHandlers {
  onStatus?: (stage: string, detail?: string) => void
  onToken: (delta: string) => void
  onDone: (sessionId: string, cameras?: string[], clip?: ClipInfo | null) => void
  onError: (message: string) => void
}
