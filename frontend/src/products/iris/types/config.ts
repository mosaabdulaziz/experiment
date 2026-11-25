export type Language = 'english' | 'arabic'

export type Mode = 'proactive' | 'chat'

export interface ModeResponse {
  mode: Mode
}

export interface Persona {
  id: string
  name: string
}

export interface PersonasResponse {
  personas: Persona[]
  active: string
}

export interface PersonaSwitchResponse {
  status: string
  id: string
  name: string
  instructions: string
}

export interface PromptResponse {
  instructions: string
}

export interface LanguageResponse {
  language: Language
}

/**
 * Only the fields the UI actually reads. `GET /api/config` returns the whole
 * backend config minus API credentials, so this is deliberately a narrow view of
 * a much larger object.
 */
export interface AppConfigResponse {
  alerts?: {
    clip?: {
      enabled?: boolean
      ttl_seconds?: number
    }
  }
  monitoring?: {
    vlm_enabled?: boolean
  }
  perception?: {
    backend?: string
  }
}
