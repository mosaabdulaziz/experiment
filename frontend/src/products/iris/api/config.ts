import { apiGet, apiPost } from './client'
import type {
  AppConfigResponse,
  Language,
  LanguageResponse,
  Mode,
  ModeResponse,
  PersonaSwitchResponse,
  PersonasResponse,
  PromptResponse,
} from '@/types/config'

export const configApi = {
  getPrompt: () => apiGet<PromptResponse>('/api/prompt'),
  updatePrompt: (instructions: string) =>
    apiPost<{ status: string }>('/api/prompt', { instructions }),

  getPersonas: () => apiGet<PersonasResponse>('/api/personas'),
  setPersona: (id: string) => apiPost<PersonaSwitchResponse>(`/api/persona/${id}`),

  getLanguage: () => apiGet<LanguageResponse>('/api/language'),
  setLanguage: (language: Language) =>
    apiPost<{ status: string; language: Language }>('/api/language', { language }),

  getMode: () => apiGet<ModeResponse>('/api/mode'),
  setMode: (mode: Mode) => apiPost<{ status: string; mode: Mode }>('/api/mode', { mode }),

  getAppConfig: () => apiGet<AppConfigResponse>('/api/config'),
}
