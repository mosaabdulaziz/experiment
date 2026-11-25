import { useEffect, useState } from 'react'
import { chatApi } from '@/api/chat'
import { configApi } from '@/api/config'
import { useChatStore } from '@/stores/chatStore'
import { useConfigStore } from '@/stores/configStore'
import { IconChevronDown, IconLines } from '@/components/ui/icons'
import type { Language } from '@/types/config'
import styles from './SystemPromptPanel.module.css'

const COLLAPSE_KEY = 'prompt-panel-collapsed'

type SaveState = 'idle' | 'saving' | 'saved' | 'error'

export function SystemPromptPanel() {
  const personas = useConfigStore(s => s.personas)
  const activePersona = useConfigStore(s => s.persona)
  const language = useConfigStore(s => s.language)
  const prompt = useConfigStore(s => s.prompt)

  const setPersonas = useConfigStore(s => s.setPersonas)
  const setPersona = useConfigStore(s => s.setPersona)
  const setLanguageStore = useConfigStore(s => s.setLanguage)
  const setPrompt = useConfigStore(s => s.setPrompt)

  const [collapsed, setCollapsed] = useState(() => {
    return localStorage.getItem(COLLAPSE_KEY) !== 'false'
  })
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [busy, setBusy] = useState(false)

  // Initial load
  useEffect(() => {
    Promise.all([
      configApi.getPersonas().catch(() => null),
      configApi.getPrompt().catch(() => null),
      configApi.getLanguage().catch(() => null),
    ]).then(([personasRes, promptRes, langRes]) => {
      if (personasRes) setPersonas(personasRes.personas, personasRes.active)
      if (promptRes) setPrompt(promptRes.instructions ?? '')
      if (langRes) setLanguageStore(langRes.language)
    })
  }, [setPersonas, setPrompt, setLanguageStore])

  const toggle = () => {
    setCollapsed(c => {
      const next = !c
      localStorage.setItem(COLLAPSE_KEY, String(next))
      return next
    })
  }

  const handlePersonaChange = async (id: string) => {
    setBusy(true)
    try {
      const data = await configApi.setPersona(id)
      setPersona(id)
      if (data.instructions !== undefined) setPrompt(data.instructions)
    } catch (err) {
      console.error('Persona switch failed:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleLanguageChange = async (lang: Language) => {
    try {
      await configApi.setLanguage(lang)
      setLanguageStore(lang)
    } catch (err) {
      console.error('Language change failed:', err)
    }
    // Clear chat session on language change
    const sid = useChatStore.getState().sessionId
    if (sid) await chatApi.deleteSession(sid).catch(() => null)
    useChatStore.getState().reset()
  }

  const handleSave = async () => {
    setSaveState('saving')
    try {
      await configApi.updatePrompt(prompt)
      setSaveState('saved')
    } catch {
      setSaveState('error')
    } finally {
      setTimeout(() => setSaveState('idle'), 2000)
    }
  }

  const saveLabel =
    saveState === 'saving' ? 'Saving…' : saveState === 'saved' ? 'Saved!' : saveState === 'error' ? 'Error' : 'Save'

  return (
    <section className={`${styles.panel} ${collapsed ? styles.collapsed : ''}`}>
      <div className={styles.title} onClick={toggle}>
        <div className={styles.titleLeft}>
          <button
            type="button"
            className={styles.collapseBtn}
            onClick={e => {
              e.stopPropagation()
              toggle()
            }}
            title="Expand / collapse"
            aria-label="Expand or collapse"
          >
            <IconChevronDown />
          </button>
          <IconLines opacity={0.6} />
          <span className={styles.label}>Monitoring Instructions</span>
        </div>
        <div
          className={styles.controls}
          onClick={e => e.stopPropagation()}
        >
          <div className={styles.selectGroup}>
            <span className={styles.selectGroupLabel}>Persona</span>
            <select
              className="iris-select iris-select-accent"
              value={activePersona}
              onChange={e => handlePersonaChange(e.target.value)}
              disabled={busy || personas.length === 0}
              title="Select monitoring persona"
            >
              {personas.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.selectGroup}>
            <span className={styles.selectGroupLabel}>Language</span>
            <select
              className="iris-select iris-select-accent"
              value={language}
              onChange={e => handleLanguageChange(e.target.value as Language)}
              title="Select response language"
            >
              <option value="english">English</option>
              <option value="arabic">العربية</option>
            </select>
          </div>
          <button
            type="button"
            className="iris-btn iris-btn-accent iris-btn-sm"
            onClick={handleSave}
            disabled={saveState === 'saving'}
          >
            {saveLabel}
          </button>
        </div>
      </div>

      <textarea
        className={styles.textarea}
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        placeholder="Describe what the AI should watch for, comment on, and alert about…"
        spellCheck={false}
      />
    </section>
  )
}
