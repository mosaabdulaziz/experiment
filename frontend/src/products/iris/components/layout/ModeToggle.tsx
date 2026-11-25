import { useEffect, useState } from 'react'
import { configApi } from '@/api/config'
import { useModeStore } from '@/stores/modeStore'
import type { Mode } from '@/types/config'
import styles from './ModeToggle.module.css'

const MODES: { id: Mode; label: string; title: string }[] = [
  { id: 'proactive', label: 'Proactive', title: 'Run proactive AI analysis' },
  { id: 'chat', label: 'Chat', title: 'Pause analysis — chat only' },
]

export function ModeToggle() {
  const mode = useModeStore(s => s.mode)
  const setMode = useModeStore(s => s.setMode)
  const [busy, setBusy] = useState(false)

  // Load the current mode on mount (in case the WS history hasn't arrived yet).
  useEffect(() => {
    configApi.getMode().then(r => setMode(r.mode)).catch(() => null)
  }, [setMode])

  const choose = async (next: Mode) => {
    if (next === mode || busy) return
    setBusy(true)
    const prev = mode
    setMode(next) // optimistic
    try {
      await configApi.setMode(next)
    } catch {
      setMode(prev) // revert on failure
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={styles.toggle} role="group" aria-label="Working mode">
      {MODES.map(m => (
        <button
          key={m.id}
          type="button"
          className={`${styles.option} ${mode === m.id ? styles.active : ''}`}
          onClick={() => choose(m.id)}
          disabled={busy}
          title={m.title}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
