import { useState } from 'react'
import { eventsApi } from '@/api/events'
import { chatApi } from '@/api/chat'
import { useEventStore } from '@/stores/eventStore'
import { useChatStore } from '@/stores/chatStore'
import { useEventFilters } from '@/hooks/useEventFilters'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { IconChat, IconCircle, IconClose } from '@/components/ui/icons'
import { EventFilters } from './EventFilters'
import { EventLog } from './EventLog'
import styles from './RightPanel.module.css'
import type { Mode } from '@/types/config'

export function RightPanel() {
  // Independent from the top mode switch (modeStore) on purpose — the tab shown
  // here is a view selection, not the backend proactive/chat working mode.
  const [tab, setTab] = useState<Mode>('proactive')
  const events = useEventStore(s => s.events)
  const clearEvents = useEventStore(s => s.clear)
  const { filters, setFilters, filtered } = useEventFilters(events)
  const [busy, setBusy] = useState(false)

  const handleClear = async () => {
    setBusy(true)
    try {
      if (tab === 'chat') {
        const sid = useChatStore.getState().sessionId
        if (sid) await chatApi.deleteSession(sid).catch(() => null)
        useChatStore.getState().reset()
      } else {
        await eventsApi.clear().catch(() => null)
        clearEvents()
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className={styles.panel}>
      <div className={styles.modeHeader}>
        <div className={styles.modePill} role="tablist" aria-label="Right panel mode">
          <button
            type="button"
            className={`${styles.modeBtn} ${tab === 'proactive' ? styles.modeBtnActive : ''}`}
            onClick={() => setTab('proactive')}
            role="tab"
            aria-selected={tab === 'proactive'}
          >
            <IconCircle /> Proactive
          </button>
          <button
            type="button"
            className={`${styles.modeBtn} ${tab === 'chat' ? styles.modeBtnActive : ''}`}
            onClick={() => setTab('chat')}
            role="tab"
            aria-selected={tab === 'chat'}
          >
            <IconChat /> Chat
          </button>
        </div>
        <button
          type="button"
          className={styles.clearBtn}
          onClick={handleClear}
          disabled={busy}
          title={tab === 'chat' ? 'Clear chat & start new session' : 'Clear events'}
        >
          <IconClose /> Clear
        </button>
      </div>

      {tab === 'proactive' ? (
        <>
          {events.length > 0 && (
            <EventFilters
              filters={filters}
              setFilters={setFilters}
              total={events.length}
              visible={filtered.length}
            />
          )}
          <EventLog events={events} filtered={filtered} />
        </>
      ) : (
        <ChatPanel />
      )}
    </section>
  )
}
