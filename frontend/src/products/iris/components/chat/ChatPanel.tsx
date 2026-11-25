import { useEffect, useRef, useState } from 'react'
import { chatApi } from '@/api/chat'
import { useChatStore } from '@/stores/chatStore'
import { BrandLogo } from '@/components/ui/BrandLogo'
import { IconSend, IconPlay } from '@/components/ui/icons'
import { formatTime } from '@/lib/format'
import type { ClipInfo } from '@/types/chat'
import { ClipPreviewModal } from './ClipPreviewModal'
import styles from './ChatPanel.module.css'

const SUGGESTIONS = [
  { label: "What's in frame?", q: 'What do you see in this frame?' },
  { label: 'Anything suspicious?', q: 'Is there anything unusual or suspicious?' },
  { label: 'Count people', q: 'How many people are visible?' },
  { label: 'Describe scene', q: 'Describe the lighting and environment.' },
]

export function ChatPanel() {
  const messages = useChatStore(s => s.messages)
  const typing = useChatStore(s => s.typing)
  const append = useChatStore(s => s.appendMessage)
  const setTyping = useChatStore(s => s.setTyping)
  const setSessionId = useChatStore(s => s.setSessionId)
  const searchMode = useChatStore(s => s.searchMode)
  const setSearchMode = useChatStore(s => s.setSearchMode)

  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [searchStatus, setSearchStatus] = useState<string | null>(null)
  const [previewClip, setPreviewClip] = useState<ClipInfo | null>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll on new message
  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length, typing])

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }, [text])

  const send = async (raw: string) => {
    const message = raw.trim()
    if (!message || sending) return
    setText('')
    setSending(true)
    setTyping(true)
    append('user', message)
    append('ai', '', undefined, true) // empty streaming bubble; grows as tokens arrive
    let firstToken = true

    setSearchStatus(null)

    try {
      await chatApi.stream(
        message,
        useChatStore.getState().sessionId,
        {
          onStatus: (stage, detail) => {
            if (stage === 'searching_history') {
              setTyping(false)
              setSearchStatus(detail ? `Searching history… ${detail}` : 'Searching history…')
            } else if (stage === 'querying_events') {
              setTyping(true)
            }
          },
          onToken: delta => {
            if (firstToken) {
              setTyping(false)
              setSearchStatus(null)
              firstToken = false
            }
            useChatStore.getState().appendToLast(delta)
          },
          onDone: (sid, cameras, clip) => {
            setSessionId(sid)
            useChatStore.getState().finalizeLast(cameras, clip, false)
          },
          onError: msg => {
            useChatStore
              .getState()
              .appendToLast(firstToken ? `Error: ${msg}` : `\n[stream error: ${msg}]`)
            useChatStore.getState().finalizeLast(undefined, null, true)
          },
        },
        searchMode ? 'search' : 'normal',
      )
    } catch (err) {
      console.error('Chat failed:', err)
      useChatStore.getState().appendToLast('Request failed. Please check your connection.')
      useChatStore.getState().finalizeLast(undefined, null, true)
    } finally {
      setTyping(false)
      setSearchStatus(null)
      setSending(false)
      textareaRef.current?.focus()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send(text)
    }
  }

  const showWelcome = messages.length === 0 && !typing

  return (
    <div className={styles.view}>
      <div ref={scrollerRef} className={styles.messages}>
        {showWelcome ? (
          <div className={styles.welcome}>
            <div className={styles.welcomeIcon}>
              <BrandLogo size={44} />
            </div>
            <p className={styles.welcomeTitle}>Ask the camera</p>
            <p className={styles.welcomeBody}>
              Type a question about what&apos;s currently visible in the feed. The AI analyzes the
              live frame and responds.
            </p>
            <div className={styles.suggestions}>
              {SUGGESTIONS.map(s => (
                <button
                  key={s.q}
                  type="button"
                  className={styles.chip}
                  onClick={() => void send(s.q)}
                  disabled={sending}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map(m => (
            <div
              key={m.id}
              className={`${styles.bubble} ${
                m.role === 'user' ? styles.bubbleUser : styles.bubbleAi
              }`}
              dir="auto"
            >
              {m.text}
              {m.role === 'ai' && m.cameras && m.cameras.length > 0 && (
                <div className={styles.camBadges}>
                  {m.cameras.map(c => (
                    <span key={c} className={styles.camBadge}>
                      {c}
                    </span>
                  ))}
                </div>
              )}
              {m.role === 'ai' && m.clip && (
                <button
                  type="button"
                  className={styles.previewBtn}
                  onClick={() => setPreviewClip(m.clip!)}
                  title="Play the matched footage"
                >
                  <IconPlay /> Preview
                </button>
              )}
              <span className={styles.bubbleTime}>{formatTime(m.timestamp)}</span>
            </div>
          ))
        )}

        {searchStatus && (
          <div className={styles.searchStatus} dir="auto">
            <span className={styles.searchSpinner} />
            {searchStatus}
          </div>
        )}

        {typing && (
          <div className={styles.typing}>
            <span className={styles.typingDot} />
            <span className={styles.typingDot} />
            <span className={styles.typingDot} />
          </div>
        )}
      </div>

      <div className={styles.inputArea}>
        <div className={styles.modeToggle} role="group" aria-label="Chat mode">
          <button
            type="button"
            className={`${styles.modeOption} ${!searchMode ? styles.modeActive : ''}`}
            onClick={() => setSearchMode(false)}
            disabled={sending}
            title="Answer from the current frames"
          >
            Normal
          </button>
          <button
            type="button"
            className={`${styles.modeOption} ${searchMode ? styles.modeActive : ''}`}
            onClick={() => setSearchMode(true)}
            disabled={sending}
            title="Search the recorded history"
          >
            Search
          </button>
        </div>
        <textarea
          ref={textareaRef}
          className={styles.input}
          placeholder={searchMode ? 'Search the recorded history…' : 'Ask about the current frame…'}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          spellCheck={false}
          disabled={sending}
        />
        <button
          type="button"
          className={styles.sendBtn}
          onClick={() => void send(text)}
          disabled={sending || !text.trim()}
          title="Send (Enter)"
        >
          <IconSend />
        </button>
      </div>

      <ClipPreviewModal clip={previewClip} onClose={() => setPreviewClip(null)} />
    </div>
  )
}
