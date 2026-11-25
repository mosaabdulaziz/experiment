/** What each feature does with its incidents (acknowledgement, Telegram and from which severity), Telegram's own
 *  switch (off by default) and where it posts -- with the chats the bot is in to choose from and a test message -- and
 *  the console's default language (decisions D15, D24, D25). */
import { useEffect, useState } from 'react'
import { useConsole } from '../App'
import { api, type TelegramChat } from '../api'
import { Pictogram } from '../components/Pictogram'
import { useLang } from '../i18n'
import { FEATURES, type Settings, type Severity } from '../lib/types'

const SEVERITIES: Severity[] = ['info', 'warning', 'critical']

export function SettingsPage() {
  const { t } = useLang()
  const { settings, reloadSettings } = useConsole()
  const [draft, setDraft] = useState<Settings | null>(settings)
  useEffect(() => { if (settings && !draft) setDraft(settings) }, [settings, draft])
  const [state, setState] = useState<{ kind: 'idle' | 'saving' | 'saved' } | { kind: 'error'; error: string }>(
    { kind: 'idle' })

  if (!draft) return <div className="sf-page"><p className="sf-empty">{t('common.loading')}</p></div>

  const setFeature = (f: keyof Settings['features'], change: Partial<Settings['features'][typeof f]>) => {
    setDraft({ ...draft, features: { ...draft.features, [f]: { ...draft.features[f], ...change } } })
    setState({ kind: 'idle' })
  }
  const save = () => {
    setState({ kind: 'saving' })
    api.saveSettings({ features: draft.features, telegram: { enabled: draft.telegram.enabled, chat_id: draft.telegram.chat_id },
      language: draft.language })
      .then(saved => { setDraft(saved); setState({ kind: 'saved' }); reloadSettings() })
      .catch(e => setState({ kind: 'error', error: e instanceof Error ? e.message : String(e) }))
  }

  return (
    <div className="sf-page">
      <h2>{t('settings.features')}</h2>
      <div className="sf-panel sf-table-wrap">
        <table className="sf-table">
          <thead>
            <tr>
              <th>{t('settings.feature')}</th><th>{t('settings.ack')}</th><th>{t('settings.notify')}</th>
              <th>{t('settings.minSeverity')}</th>
            </tr>
          </thead>
          <tbody>
            {FEATURES.map(f => {
              const s = draft.features[f]
              return (
                <tr key={f}>
                  <td><span className="sf-feature-cell"><Pictogram feature={f} size={24} />{t(`feature.${f}`)}</span></td>
                  <td>
                    <input type="checkbox" className="sf-switch" checked={s.ack_required}
                      aria-label={`${t(`feature.${f}`)}: ${t('settings.ack')}`}
                      onChange={e => setFeature(f, { ack_required: e.target.checked })} />
                  </td>
                  <td>
                    <input type="checkbox" className="sf-switch" checked={s.notify}
                      aria-label={`${t(`feature.${f}`)}: ${t('settings.notify')}`}
                      onChange={e => setFeature(f, { notify: e.target.checked })} />
                  </td>
                  <td>
                    <select className="iris-select" value={s.min_severity} disabled={!s.notify}
                      aria-label={`${t(`feature.${f}`)}: ${t('settings.minSeverity')}`}
                      onChange={e => setFeature(f, { min_severity: e.target.value as Severity })}>
                      {SEVERITIES.map(v => <option key={v} value={v}>{t(`severity.${v}`)}</option>)}
                    </select>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <h2>{t('settings.telegram')}</h2>
      <TelegramSetup draft={draft} saved={settings}
        setTelegram={change => { setDraft({ ...draft, telegram: { ...draft.telegram, ...change } }); setState({ kind: 'idle' }) }} />

      <h2>{t('settings.language')}</h2>
      <div className="sf-panel sf-form">
        <select className="iris-select" value={draft.language} aria-label={t('settings.language')}
          onChange={e => { setDraft({ ...draft, language: e.target.value as Settings['language'] }); setState({ kind: 'idle' }) }}>
          <option value="en">{t('lang.en')}</option>
          <option value="ar">{t('lang.ar')}</option>
        </select>
      </div>

      <div className="sf-page-head">
        <span role="status">
          {state.kind === 'saved' && <span className="sf-ok">{t('settings.saved')}</span>}
          {state.kind === 'error' && <span className="sf-error">{t('common.saveError', { error: state.error })}</span>}
        </span>
        <button type="button" className="iris-btn iris-btn-accent" disabled={state.kind === 'saving'} onClick={save}>
          {t('settings.save')}
        </button>
      </div>
    </div>
  )
}

const CHAT_TYPE: Record<string, 'settings.chatType.group' | 'settings.chatType.private' | 'settings.chatType.channel'> = {
  group: 'settings.chatType.group', supergroup: 'settings.chatType.group', private: 'settings.chatType.private',
  channel: 'settings.chatType.channel',
}

/** Telegram on or off; the chat incidents go to: typed, or chosen from the chats the bot has seen; and a test message
 *  to the saved one (on or off, to set it up first). The bot token itself stays in the stack's .env (never sent to the
 *  browser). */
function TelegramSetup({ draft, saved, setTelegram }: {
  draft: Settings; saved: Settings | null; setTelegram: (change: Partial<Pick<Settings['telegram'], 'enabled' | 'chat_id'>>) => void
}) {
  const { t } = useLang()
  const tokenSet = draft.telegram.token_set
  const enabled = draft.telegram.enabled
  const setChat = (chat_id: string) => setTelegram({ chat_id })
  const [chats, setChats] = useState<TelegramChat[] | null>(null)
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const savedChat = saved?.telegram.chat_id ?? ''
  const unsaved = draft.telegram.chat_id !== savedChat
  // The server's reason starts "telegram: " (it reads well in its log); the sentence here already says Telegram.
  const reason = (e: unknown) => (e instanceof Error ? e.message : String(e)).replace(/^telegram:\s*/i, '')

  const find = () => {
    setBusy(true)
    setResult(null)
    api.telegramChats()
      .then(setChats)
      .catch(e => setResult({ ok: false, text: t('settings.testFailed', { error: reason(e) }) }))
      .finally(() => setBusy(false))
  }
  const test = () => {
    setBusy(true)
    setResult(null)
    api.telegramTest()
      .then(() => setResult({ ok: true, text: t('settings.testSent') }))
      .catch(e => setResult({ ok: false, text: t('settings.testFailed', { error: reason(e) }) }))
      .finally(() => setBusy(false))
  }

  return (
    <div className="sf-panel sf-form">
      <label className="sf-feature-toggle" data-on={enabled || undefined}>
        <span className="sf-feature-text">
          {t('settings.telegramOn')}
          <span className="sf-feature-sub">{t(enabled ? 'settings.telegramOnHint' : 'settings.telegramOffHint')}</span>
        </span>
        <input type="checkbox" className="sf-switch" checked={enabled} aria-label={t('settings.telegramOn')}
          onChange={e => setTelegram({ enabled: e.target.checked })} />
      </label>
      <p className={tokenSet ? 'sf-muted' : 'sf-warn'}>{t(tokenSet ? 'settings.tokenSet' : 'settings.tokenMissing')}</p>
      <label className="sf-field">
        <span>{t('settings.chat')}</span>
        <input className="sf-input" dir="ltr" value={draft.telegram.chat_id} onChange={e => setChat(e.target.value)} />
        <span className="sf-feature-sub">{t('settings.chatHint')}</span>
      </label>
      <div className="sf-field">
        <span className="sf-feature-sub">{t('settings.findHint')}</span>
        <div className="sf-filters">
          <button type="button" className="iris-btn" disabled={!tokenSet || busy} onClick={find}>
            {t('settings.findChats')}
          </button>
        </div>
        {chats && !chats.length && <p className="sf-warn">{t('settings.noChats')}</p>}
        {chats && chats.length > 0 && (
          <>
            <span className="sf-feature-sub">{t('settings.pickChat')}</span>
            <div className="sf-filters">
              {chats.map(c => (
                <button key={c.id} type="button" className="sf-chip sf-chip-text"
                  aria-pressed={draft.telegram.chat_id === c.id} onClick={() => setChat(c.id)}>
                  {c.title} ({CHAT_TYPE[c.type] ? t(CHAT_TYPE[c.type]) : c.type})
                </button>
              ))}
            </div>
          </>
        )}
      </div>
      <div className="sf-filters">
        <button type="button" className="iris-btn" disabled={!tokenSet || !savedChat || unsaved || busy}
          title={unsaved ? t('settings.saveChatFirst') : undefined} onClick={test}>
          {t('settings.test')}
        </button>
        {unsaved && <span className="sf-feature-sub">{t('settings.saveChatFirst')}</span>}
      </div>
      {result && <p role="status" className={result.ok ? 'sf-ok' : 'sf-error'}>{result.text}</p>}
    </div>
  )
}
