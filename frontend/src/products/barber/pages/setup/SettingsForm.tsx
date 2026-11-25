/** Setup's settings: the three knobs a shop owner tunes, each with a one-line
 * explanation, and every other policy key under Advanced, by its backend
 * name. One Save sends only the keys that changed. */
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { api, ApiError } from '../../api/client'
import type { Policy } from '../../api/types'
import { useLang, type Key } from '../../i18n'
import { fmtTarget, parseTimer } from '../../ui/format'
import { Num } from '../../ui/Num'

type Kind = 'timer' | 'hour'
const MAIN: { key: string; kind: Kind }[] = [
  { key: 'wait_target_s', kind: 'timer' },
  { key: 'service_target_s', kind: 'timer' },
  { key: 'day_start_hour', kind: 'hour' },
]
const MAIN_KEYS = new Set(MAIN.map(m => m.key))
const ZONE_TYPES = ['entrance_line', 'chair', 'seat', 'waiting', 'washing', 'row', 'door_area', 'face_area'] as const
const ZONE_KEYS = new Set(ZONE_TYPES.map(z => `zone_show_${z}`))
const kindOf = (key: string): Kind | 'raw' => MAIN.find(m => m.key === key)?.kind ?? 'raw'

const toText = (key: string, v: number) => (kindOf(key) === 'timer' ? fmtTarget(v) : String(v))

/** The text of a field as a number, or null when it isn't a valid value. */
function parse(key: string, text: string): number | null {
  const kind = kindOf(key)
  if (kind === 'timer') return parseTimer(text)
  if (text.trim() === '') return null
  const n = Number(text)
  if (!Number.isFinite(n) || n < 0) return null
  if (kind === 'hour') return Number.isInteger(n) && n <= 23 ? n : null
  return n
}

const textsOf = (p: Policy) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, toText(k, v)]))

export function SettingsForm() {
  const { t } = useLang()
  const [policy, setPolicy] = useState<Policy | null>(null)
  const [texts, setTexts] = useState<Record<string, string>>({})
  const [loadFailed, setLoadFailed] = useState(false)
  const [saving, setSaving] = useState(false)
  const [errors, setErrors] = useState<{ key: string | null; detail: string } | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    api.policy().then(p => { setPolicy(p); setTexts(textsOf(p)) }).catch(() => setLoadFailed(true))
  }, [])

  const { changes, invalid } = useMemo(() => {
    const changes: Policy = {}
    const invalid = new Set<string>()
    if (!policy) return { changes, invalid }
    for (const key of Object.keys(policy)) {
      const text = texts[key] ?? ''
      if (text === toText(key, policy[key])) continue          // untouched (even if the stored value is odd)
      const v = parse(key, text)
      if (v === null) invalid.add(key)
      else if (v !== policy[key]) changes[key] = v
    }
    return { changes, invalid }
  }, [policy, texts])

  if (loadFailed) return <section className="card"><p className="errorBar" role="alert">{t('common.loadFailed')}</p></section>
  if (!policy) return <section className="card"><p className="muted">{t('common.loading')}</p></section>

  const changed = Object.keys(changes).length > 0
  const advanced = Object.keys(policy).filter(k => !MAIN_KEYS.has(k) && !ZONE_KEYS.has(k)).sort()
  const setText = (key: string, v: string) => {
    setTexts(s => ({ ...s, [key]: v }))
    setSaved(false)
    setErrors(err => (err?.key === key ? null : err))
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    if (!changed || invalid.size) return
    setSaving(true)
    setErrors(null)
    try {
      const p = await api.savePolicy(changes)
      setPolicy(p)
      setTexts(textsOf(p))
      setSaved(true)
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : String(err)
      const key = /^(\w+) /.exec(detail)?.[1] ?? null
      setErrors({ key: key && key in policy! ? key : null, detail })
    } finally {
      setSaving(false)
    }
  }

  const fieldNote = (key: string) => (
    <>
      {invalid.has(key) && <span className="fieldError" role="alert">{t('se.invalid')}</span>}
      {errors?.key === key && <span className="fieldError" role="alert" dir="ltr">{errors.detail}</span>}
    </>
  )

  return (
    <section className="card settingsForm">
      <h2 className="sectionLabel">{t('se.title')}</h2>
      <form onSubmit={onSubmit} noValidate>
        <div className="settingsGrid">
          {MAIN.filter(m => m.key in policy).map(({ key, kind }) => (
            <label key={key} className="field">
              {t(`se.${key}` as Key)}
              <span className="fieldHint">{t(`se.${key}.help` as Key)}</span>
              {kind === 'timer'
                ? <input dir="ltr" inputMode="numeric" value={texts[key] ?? ''} placeholder={t('se.minutes')}
                         aria-invalid={invalid.has(key)} onChange={e => setText(key, e.target.value)} />
                : <input dir="ltr" type="number" min={0} max={kind === 'hour' ? 23 : undefined} step={1}
                         value={texts[key] ?? ''} aria-invalid={invalid.has(key)} onChange={e => setText(key, e.target.value)} />}
              {kind === 'timer' && <span className="fieldHint">{t('se.minutes')}</span>}
              {fieldNote(key)}
            </label>
          ))}
        </div>
        <fieldset className="zoneDisplay">
          <legend className="sectionLabel">{t('se.zoneDisplay')}</legend>
          <p className="fieldHint">{t('se.zoneDisplay.help')}</p>
          {ZONE_TYPES.filter(z => `zone_show_${z}` in policy).map(z => {
            const key = `zone_show_${z}`
            return (
              <label key={key} className="checkRow">
                <input type="checkbox" checked={texts[key] !== '0'}
                       onChange={e => setText(key, e.target.checked ? '1' : '0')} />
                {t(`zt.${z}` as Key)}
              </label>
            )
          })}
        </fieldset>
        <details className="advanced">
          <summary>{t('se.advanced')}</summary>
          <div className="advGrid">
            {advanced.map(key => (
              <label key={key} className="field">
                <Num className="rawKey">{key}</Num>
                <input dir="ltr" type="number" min={0} step="any" value={texts[key] ?? ''}
                       aria-invalid={invalid.has(key)} onChange={e => setText(key, e.target.value)} />
                {fieldNote(key)}
              </label>
            ))}
          </div>
        </details>
        <div className="zoneSave">
          <button type="submit" className="btnPrimary" disabled={!changed || invalid.size > 0 || saving}>{t('common.save')}</button>
          {errors && errors.key === null && <p className="fieldError" role="alert" dir="ltr">{errors.detail}</p>}
          <p className="saveNote" role="status">{saved && !changed ? t('se.saved') : ''}</p>
        </div>
      </form>
    </section>
  )
}
