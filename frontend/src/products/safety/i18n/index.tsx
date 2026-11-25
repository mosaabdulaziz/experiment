/** English and Arabic for the safety console (the barber product's approach: two typed dictionaries, no library).
 *  The language is the operator's choice (remembered in this browser), else the stack's default from Settings. */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { ar } from './ar'
import { en } from './en'

export type Lang = 'en' | 'ar'
export type Key = keyof typeof en
const DICTS: Record<Lang, Record<Key, string>> = { en, ar }
const LOCALE: Record<Lang, string> = { en: 'en-GB', ar: 'ar-EG' }
const STORAGE_KEY = 'safety.lang'

/** A number in this language's digits (Arabic-Indic in Arabic), with at most `maxDigits` after the point. */
export function formatNumber(lang: Lang, n: number, maxDigits = 2): string {
  return new Intl.NumberFormat(LOCALE[lang], { maximumFractionDigits: maxDigits }).format(n)
}

/** "18:20" / "٦:٢٠ م": a time of day in this language, its digits too. */
export function formatTime(lang: Lang, iso: string): string {
  return new Intl.DateTimeFormat(LOCALE[lang], { hour: '2-digit', minute: '2-digit' }).format(new Date(iso))
}

/** The text of `key`, its {values} filled in -- a number in this language's digits, any other value as it is. */
export function translate(lang: Lang, key: Key, vars?: Record<string, string | number>): string {
  const text = DICTS[lang][key] ?? en[key]
  if (!vars) return text
  return text.replace(/\{(\w+)\}/g, (m, k) => {
    if (!(k in vars)) return m
    const v = vars[k]
    return typeof v === 'number' ? formatNumber(lang, v) : String(v)
  })
}

/** "45 s", "3 min", "1 h 20 min" -- in this language's units. */
export function formatDuration(lang: Lang, seconds: number): string {
  if (seconds < 60) return translate(lang, 'unit.seconds', { n: seconds })
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return translate(lang, 'unit.minutes', { n: minutes })
  return translate(lang, 'unit.hours', { h: Math.floor(minutes / 60), m: minutes % 60 })
}

function storedLang(): Lang | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'ar' || v === 'en' ? v : null
  } catch {
    return null
  }
}

interface LangCtx {
  lang: Lang
  /** The operator's choice, remembered in this browser. */
  setLang: (l: Lang) => void
  /** The stack's default (Settings), used until the operator chooses. */
  setDefault: (l: Lang) => void
  t: (key: Key, vars?: Record<string, string | number>) => string
  dir: 'ltr' | 'rtl'
  time: (iso: string) => string
  dateTime: (iso: string) => string
  /** A calendar day ("Friday 6 November"). */
  date: (d: Date) => string
  /** "a, b, c" with this language's comma. */
  list: (items: string[]) => string
  duration: (seconds: number) => string
  /** A number in this language's digits. */
  num: (n: number, maxDigits?: number) => string
}
const Ctx = createContext<LangCtx | null>(null)

export function LangProvider({ children }: { children: ReactNode }) {
  const [chosen, setChosen] = useState<Lang | null>(storedLang)
  const [fallback, setFallback] = useState<Lang>('en')
  const lang = chosen ?? fallback
  const setLang = useCallback((l: Lang) => {
    setChosen(l)
    try { localStorage.setItem(STORAGE_KEY, l) } catch { /* private mode: not remembered */ }
  }, [])
  const dir = lang === 'ar' ? 'rtl' : 'ltr'
  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = dir
  }, [lang, dir])
  const value = useMemo<LangCtx>(() => {
    // Arabic in Arabic-Indic digits (ar-EG's own), times and dates included
    return {
      lang, setLang, setDefault: setFallback, dir,
      t: (key, vars) => translate(lang, key, vars),
      time: iso => formatTime(lang, iso),
      dateTime: iso => new Intl.DateTimeFormat(LOCALE[lang], { day: 'numeric', month: 'short',
        hour: '2-digit', minute: '2-digit' }).format(new Date(iso)),
      date: d => new Intl.DateTimeFormat(LOCALE[lang], { weekday: 'long', day: 'numeric', month: 'long' }).format(d),
      list: items => items.join(lang === 'ar' ? '، ' : ', '),
      duration: s => formatDuration(lang, s),
      num: (n, maxDigits) => formatNumber(lang, n, maxDigits),
    }
  }, [lang, setLang, dir])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useLang(): LangCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('useLang outside LangProvider')
  return v
}
