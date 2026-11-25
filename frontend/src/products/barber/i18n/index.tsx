import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { en } from './en'
import { ar } from './ar'

export type Lang = 'en' | 'ar'
export type Key = keyof typeof en
const DICTS: Record<Lang, Record<Key, string>> = { en, ar }
export const LOCALE: Record<Lang, string> = { en: 'en-GB', ar: 'ar-EG' }
const STORAGE_KEY = 'barber.lang'

export function translate(lang: Lang, key: Key, vars?: Record<string, string | number>): string {
  const text = DICTS[lang][key] ?? en[key]
  return vars ? text.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : text
}

export function fmtNumber(lang: Lang, n: number, digits = 0): string {
  return new Intl.NumberFormat(LOCALE[lang], { numberingSystem: 'latn', maximumFractionDigits: digits,
    minimumFractionDigits: digits }).format(n)
}

/** "Friday, 25 September 2026" / "الجمعة، 25 سبتمبر 2026" from YYYY-MM-DD. */
export function fmtLongDate(lang: Lang, isoDay: string): string {
  return new Intl.DateTimeFormat(LOCALE[lang], { numberingSystem: 'latn', weekday: 'long', day: 'numeric',
    month: 'long', year: 'numeric' }).format(new Date(`${isoDay.slice(0, 10)}T12:00:00`))
}

/** "25 Sep" from an ISO date or datetime. */
export function fmtShortDate(lang: Lang, iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Intl.DateTimeFormat(LOCALE[lang], { numberingSystem: 'latn', day: 'numeric', month: 'short' })
    .format(new Date(`${iso.slice(0, 10)}T12:00:00`))
}

function readLang(): Lang {
  try { return localStorage.getItem(STORAGE_KEY) === 'ar' ? 'ar' : 'en' } catch { return 'en' }
}

interface LangCtx {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: Key, vars?: Record<string, string | number>) => string
  dir: 'ltr' | 'rtl'
  fmt: { number: (n: number, digits?: number) => string; longDate: (d: string) => string; date: (iso: string | null | undefined) => string }
}
const Ctx = createContext<LangCtx | null>(null)

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(readLang)
  const setLang = useCallback((l: Lang) => {
    setLangState(l)
    try { localStorage.setItem(STORAGE_KEY, l) } catch { /* private mode: not remembered */ }
  }, [])
  const dir = lang === 'ar' ? 'rtl' : 'ltr'
  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = dir
  }, [lang, dir])
  const value = useMemo<LangCtx>(() => ({
    lang, setLang, dir,
    t: (key, vars) => translate(lang, key, vars),
    fmt: { number: (n, d) => fmtNumber(lang, n, d), longDate: d => fmtLongDate(lang, d), date: iso => fmtShortDate(lang, iso) },
  }), [lang, setLang, dir])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useLang(): LangCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('useLang outside LangProvider')
  return v
}
