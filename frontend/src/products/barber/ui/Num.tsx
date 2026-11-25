import type { ReactNode } from 'react'
/** Numbers, tickets, times and chair ids stay left-to-right inside Arabic text. */
export function Num({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <bdi dir="ltr" className={`num ${className}`.trim()}>{children}</bdi>
}

/** A translated text with its `{n}` shown as a <Num>: pass t(key, { n: '{n}' }) (a text without `{n}`, e.g. Arabic
 * "زيارة واحدة", renders as is). */
export function NumIn({ text, n }: { text: string; n: ReactNode }) {
  if (!text.includes('{n}')) return <>{text}</>
  const [before, after = ''] = text.split('{n}')
  return <>{before}<Num>{n}</Num>{after}</>
}
