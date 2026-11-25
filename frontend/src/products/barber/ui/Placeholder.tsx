import type { PageProps } from '../App'
import { useLang } from '../i18n'

/** Stand-in for a page that a later task rebuilds; it takes the page props
 * the real page will use and ignores them. */
export function Placeholder({ name }: { name: string } & Partial<PageProps>) {
  const { t } = useLang()
  return (
    <section className="card placeholder">
      <h1>{name}</h1>
      <p>{t('common.comingSoon')}</p>
    </section>
  )
}
