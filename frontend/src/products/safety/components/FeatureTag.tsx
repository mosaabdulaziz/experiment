/** A feature as a coloured tag with its name: on the stream tiles (where it also picks whose overlay the tile shows),
 *  the incident cards and the banner. Each feature has its colour (safety.css, --f-<feature>). */
import { useLang } from '../i18n'
import type { Feature } from '../lib/types'

interface Props {
  feature: Feature
  /** A tag that is a button (the tile's view picker): pressed = the view shown. */
  onClick?: () => void
  pressed?: boolean
  /** Its sidecar has the camera but no picture comes from it. */
  stopped?: boolean
}

export function FeatureTag({ feature, onClick, pressed, stopped }: Props) {
  const { t } = useLang()
  const className = `sf-tag sf-tag-${feature}`
  if (!onClick) return <span className={className}>{t(`tag.${feature}`)}</span>
  return (
    <button type="button" className={className} aria-pressed={pressed} data-stopped={stopped || undefined}
      title={t(`feature.${feature}`)} onClick={e => { e.stopPropagation(); onClick() }}>
      {t(`tag.${feature}`)}
    </button>
  )
}
