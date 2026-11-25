import { useEffect, type ReactNode } from 'react'
import { logoUrl } from '../api/client'
import type { Route } from '../App'
import type { Key } from '../i18n'
import { useLang } from '../i18n'
import { BrandLogo } from './BrandLogo'
import { shopName, tabTitle, type Site } from './brand'
import { Icon, ICONS } from './ui'
import { productHref } from '../../../shared/router'

const NAV: { path: string; label: Key; icon: string; stroke: boolean }[] = [
  { path: '/', label: 'nav.live', icon: ICONS.live, stroke: false },
  { path: '/insights', label: 'nav.insights', icon: ICONS.insights, stroke: true },
  { path: '/reports', label: 'nav.reports', icon: ICONS.reports, stroke: true },
  { path: '/customers', label: 'nav.customers', icon: ICONS.customers, stroke: true },
  { path: '/alerts', label: 'nav.alerts', icon: ICONS.alerts, stroke: true },
  { path: '/staff', label: 'nav.staff', icon: ICONS.staff, stroke: true },
  { path: '/setup', label: 'nav.setup', icon: ICONS.zones, stroke: true },
]

/** The rail (brand, seven pages, open-alert badge, the shop, language switch) and the page. 2026-10-18 (user): IRIS and
 * the shop on every page without taking any of it -- IRIS under its logo, the shop's logo up the rail (white; its
 * name in type when the site has none), and both in the browser tab. */
export function Shell({ route, openAlerts, site, children }: {
  route: Route; openAlerts: number; site?: Site | null; children: ReactNode
}) {
  const { t, fmt, lang, setLang } = useLang()
  const shop = shopName(site)
  const title = tabTitle(site)
  useEffect(() => { document.title = title }, [title])
  return (
    <div className="shell">
      <nav className="rail" aria-label={t('nav.label')}>
        <span className="railLogo" title="IRIS · Barber Ops">
          <BrandLogo size={32} />
          <span className="railBrandName">IRIS</span>
        </span>
        {NAV.map(item => {
          const label = t(item.label)
          const badge = item.path === '/alerts' && openAlerts > 0
          return (
            <a
              key={item.path}
              className={route.path === item.path ? 'railBtn railActive' : 'railBtn'}
              href={productHref(item.path)}
              title={label}
              aria-label={badge ? t('nav.alertsOpen', { label, count: fmt.number(openAlerts) }) : label}
              aria-current={route.path === item.path ? 'page' : undefined}
            >
              <Icon d={item.icon} stroke={item.stroke} />
              {badge && <span className="railCount">{fmt.number(openAlerts)}</span>}
            </a>
          )
        })}
        {shop && (
          <div className={site?.logo ? 'railShop railShopLogo' : 'railShop'} title={[shop, site?.branch].filter(Boolean).join(' — ')}>
            {site?.logo ? <img src={logoUrl} alt={shop} /> : <span className="railShopName">{shop}</span>}
          </div>
        )}
        <button
          type="button"
          className="railLang"
          aria-label={t('lang.switchLabel')}
          title={t('lang.switchLabel')}
          onClick={() => setLang(lang === 'en' ? 'ar' : 'en')}
        >
          {t('lang.switch')}
        </button>
      </nav>
      <main className="page">{children}</main>
    </div>
  )
}
