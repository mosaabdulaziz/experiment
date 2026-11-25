/** The rail and the page: the house's shell (barber/ui/Shell.tsx), with two pages instead of
 * seven -- a fall product has two things to look at, and a third would be a page with no
 * data behind it. */
import type { ReactNode } from 'react'
import { BrandLogo } from '../../../shared/ui/BrandLogo'
import { productHref } from '../../../shared/router'
import { Icon, ICONS } from './ui'

const NAV: { path: string; label: string; icon: string; stroke: boolean }[] = [
  { path: '/', label: 'Live', icon: ICONS.live, stroke: false },
  { path: '/review', label: 'Review', icon: ICONS.review, stroke: true },
]

export function Shell({ route, unseen, children }: {
  route: { path: string }; unseen: number; children: ReactNode
}) {
  return (
    <div className="shell">
      <nav className="rail" aria-label="Fall detection">
        <span className="railLogo" title="IRIS · Fall Detection">
          <BrandLogo size={32} />
          <span className="railBrandName">IRIS</span>
        </span>
        {NAV.map(item => {
          const badge = item.path === '/' && unseen > 0
          return (
            <a
              key={item.path}
              className={route.path === item.path ? 'railBtn railActive' : 'railBtn'}
              href={productHref(item.path)}
              title={item.label}
              aria-label={badge ? `${item.label} (${unseen} new)` : item.label}
              aria-current={route.path === item.path ? 'page' : undefined}
            >
              <Icon d={item.icon} stroke={item.stroke} />
              {badge && <span className="railCount">{unseen}</span>}
            </a>
          )
        })}
      </nav>
      <main className="page">{children}</main>
    </div>
  )
}
