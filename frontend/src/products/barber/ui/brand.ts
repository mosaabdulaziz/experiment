/** The shop's name on every page (2026-10-18, user): the site config's place, sent with the live snapshot. */
export interface Site { place: string; branch: string | null; logo?: boolean }

/** "BekkyBarber" -> "Bekky Barber"; null without a site. */
export function shopName(site: Site | null | undefined): string | null {
  const p = site?.place?.trim()
  return p ? p.replace(/(?<=[a-z])(?=[A-Z])/g, ' ') : null
}

/** The browser tab: the shop, its branch, IRIS. */
export function tabTitle(site: Site | null | undefined): string {
  const name = shopName(site)
  return name ? [name, site?.branch, 'IRIS'].filter(Boolean).join(' · ') : 'IRIS · Barber Ops'
}
