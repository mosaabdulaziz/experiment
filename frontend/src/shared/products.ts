/** The products this dashboard can show, and which of them are enabled here.
 *
 * A product is a folder under src/products/. It owns its screens, its links
 * and its stylesheet; the shell owns only what surrounds them. Adding one is a
 * new folder plus a line in PRODUCTS -- no edit to another product's code.
 *
 * Which products a deployment shows is decided at RUN TIME, not at build time:
 * one build serves every customer, and the backend says what they bought
 * (MULTI_PRODUCT_PLAN.md decision 10). Until that endpoint exists, an install
 * shows everything it was built with, which is what a developer wants anyway.
 */
import type { ComponentType } from 'react'

export type Product = {
  id: string
  title: string
  lang: string
  /** Text direction of this product's screens; the shell sets it on <html>. */
  dir: 'rtl' | 'ltr'
  load: () => Promise<{ default: ComponentType }>
}

export const PRODUCTS: Product[] = [
  {
    id: 'attendance',
    title: 'الحضور والانصراف',
    lang: 'ar',
    dir: 'rtl',
    load: () => import('../products/attendance'),
  },
  {
    id: 'iris',
    title: 'IRIS Surveillance',
    lang: 'en',
    dir: 'ltr',
    load: () => import('../products/iris'),
  },
  {
    id: 'barber',
    title: 'Barber Ops',
    lang: 'en',
    dir: 'ltr',
    load: () => import('../products/barber'),
  },
  {
    id: 'fall_detection',
    title: 'Fall Detection',
    lang: 'en',
    dir: 'ltr',
    load: () => import('../products/fall_detection'),
  },
  {
    id: 'safety',
    title: 'Site safety',
    lang: 'en',
    dir: 'ltr',
    load: () => import('../products/safety'),
  },
  {
    id: 'concealment_detection',
    title: 'Concealment Review',
    lang: 'en',
    dir: 'ltr',
    load: () => import('../products/concealment_detection'),
  },
]

/** Product ids this deployment may show.
 *
 * Asks the backend first. A deployment that has not been told (no endpoint, or
 * it fails) shows every product in the build rather than an empty sidebar --
 * failing open matters here because the alternative is a customer seeing
 * nothing at all when one request times out.
 */
export async function enabledProductIds(): Promise<string[]> {
  try {
    const res = await fetch('/api/products', { signal: AbortSignal.timeout(3000) })
    if (res.ok) {
      const body = await res.json()
      const ids = (body?.products ?? []).filter((id: string) => PRODUCTS.some(p => p.id === id))
      if (ids.length) return ids
    }
  } catch {
    /* fall through: show what this build contains */
  }
  return PRODUCTS.map(p => p.id)
}

/** The product to show: ?product=… wins, then the last one used, then the first enabled. */
export function activeProductId(enabled: string[]): string {
  const asked = new URLSearchParams(location.search).get('product')
  if (asked && enabled.includes(asked)) return asked
  try {
    const remembered = localStorage.getItem('iris.product')
    if (remembered && enabled.includes(remembered)) return remembered
  } catch {
    /* private window or blocked storage: fall through */
  }
  return enabled[0]
}
