/** Addresses, namespaced per product.
 *
 * Two products cannot both own `#/settings`. While only one product was ever
 * mounted that stayed hidden, at the price of reloading the page to switch.
 * Each product now owns a prefix -- `#/attendance/...`, `#/barber/...` -- so
 * switching is a state change, a deep link survives a refresh, and a link can
 * never land in another product's screen.
 *
 * The base is set once by the shell, before a product is mounted. With no base
 * set every function here behaves exactly as raw `location.hash` did, so a
 * product that has not been converted still works.
 */
let base = ''

export function setProductBase(productId: string): void {
  base = productId ? `/${productId}` : ''
}

export function productBase(): string {
  return base
}

/** The current path as this product sees it: '#/barber/insights' -> '/insights'. */
export function productPath(): string {
  const raw = location.hash.replace(/^#/, '') || '/'
  if (!base) return raw
  if (raw === base) return '/'
  return raw.startsWith(base + '/') ? raw.slice(base.length) : '/'
}

/** An href for a path inside this product: '/insights' -> '#/barber/insights'. */
export function productHref(path: string): string {
  return `#${base}${path.startsWith('/') ? path : `/${path}`}`
}

/** Go to a path inside this product (what `location.hash = '/x'` used to do). */
export function navigate(path: string): void {
  location.hash = `${base}${path.startsWith('/') ? path : `/${path}`}`
}

/** Run `onChange` whenever the address changes; returns the unsubscribe. */
export function onRouteChange(onChange: () => void): () => void {
  window.addEventListener('hashchange', onChange)
  return () => window.removeEventListener('hashchange', onChange)
}
