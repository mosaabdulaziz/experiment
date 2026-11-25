/** The console's pages, addressed inside the product's hash prefix (#/safety/<page>?<query>). Cameras are configured
 *  from their tiles on Live (the zone editor holds each camera's name, features and zones), so there is no Cameras
 *  page: an old link to it lands on Live. */
export type Page = 'live' | 'incidents' | 'health' | 'settings'
export const PAGES: Page[] = ['live', 'incidents', 'health', 'settings']

export function parseRoute(path: string): { page: Page; query: Record<string, string> } {
  const [rawPage, rawQuery = ''] = path.split('?')
  const name = rawPage.replace(/^\/+/, '').split('/')[0]
  const page = (PAGES as string[]).includes(name) ? (name as Page) : 'live'
  return { page, query: Object.fromEntries(new URLSearchParams(rawQuery)) }
}
