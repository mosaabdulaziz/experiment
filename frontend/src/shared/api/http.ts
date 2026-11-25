/** Calling a service: the part every product does the same way.
 *
 * Both products hand-rolled this -- call, check the status, parse JSON, throw
 * something useful. The error message is the part worth sharing: a service
 * that fails explains itself in a `detail` field, and showing that beats
 * showing "HTTP 500" to whoever is looking at the screen.
 *
 * What is NOT here: which base path a product calls. That is the product's
 * own business (/attendance/api, /barber/api), and a shared module that knew
 * about it would need a branch per product -- the thing this layout exists to
 * avoid.
 */
export async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    const detail = (await res.json().catch(() => null))?.detail
    throw new Error(typeof detail === 'string' ? detail : `${path}: HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

/** A JSON request body with the header that goes with it. */
export const jsonBody = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})
