export class ApiError extends Error {
  status: number
  body?: unknown

  constructor(message: string, status: number, body?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

async function handle<T>(res: Response): Promise<T> {
  const ct = res.headers.get('content-type') ?? ''
  const isJson = ct.includes('application/json')
  const body = isJson ? await res.json().catch(() => null) : await res.text().catch(() => null)
  if (!res.ok || !isJson) {
    // A 200 with a non-JSON body means the backend's SPA catch-all route served
    // index.html instead of a real API response — the endpoint isn't implemented
    // server-side, not a genuine success. Surface it as an error either way.
    const msg =
      (body && typeof body === 'object' && 'error' in body && typeof body.error === 'string'
        ? body.error
        : null) ??
      (isJson ? `Request failed: ${res.status}` : `Endpoint did not return JSON (${res.status}) — it may not exist on the backend yet`)
    throw new ApiError(msg, res.status, body)
  }
  return body as T
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: 'GET' })
  return handle<T>(res)
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method: 'POST' }
  if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' }
    init.body = JSON.stringify(body)
  }
  const res = await fetch(path, init)
  return handle<T>(res)
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(path, { method: 'POST', body: form })
  return handle<T>(res)
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: 'DELETE' })
  return handle<T>(res)
}
