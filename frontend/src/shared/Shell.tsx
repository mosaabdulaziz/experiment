/** What surrounds every product: pick one, set the page's language, mount it.
 *
 * Deliberately thin. The shell must never know what a product's screens are --
 * the day it does, adding a product means editing the shell, which is the
 * coupling this layout exists to remove. It knows three things: which products
 * are enabled, which one is active, and how to load it.
 */
import { useEffect, useState } from 'react'
import { PRODUCTS, activeProductId, enabledProductIds, type Product } from './products'
import { navigate, setProductBase } from './router'

type Loaded = { product: Product; Screen: React.ComponentType }

export function Shell() {
  const [enabled, setEnabled] = useState<string[] | null>(null)
  const [active, setActive] = useState<string | null>(null)
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    enabledProductIds().then(ids => { setEnabled(ids); setActive(activeProductId(ids)) })
  }, [])

  useEffect(() => {
    if (!enabled || !active) return
    const product = PRODUCTS.find(p => p.id === active)
    if (!product) { setError('no product is enabled for this installation'); return }
    // The page's language and direction belong to the product, not the shell:
    // attendance is Arabic right-to-left, barber is English left-to-right.
    document.documentElement.lang = product.lang
    document.documentElement.dir = product.dir
    document.title = product.title
    // Before the product mounts: every link it renders is built from this.
    setProductBase(product.id)
    // An address outside this product's slice -- a bare URL on first load, or a link into another
    // product -- matches none of its routes. The surveillance product mounts its router under
    // basename="/iris" and renders nothing for a path outside it, so `http://host:18000/` was a blank
    // page. Move to the product's home, exactly as switchTo does.
    const here = location.hash.replace(/^#/, '') || '/'
    if (here !== `/${product.id}` && !here.startsWith(`/${product.id}/`)) navigate('/')
    product.load()
      .then(mod => setLoaded({ product, Screen: mod.default }))
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
  }, [enabled, active])

  function switchTo(id: string) {
    // No reload: each product's links carry its own prefix now, so swapping
    // the mounted product is enough. The address moves to that product's home.
    try { localStorage.setItem('iris.product', id) } catch { /* ignore */ }
    setProductBase(id)
    navigate('/')
    setLoaded(null)
    setActive(id)
  }

  if (error) return <div style={{ padding: 24, fontFamily: 'system-ui' }}>{error}</div>
  if (!enabled || !loaded) return null   // brief; each product renders its own loading state

  const { product, Screen } = loaded
  return (
    <>
      {enabled.length > 1 && (
        <nav className="productSwitch" aria-label="products">
          {PRODUCTS.filter(p => enabled.includes(p.id)).map(p => (
            <button
              key={p.id}
              type="button"
              onClick={() => p.id !== product.id && switchTo(p.id)}
              aria-current={p.id === product.id}
              className={p.id === product.id ? 'productSwitchActive' : undefined}
            >
              {p.title}
            </button>
          ))}
        </nav>
      )}
      <Screen />
    </>
  )
}
