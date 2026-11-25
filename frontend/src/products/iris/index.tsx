/** The IRIS surveillance UI as a product of this dashboard.
 *
 * Its source is copied from pyai/iris/iris-frontend and left untouched, so the
 * team that owns it can keep working there and the next copy across stays a
 * copy -- that is also why its `@/...` imports are mapped in vite.config.ts
 * instead of being rewritten here.
 *
 * What its own main.tsx used to do (create the query client, wrap in a router)
 * happens here instead: booting the page belongs to the shell now.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { HashRouter } from 'react-router-dom'

import { App } from './App'

import './styles/globals.css'
import './styles/buttons.css'

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
})

export default function Iris() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* basename: this product's slice of the address, so its routes
          live under #/iris/... like every other product's. */}
      <HashRouter basename="/iris">
        <App />
      </HashRouter>
    </QueryClientProvider>
  )
}
