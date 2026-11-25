/** Barber shop monitoring: this product's entry point. See attendance/index.ts. */
import '@fontsource/cairo/400.css'
import '@fontsource/cairo/600.css'
import '@fontsource/cairo/700.css'
import '@fontsource/cairo/800.css'
import './app.css'
import { App } from './App'
import { LangProvider } from './i18n'

export default function BarberProduct() {
  return (
    <LangProvider>
      <App />
    </LangProvider>
  )
}
