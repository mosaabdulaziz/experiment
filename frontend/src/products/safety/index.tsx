/** Site safety: this product's entry point (products/safety/README.md). It wears IRIS's look -- IRIS's tokens, base
 *  styles and buttons (its zone editor and header are reused too); safety.css styles the console's own pages with the
 *  same tokens. Cairo only fills in the Arabic letters IRIS's type does not have. */
import '@fontsource/cairo/arabic-400.css'
import '@fontsource/cairo/arabic-600.css'
import '@fontsource/cairo/arabic-700.css'
import '../iris/styles/globals.css'
import '../iris/styles/buttons.css'
import './safety.css'
import { App } from './App'

export default function SafetyProduct() {
  return <App />
}
