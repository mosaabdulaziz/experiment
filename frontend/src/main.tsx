import React from 'react'
import ReactDOM from 'react-dom/client'

import './shared/shell.css'
import { Shell } from './shared/Shell'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Shell />
  </React.StrictMode>,
)
