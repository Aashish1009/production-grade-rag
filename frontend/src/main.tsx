import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
// KaTeX first, so the app's own rules win on equal specificity.
import 'katex/dist/katex.min.css'
import './styles.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('index.html has no #root element to mount into.')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
