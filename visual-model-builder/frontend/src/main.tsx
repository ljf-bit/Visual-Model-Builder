import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

function isBenignResizeObserverError(message: unknown): boolean {
  return typeof message === 'string'
    && (
      message.includes('ResizeObserver loop completed with undelivered notifications')
      || message.includes('ResizeObserver loop limit exceeded')
    )
}

window.addEventListener('error', (event) => {
  if (isBenignResizeObserverError(event.message)) {
    event.stopImmediatePropagation()
    event.preventDefault()
  }
}, true)

window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason
  const message = reason instanceof Error ? reason.message : String(reason ?? '')
  if (isBenignResizeObserverError(message)) {
    event.preventDefault()
  }
}, true)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
