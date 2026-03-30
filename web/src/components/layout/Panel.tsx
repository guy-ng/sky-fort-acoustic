import type { ReactNode } from 'react'

interface PanelProps {
  title?: string
  children: ReactNode
  className?: string
}

export function Panel({ title, children, className = '' }: PanelProps) {
  return (
    <div className={`bg-hud-panel border border-hud-border rounded-lg overflow-hidden ${className}`}>
      {title && (
        <div className="px-3 py-2 border-b border-hud-border">
          <h2 className="text-hud-text-dim text-sm font-medium uppercase tracking-wider">{title}</h2>
        </div>
      )}
      <div className="p-3">{children}</div>
    </div>
  )
}
