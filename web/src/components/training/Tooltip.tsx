import { useState, useRef, type ReactNode } from 'react'

interface TooltipProps {
  text: string
  children: ReactNode
}

export function Tooltip({ text, children }: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const timeout = useRef<ReturnType<typeof setTimeout>>()

  function show() {
    timeout.current = setTimeout(() => setVisible(true), 300)
  }

  function hide() {
    clearTimeout(timeout.current)
    setVisible(false)
  }

  return (
    <span className="relative inline-flex" onMouseEnter={show} onMouseLeave={hide}>
      {children}
      {visible && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs normal-case tracking-normal font-normal text-hud-text bg-gray-900 border border-hud-border rounded shadow-lg whitespace-nowrap z-50 pointer-events-none">
          {text}
        </span>
      )}
    </span>
  )
}
