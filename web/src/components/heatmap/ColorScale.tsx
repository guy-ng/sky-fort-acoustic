import { useEffect, useRef } from 'react'
import { jetColormap } from '../../utils/colormap'

export function ColorScale() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const height = canvas.height
    const width = canvas.width

    for (let y = 0; y < height; y++) {
      // Top = high (1.0), bottom = low (0.0)
      const t = 1 - y / (height - 1)
      const [r, g, b] = jetColormap(t)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(0, y, width, 1)
    }
  }, [])

  return (
    <div className="flex flex-col items-center gap-1 h-full py-2">
      <span className="text-[10px] text-hud-text-dim uppercase">High</span>
      <canvas
        ref={canvasRef}
        width={16}
        height={200}
        className="flex-1 rounded-sm"
        style={{ imageRendering: 'pixelated', width: '16px' }}
      />
      <span className="text-[10px] text-hud-text-dim uppercase">Low</span>
    </div>
  )
}
