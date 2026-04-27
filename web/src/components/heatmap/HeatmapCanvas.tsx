import { useCallback, useEffect, useImperativeHandle, useRef, forwardRef } from 'react'
import type { HeatmapHandshake } from '../../utils/types'

export interface HeatmapCanvasHandle {
  renderFrame: (buffer: ArrayBuffer) => void
  canvas: HTMLCanvasElement | null
}

interface HeatmapCanvasProps {
  gridInfo: HeatmapHandshake | null
}

export const HeatmapCanvas = forwardRef<HeatmapCanvasHandle, HeatmapCanvasProps>(
  function HeatmapCanvas({ gridInfo }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const imageDataRef = useRef<ImageData | null>(null)

    // Resize canvas to match grid dimensions
    useEffect(() => {
      if (!gridInfo || !canvasRef.current) return
      const canvas = canvasRef.current
      canvas.width = gridInfo.width
      canvas.height = gridInfo.height
      imageDataRef.current = new ImageData(gridInfo.width, gridInfo.height)
    }, [gridInfo])

    const renderFrame = useCallback((buffer: ArrayBuffer) => {
      if (!gridInfo || !canvasRef.current || !imageDataRef.current) return

      const floats = new Float32Array(buffer)
      const { width, height } = gridInfo
      const expectedLen = width * height

      if (floats.length < expectedLen) return

      const pixels = imageDataRef.current.data

      for (let i = 0; i < expectedLen; i++) {
        const v = floats[i]
        const pixIdx = i * 4
        // Dark background, red where there's sound energy
        // v is [0,1] — 0 = no energy, 1 = peak
        const r = Math.round(v * 255)
        pixels[pixIdx] = r          // red channel = signal strength
        pixels[pixIdx + 1] = 0      // no green
        pixels[pixIdx + 2] = 0      // no blue
        pixels[pixIdx + 3] = 255    // fully opaque
      }

      const ctx = canvasRef.current.getContext('2d')
      if (ctx) {
        ctx.putImageData(imageDataRef.current, 0, 0)
      }
    }, [gridInfo])

    useImperativeHandle(ref, () => ({
      renderFrame,
      get canvas() { return canvasRef.current },
    }), [renderFrame])

    // Fit canvas within the container
    useEffect(() => {
      if (!containerRef.current) return
      const observer = new ResizeObserver(() => {
        if (!canvasRef.current || !containerRef.current) return
        const { clientWidth, clientHeight } = containerRef.current
        const canvas = canvasRef.current
        canvas.style.width = `${clientWidth}px`
        canvas.style.height = `${clientHeight}px`
      })
      observer.observe(containerRef.current)
      return () => observer.disconnect()
    }, [])

    return (
      <div ref={containerRef} className="relative w-full h-full overflow-hidden flex items-center justify-center">
        <canvas
          ref={canvasRef}
          className="block"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
    )
  },
)
