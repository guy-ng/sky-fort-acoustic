import { useCallback, useEffect, useImperativeHandle, useRef, forwardRef } from 'react'
import { jetColormap } from '../../utils/colormap'
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

    // Pre-build a lookup table for the jet colormap (256 entries)
    const colormapLut = useRef<Uint8Array | null>(null)
    if (!colormapLut.current) {
      const lut = new Uint8Array(256 * 3)
      for (let i = 0; i < 256; i++) {
        const [r, g, b] = jetColormap(i / 255)
        lut[i * 3] = r
        lut[i * 3 + 1] = g
        lut[i * 3 + 2] = b
      }
      colormapLut.current = lut
    }

    // Resize canvas to match grid dimensions
    useEffect(() => {
      if (!gridInfo || !canvasRef.current) return
      const canvas = canvasRef.current
      canvas.width = gridInfo.width
      canvas.height = gridInfo.height
      imageDataRef.current = new ImageData(gridInfo.width, gridInfo.height)
    }, [gridInfo])

    const renderFrame = useCallback((buffer: ArrayBuffer) => {
      if (!gridInfo || !canvasRef.current || !imageDataRef.current || !colormapLut.current) return

      const floats = new Float32Array(buffer)
      const { width, height } = gridInfo
      const expectedLen = width * height

      if (floats.length < expectedLen) return

      // Backend sends pre-normalized [0,1] values (dB + origin suppression + top-dB masking)
      const pixels = imageDataRef.current.data
      const lut = colormapLut.current

      for (let i = 0; i < expectedLen; i++) {
        const v = floats[i]
        // Squared normalization for visual contrast (POC uses alpha**2)
        const normalized = v * v
        const lutIdx = Math.round(normalized * 255) * 3
        const pixIdx = i * 4
        pixels[pixIdx] = lut[lutIdx]
        pixels[pixIdx + 1] = lut[lutIdx + 1]
        pixels[pixIdx + 2] = lut[lutIdx + 2]
        pixels[pixIdx + 3] = 255
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

    // Fit canvas as a square within the container
    useEffect(() => {
      if (!containerRef.current) return
      const observer = new ResizeObserver(() => {
        if (!canvasRef.current || !containerRef.current) return
        const { clientWidth, clientHeight } = containerRef.current
        const size = Math.min(clientWidth, clientHeight)
        const canvas = canvasRef.current
        canvas.style.width = `${size}px`
        canvas.style.height = `${size}px`
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
