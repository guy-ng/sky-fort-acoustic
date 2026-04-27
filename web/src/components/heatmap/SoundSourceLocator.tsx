import { useRef, useEffect } from 'react'
import type { MassCenter } from '../../hooks/useBfPeaksSocket'
import type { HeatmapHandshake } from '../../utils/types'

interface SoundSourceLocatorProps {
  massCenter: MassCenter | null
  gridInfo: HeatmapHandshake | null
}

const TRAIL_LENGTH = 15
const GRID_LINES_AZ = 9
const GRID_LINES_EL = 5

export function SoundSourceLocator({ massCenter, gridInfo }: SoundSourceLocatorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const trailRef = useRef<{ x: number; y: number; t: number }[]>([])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return

    const resize = () => {
      const rect = parent.getBoundingClientRect()
      canvas.width = rect.width * devicePixelRatio
      canvas.height = rect.height * devicePixelRatio
      canvas.style.width = `${rect.width}px`
      canvas.style.height = `${rect.height}px`
    }
    const ro = new ResizeObserver(resize)
    ro.observe(parent)
    resize()
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !gridInfo) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    if (w === 0 || h === 0) return

    const dpr = devicePixelRatio
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const cw = w / dpr
    const ch = h / dpr

    // Full-bleed — plot fills entire canvas
    const plotW = cw
    const plotH = ch

    const azMin = gridInfo.az_min
    const azMax = gridInfo.az_max
    const elMin = gridInfo.el_min
    const elMax = gridInfo.el_max

    function azToX(az: number) { return ((az - azMin) / (azMax - azMin)) * plotW }
    function elToY(el: number) { return ((elMax - el) / (elMax - elMin)) * plotH }
    function degToPixW(deg: number) { return (deg / (azMax - azMin)) * plotW }
    function degToPixH(deg: number) { return (deg / (elMax - elMin)) * plotH }

    // Clear
    ctx.clearRect(0, 0, cw, ch)

    // Background
    ctx.fillStyle = '#000a0f'
    ctx.fillRect(0, 0, plotW, plotH)

    // Grid lines with labels inside
    ctx.lineWidth = 1
    for (let i = 0; i <= GRID_LINES_AZ; i++) {
      const az = azMin + (i / GRID_LINES_AZ) * (azMax - azMin)
      const x = azToX(az)
      ctx.strokeStyle = i === GRID_LINES_AZ / 2 ? 'rgba(0, 200, 255, 0.12)' : 'rgba(0, 200, 255, 0.06)'
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, plotH)
      ctx.stroke()
      // Label at bottom inside
      ctx.fillStyle = 'rgba(0, 200, 255, 0.35)'
      ctx.font = '10px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      ctx.fillText(`${Math.round(az)}°`, x, plotH - 3)
    }
    for (let i = 0; i <= GRID_LINES_EL; i++) {
      const el = elMin + (i / GRID_LINES_EL) * (elMax - elMin)
      const y = elToY(el)
      ctx.strokeStyle = 'rgba(0, 200, 255, 0.06)'
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(plotW, y)
      ctx.stroke()
      // Label at left inside
      ctx.fillStyle = 'rgba(0, 200, 255, 0.35)'
      ctx.font = '10px monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.fillText(`${Math.round(el)}°`, 4, y + 2)
    }

    // Border
    ctx.strokeStyle = 'rgba(0, 200, 255, 0.12)'
    ctx.lineWidth = 1
    ctx.strokeRect(0, 0, plotW, plotH)

    // Center marker — red cross at visual center of map
    const centerX = plotW / 2
    const centerY = plotH / 2
    ctx.strokeStyle = 'rgba(255, 60, 60, 0.5)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(centerX - 8, centerY)
    ctx.lineTo(centerX + 8, centerY)
    ctx.moveTo(centerX, centerY - 8)
    ctx.lineTo(centerX, centerY + 8)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(centerX, centerY, 3, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255, 60, 60, 0.6)'
    ctx.fill()

    // Trail — fading dots
    const trail = trailRef.current
    const now = performance.now()
    if (massCenter) {
      trail.push({ x: massCenter.az_deg, y: massCenter.el_deg, t: now })
    }
    while (trail.length > TRAIL_LENGTH) trail.shift()

    for (let i = 0; i < trail.length; i++) {
      const p = trail[i]
      const age = (now - p.t) / 1000
      const alpha = Math.max(0, 0.15 - age * 0.04)
      if (alpha <= 0) continue
      const size = 2 + (i / trail.length) * 3
      ctx.beginPath()
      ctx.arc(azToX(p.x), elToY(p.y), size, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(0, 220, 255, ${alpha})`
      ctx.fill()
    }

    if (massCenter) {
      const cx = azToX(massCenter.az_deg)
      const cy = elToY(massCenter.el_deg)

      // Error area (±1σ)
      const errW = degToPixW(massCenter.az_max - massCenter.az_min)
      const errH = degToPixH(massCenter.el_max - massCenter.el_min)
      const bx = azToX(massCenter.az_min)
      const by = elToY(massCenter.el_max)

      // Radial glow
      const glowR = Math.max(errW, errH, 20)
      const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR)
      gradient.addColorStop(0, 'rgba(0, 220, 255, 0.3)')
      gradient.addColorStop(0.5, 'rgba(0, 180, 255, 0.1)')
      gradient.addColorStop(1, 'rgba(0, 150, 255, 0.0)')
      ctx.fillStyle = gradient
      ctx.fillRect(cx - glowR, cy - glowR, glowR * 2, glowR * 2)

      // Solid mass area
      ctx.fillStyle = 'rgba(0, 220, 255, 0.2)'
      ctx.fillRect(bx, by, errW, errH)

      // Border of mass area
      ctx.strokeStyle = 'rgba(0, 220, 255, 0.5)'
      ctx.lineWidth = 1.5
      ctx.strokeRect(bx, by, errW, errH)

      // Center crosshair spanning mass area
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(bx, cy)
      ctx.lineTo(bx + errW, cy)
      ctx.moveTo(cx, by)
      ctx.lineTo(cx, by + errH)
      ctx.stroke()

      // Center dot
      ctx.shadowColor = 'rgba(0, 220, 255, 0.8)'
      ctx.shadowBlur = 10
      ctx.beginPath()
      ctx.arc(cx, cy, 3, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(0, 255, 255, 0.95)'
      ctx.fill()
      ctx.shadowBlur = 0

      // Label — adaptive position
      ctx.fillStyle = 'rgba(0, 255, 255, 0.9)'
      ctx.font = 'bold 11px monospace'
      const label = `${massCenter.az_deg > 0 ? '+' : ''}${massCenter.az_deg.toFixed(1)}° / ${massCenter.el_deg.toFixed(1)}°`
      const tw = ctx.measureText(label).width
      const labelX = (cx + errW / 2 + tw + 10 < plotW)
        ? cx + errW / 2 + 6
        : cx - errW / 2 - tw - 6
      const labelY = by - 4 > 14 ? by - 4 : by + errH + 14
      ctx.textAlign = 'left'
      ctx.textBaseline = 'bottom'
      ctx.fillText(label, labelX, labelY)
    } else {
      ctx.fillStyle = 'rgba(0, 200, 255, 0.12)'
      ctx.font = '13px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('NO SOURCE', plotW / 2, plotH / 2)
    }
  }, [massCenter, gridInfo])

  return (
    <div className="w-full h-full absolute inset-0">
      <canvas ref={canvasRef} className="block w-full h-full" />
    </div>
  )
}
