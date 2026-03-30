export function jetColormap(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  let r: number, g: number, b: number

  if (t < 0.125) {
    r = 0; g = 0; b = 0.5 + t * 4
  } else if (t < 0.375) {
    r = 0; g = (t - 0.125) * 4; b = 1
  } else if (t < 0.625) {
    r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4
  } else if (t < 0.875) {
    r = 1; g = 1 - (t - 0.625) * 4; b = 0
  } else {
    r = 1 - (t - 0.875) * 4; g = 0; b = 0
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}
