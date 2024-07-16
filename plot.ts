import { writeFileSync } from 'fs'
import { derivative, fn } from './index'

for (let name in fn) {
  let activation = fn[name as keyof typeof fn]
  let text = `x\ty\td\n`
  for (let x = -6; x <= +6; x += 0.01) {
    let y = activation(x)
    let d = derivative(activation, x)
    text += `${x}\t${y}\t${d}\n`
  }
  writeFileSync(`plot-${name}.csv`, text)
}
