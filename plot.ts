import { writeFileSync } from 'fs'
import { derivative, fn, fn_derivative } from './index'

for (let name in fn) {
  let activation = fn[name as keyof typeof fn]
  let text = `x\ty\td (formula)\td (sample)\n`
  for (let x = -6; x <= +6; x += 0.01) {
    let y = activation(x)
    let d_formula = fn_derivative.get(activation)?.(x)
    let d_sample = derivative(activation, x)
    text += `${x}\t${y}\t${d_formula}\t${d_sample}\n`
  }
  writeFileSync(`plot-${name}.csv`, text)
}
