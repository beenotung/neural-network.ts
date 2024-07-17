import { expect } from 'chai'
import { derivative } from '.'
import { writeFileSync } from 'fs'
import { fn, fn_derivative } from './index'

context('derivative', () => {
  for (let name in fn) {
    const activation = fn[name as keyof typeof fn]
    const activation_prime = fn_derivative.get(activation)
    if (activation_prime) {
      it(`derive ${name}`, () => {
        let text = `x\texpected\tactual\terror\n`
        // let text = `x\te\n`
        for (let x = -6; x <= 6; x += 0.1) {
          let y = activation(x)
          let actual = derivative(activation, x)
          let expected = activation_prime(x)
          let error = Math.abs(expected - actual)
          let allowed_error = 1e-6
          if (name == 'relu' && Math.abs(x) < 1e-12) {
            expected = 0.5
          }
          if (name == 'elu') {
            allowed_error = 1e-5
          }
          expect(actual).to.approximately(expected, allowed_error)
          text += `${x}\t${expected}\t${actual}\t${error}\n`
        }
        writeFileSync(`derive-${name}.csv`, text)
      })
    } else {
      it.skip(`derive ${name}`)
    }
  }
})
