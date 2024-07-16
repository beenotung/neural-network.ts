import { best } from 'ga-island'
import {
  sigmoid,
  create_ga,
  to_network_spec,
  forward,
  compile,
  tanh,
  linear,
  relu,
} from './index'
import { writeFileSync } from 'fs'

let ga = create_ga({
  // spec: to_network_spec({ sizes: [2, 2, 1], activation: sigmoid }),
  spec: {
    layers: [
      { size: 2, activation: linear },
      { size: 3, activation: relu },
      { size: 1, activation: relu },
    ],
  },
  inputs: [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ],
  targets: [[0], [1], [1], [0]],
  population_size: 1000,
  mutation_amount: 0.2,
})

let n = 100_000_000
let min_error = 1e-12
min_error = 1e-6

for (let i = 1; i <= n; i++) {
  ga.evolve()
  let { fitness } = best(ga.options)
  let mse = -fitness
  console.log(i, mse.toFixed(12))
  if (mse < min_error) {
    break
  }
}

let network = best(ga.options).gene
console.dir(network, { depth: 20 })

let inference = compile(network)

console.log(inference.toString())
writeFileSync(
  'xor.js',
  `
exports.inference = ${inference.toString()}
`,
)

function test(inputs: number[], target: number) {
  let output = inference(inputs)[0]
  console.log({ inputs, target, output })
}

test([0, 0], 0)
test([1, 0], 1)
test([0, 1], 1)
test([1, 1], 0)
