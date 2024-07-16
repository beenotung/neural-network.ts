import { best } from 'ga-island'
import { sigmoid, create_ga, to_network_spec, forward } from './index'

let ga = create_ga({
  spec: to_network_spec({ sizes: [2, 2, 1], activation: sigmoid }),
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
for (let i = 1; i <= n; i++) {
  ga.evolve()
  let { fitness } = best(ga.options)
  let mse = -fitness
  console.log(i, mse.toFixed(12))
  if (mse < 1e-12) {
    break
  }
}

let network = best(ga.options).gene
console.dir(network, { depth: 20 })

function test(inputs: number[], target: number) {
  let output = forward(network, inputs)[0]
  console.log({ inputs, target, output })
}

test([0, 0], 0)
test([1, 0], 1)
test([0, 1], 1)
test([1, 1], 0)
