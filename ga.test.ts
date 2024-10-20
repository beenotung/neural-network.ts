import { best } from 'ga-island'
import {
  compile,
  create_ga,
  forward,
  linear,
  sample_to_fitness,
  sigmoid,
  tanh,
  to_json,
} from './index'
import { writeFileSync } from 'fs'
import debug from 'debug'
import { expect } from 'chai'

let log = debug('ga-nn xor')
// log.enabled = true

it('should train by genetic algorithm', () => {
  let epochs = 2000
  let min_error = 1e-6

  // epochs = 100_000_000
  // min_error = 1e-12

  let fitness = sample_to_fitness({
    inputs: [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ],
    targets: [[0], [1], [1], [0]],
  })
  fitness = network => {
    let diff = 0
    diff += Math.pow(forward(network, [0, 0])[0] - 0, 2)
    diff += Math.pow(forward(network, [1, 0])[0] - 1, 2)
    diff += Math.pow(forward(network, [0, 1])[0] - 1, 2)
    diff += Math.pow(forward(network, [1, 1])[0] - 0, 2)
    return -diff
  }

  let ga = create_ga({
    // spec: to_network_spec({ sizes: [2, 28, 28, 1], activation: relu }),
    spec: {
      layers: [
        { size: 2, activation: linear },
        { size: 2, activation: sigmoid },
        { size: 1, activation: sigmoid },
      ],
    },
    fitness,
    population_size: 1000,
    mutation_amount: 0.2,
  })

  for (let epoch = 1; epoch <= epochs; epoch++) {
    ga.evolve()
    let { fitness } = best(ga.options)
    let mse = -fitness
    log(epoch, mse.toFixed(12))
    if (mse < min_error) {
      break
    }
  }

  let network = best(ga.options).gene
  // console.dir(network, { depth: 20 })

  writeFileSync('ga-nn-xor.json', JSON.stringify(to_json(network), null, 2))
  log('saved to ga-nn-xor.json')

  let inference = compile(network)
  // log(inference.toString())

  writeFileSync(
    'ga-nn-xor.js',
    `
exports.inference = ${inference.toString()}
`,
  )
  log('saved to ga-nn-xor.js')

  function test(inputs: number[], target: number) {
    let output = inference(inputs)[0]
    log({ inputs, target, output })
    if (output > 0.8) output = 1
    else if (output < 0.2) output = 0
    expect(output).to.equals(target, `xor(${inputs}) -> ${target}`)
  }

  test([0, 0], 0)
  test([1, 0], 1)
  test([0, 1], 1)
  test([1, 1], 0)
}).timeout(10 * 1000)
