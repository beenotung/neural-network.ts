import debug from 'debug'
import {
  compile,
  learn,
  linear,
  random_network,
  relu,
  sigmoid,
  to_json,
} from './index'
import { writeFileSync } from 'fs'
import { expect } from 'chai'

let log = debug('bp-nn xor')
log.enabled = true

it('should train by back-propagation', () => {
  let epochs = 2000
  let min_error = 1e-6
  let learning_rate = 0.001

  // epochs = 100_000_000
  // min_error = 1e-12

  let activation = sigmoid
  let network = random_network({
    layers: [
      { size: 2, activation: linear },
      { size: 2, activation },
      { size: 1, activation },
    ],
  })
  let inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ]
  let targets = [[0], [1], [1], [0]]
  let sample_size = inputs.length

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let mse = 0
    for (let i = 0; i < sample_size; i++) {
      mse += learn(network, inputs[i], targets[i], learning_rate)
    }
    mse /= sample_size

    // if (epoch % 1_000_000 == 0) {
    log(epoch.toLocaleString(), mse.toFixed(12))
    // }
  }

  console.dir(network, { depth: 20 })

  writeFileSync('bp-nn-xor.json', JSON.stringify(to_json(network), null, 2))
  log('saved to bp-nn-xor.json')

  let inference = compile(network)
  log(inference.toString())

  writeFileSync(
    'bp-nn-xor.js',
    `
exports.inference = ${inference.toString()}
`,
  )
  log('saved to bp-nn-xor.js')

  function test(inputs: number[], target: number) {
    let output = inference(inputs)[0]
    log({ inputs, target, output })
    if (output > 0.8) output = 1
    else if (output < 0.2) output = 0
    // expect(output).to.equals(target)
  }

  test([0, 0], 0)
  test([1, 0], 1)
  test([0, 1], 1)
  test([1, 1], 0)
})
