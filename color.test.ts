import { rgb_to_oklab } from 'oklab.ts'
import {
  compile,
  create_ga,
  forward,
  learn,
  linear,
  random_network,
  relu,
  sigmoid,
  tanh,
  to_json,
} from './index'
import { best } from 'ga-island'
import debug from 'debug'
import { writeFileSync } from 'fs'
import { expect } from 'chai'

it('should convert rgb to oklab with ga', () => {
  let log = debug('ga-nn color')
  // log.enabled = true
  let ga = create_ga({
    spec: {
      layers: [
        { size: 3, activation: linear },
        { size: 3, activation: linear },
        { size: 3, activation: linear },
      ],
    },
    fitness: network => {
      let diff = 0

      function take_sample() {
        let rgb = {
          r: Math.round(Math.random() * 255),
          g: Math.round(Math.random() * 255),
          b: Math.round(Math.random() * 255),
        }
        let oklab = rgb_to_oklab(rgb)

        let output = forward(network, [rgb.r, rgb.b, rgb.g])

        diff += Math.sqrt(Math.pow(output[0] - oklab.L, 2))
        diff += Math.sqrt(Math.pow(output[1] - oklab.a, 2))
        diff += Math.sqrt(Math.pow(output[2] - oklab.b, 2))
      }

      let n_sample = 100
      for (let i = 0; i < n_sample; i++) {
        take_sample()
      }

      return -diff / n_sample
    },
    population_size: 1000,
    mutation_amount: 0.2,
  })

  let epochs = 20
  let min_error = 1.5
  let mse = 0
  for (let epoch = 1; epoch <= epochs; epoch++) {
    ga.evolve()
    let { fitness, gene } = best(ga.options)
    mse = -fitness
    log(epoch, mse.toFixed(12))
    // take_sample()
    function take_sample() {
      let rgb = {
        r: Math.round(Math.random() * 255),
        g: Math.round(Math.random() * 255),
        b: Math.round(Math.random() * 255),
      }
      let oklab = rgb_to_oklab(rgb)

      let output = forward(gene, [rgb.r, rgb.b, rgb.g])
      // log({ rgb, oklab, output })
    }
    if (mse < min_error) {
      break
    }
  }
  expect(mse).to.be.lessThan(min_error, 'mse should be low')

  let network = best(ga.options).gene
  writeFileSync('ga-nn-color.json', JSON.stringify(to_json(network), null, 2))
  log('saved to ga-nn-color.json')

  let inference = compile(network)
  log(inference.toString())

  writeFileSync('ga-nn-color.js', `exports.inference = ${inference.toString()}`)
  log('saved to ga-nn-color.js')
}).timeout(5000)

it('should convert rgb to oklab with back-propagation', () => {
  let log = debug('bp-nn color')
  // log.enabled = true
  let network = random_network({
    layers: [
      { size: 3, activation: linear },
      { size: 3, activation: tanh },
      { size: 3, activation: tanh },
    ],
  })
  let inputs: number[][] = []
  let targets: number[][] = []
  let n_sample = 255 * 255
  for (let i = 0; i < n_sample; i++) {
    let rgb = {
      r: Math.round(Math.random() * 255),
      g: Math.round(Math.random() * 255),
      b: Math.round(Math.random() * 255),
    }
    let oklab = rgb_to_oklab(rgb)
    inputs.push([rgb.r, rgb.b, rgb.g])
    targets.push([oklab.L, oklab.a, oklab.b])
  }
  let learning_rate = 0.003
  let mse = 0
  let epochs = 20
  let min_error = 0.02
  for (let epoch = 1; epoch <= epochs; epoch++) {
    mse = 0
    for (let i = 0; i < n_sample; i++) {
      mse += learn(network, inputs[i], targets[i], learning_rate)
    }
    mse /= n_sample
    log(epoch.toLocaleString(), mse.toFixed(12))
    // take_sample()
    function take_sample() {
      let rgb = {
        r: Math.round(Math.random() * 255),
        g: Math.round(Math.random() * 255),
        b: Math.round(Math.random() * 255),
      }
      let oklab = rgb_to_oklab(rgb)
      let output = forward(network, [rgb.r, rgb.b, rgb.g])
      // log({ rgb, oklab, output })
      if (output.some(v => Number.isNaN(v))) {
        console.error('NaN')
        debugger
      }
    }
    if (mse < min_error) {
      break
    }
  }
  expect(mse).to.be.lessThan(min_error, 'mse should be low')
  writeFileSync('bp-nn-color.json', JSON.stringify(to_json(network), null, 2))
  log('saved to bp-nn-color.json')

  let inference = compile(network)
  log(inference.toString())

  writeFileSync('bp-nn-color.js', `exports.inference = ${inference.toString()}`)
  log('saved to bp-nn-color.js')
})
