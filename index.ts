import { GaIsland } from 'ga-island'

export type Network = {
  /** @description layer -> output -> input -> weight */
  weights: number[][][]

  /** @description layer -> output -> bias */
  biases: number[][]

  /**
   * @description layer -> activation
   * @example sigmoid
   */
  activations: Activation[]
}

export let fn = {
  sigmoid,
  centered_sigmoid,
  tanh,
  normalized_tanh,
  linear,
  relu,
  elu,
}

export let fn_derivative = new Map<Activation, Activation>()
fn_derivative.set(sigmoid, sigmoid_prime)
fn_derivative.set(centered_sigmoid, centered_sigmoid_prime)
fn_derivative.set(tanh, tanh_prime)
fn_derivative.set(normalized_tanh, normalized_tanh_prime)
fn_derivative.set(linear, linear_prime)
fn_derivative.set(relu, relu_prime)
fn_derivative.set(elu, elu_prime)

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export function sigmoid_prime(x: number): number {
  let y = sigmoid(x)
  return y * (1 - y)
}

export function centered_sigmoid(x: number): number {
  return (1 / (1 + Math.exp(-x))) * 2 - 1
}

export function centered_sigmoid_prime(x: number): number {
  return sigmoid_prime(x) * 2
}

export function tanh(x: number): number {
  return Math.tanh(x)
}

export function tanh_prime(x: number): number {
  let y = tanh(x)
  return 1 - y * y
}

export function normalized_tanh(x: number): number {
  return (Math.tanh(x) + 1) / 2
}

export function normalized_tanh_prime(x: number): number {
  return tanh_prime(x) / 2
}

export function linear(x: number): number {
  return x
}

export function linear_prime(x: number): number {
  return 1
}

export function relu(x: number): number {
  return x < 0 ? 0 : x
}

export function relu_prime(x: number): number {
  return x < 0 ? 0 : 1
}

export function elu(x: number): number {
  return x < 0 ? Math.exp(x) - 1 : x
}

export function elu_prime(x: number): number {
  return x < 0 ? Math.exp(x) : 1
}

export function derivative(activation: Activation, x: number): number {
  let step = 2e-5
  let left = activation(x - step)
  let right = activation(x + step)
  let dy = right - left
  let dx = step + step
  return dy / dx
}

export type Activation = (x: number) => number

export type NetworkSpec = {
  /** [input_layer, ...hidden_layer, output_layer] */
  layers: LayerSpec[]
}

export type LayerSpec = {
  size: number
  activation: Activation
}

export function to_network_spec(options: {
  sizes: number[]
  activation: Activation
}): NetworkSpec {
  let { sizes, activation } = options
  let layers: LayerSpec[] = []
  layers.push({ size: sizes[0], activation: linear })
  for (let i = 1; i < sizes.length; i++) {
    layers.push({ size: sizes[i], activation })
  }
  return { layers }
}

export function random_network(options: NetworkSpec): Network {
  let { layers } = options
  let network: Network = {
    weights: [],
    biases: [],
    activations: [],
  }
  let { weights, biases, activations } = network
  for (let l = 1; l < layers.length; l++) {
    let input_size = layers[l - 1].size
    let output_size = layers[l].size
    let weight = new Array(output_size)
    let bias = new Array(output_size)
    for (let o = 0; o < output_size; o++) {
      let w = new Array(input_size)
      for (let i = 0; i < input_size; i++) {
        w[i] =
          /* avoid zero */
          Math.random() < 0.5
            ? random_between(-0.6, -0.4)
            : random_between(+0.4, +0.6)
        /* full range */
        // random_around_zero(1)
        /* 0..1 */
        // Math.random()
        /* zero */
        // 0
      }
      weight[o] = w
      bias[o] =
        /* slight bias */
        random_around_zero(0.1)
      /* full range */
      // random_around_zero(2)
      /* zero */
      // 0
      activations[l - 1] = layers[l].activation
    }
    weights.push(weight)
    biases.push(bias)
  }
  return network
}

export function random_between(min: number, max: number): number {
  let range = max - min
  return Math.random() * range + min
}

export function random_around_zero(range: number): number {
  return (Math.random() * 2 - 1) * range
}

export function forward(network: Network, inputs: number[]) {
  let { weights, biases, activations } = network
  let layer_size = weights.length

  for (let l = 0; l < layer_size; l++) {
    let bias = biases[l]
    let activation = activations[l]
    let input_size = weights[l][0].length
    let output_size = bias.length
    let outputs = new Array(output_size)
    for (let o = 0; o < output_size; o++) {
      let acc = 0
      let weight = weights[l][o]
      for (let i = 0; i < input_size; i++) {
        acc += weight[i] * inputs[i]
      }
      acc += bias[o]
      outputs[o] = activation(acc)
    }
    inputs = outputs
  }

  return inputs
}

export function learn(
  network: Network,
  inputs: number[],
  targets: number[],
  /** @example 0.2 or 0.01 */
  learning_rate: number,
) {
  let { weights, biases, activations } = network
  let layer_size = weights.length

  // layer index -> output index -> output value
  let values: number[][] = new Array(layer_size + 1)
  values[0] = inputs

  // forward
  for (let l = 0; l < layer_size; l++) {
    let bias = biases[l]
    let activation = activations[l]
    let input_size = weights[l][0].length
    let output_size = bias.length
    let outputs = new Array(output_size)
    for (let o = 0; o < output_size; o++) {
      let acc = 0
      let weight = weights[l][o]
      for (let i = 0; i < input_size; i++) {
        acc += weight[i] * inputs[i]
      }
      acc += bias[o]
      outputs[o] = activation(acc)
      values[l + 1] = outputs
    }
    inputs = outputs
  }

  // calculate error
  let mse = 0
  let output_errors = new Array(inputs.length)
  for (let i = 0; i < inputs.length; i++) {
    let e = targets[i] - inputs[i]
    output_errors[i] = e
    mse += e * e
  }
  mse /= inputs.length

  // backward
  for (let l = layer_size - 1; l >= 0; l--) {
    let bias = biases[l]
    const activation = activations[l]
    let activation_prime =
      fn_derivative.get(activation) ||
      ((x: number) => derivative(activation, x))
    let input_size = weights[l][0].length
    let output_size = bias.length
    let inputs = values[l]
    let outputs = values[l + 1]

    let input_errors = new Array(input_size).fill(0)

    for (let o = 0; o < output_size; o++) {
      let weight = weights[l][o]
      let output = outputs[o]

      // TODO handle when (output) or (1 - output) is zero
      let d_output = output_errors[o] * activation_prime(inputs[o])

      for (let i = 0; i < input_size; i++) {
        let input = inputs[i]

        // TODO handle when (input) is zero
        let d_input = d_output * weight[i]
        input_errors[i] += d_input

        weight[i] += d_output * input * learning_rate
      }

      bias[o] += d_output * learning_rate
    }

    output_errors = input_errors
  }

  return mse
}

export interface CompiledNetwork {
  (inputs: number[]): number[]
}

export function compile(network: Network): CompiledNetwork {
  let code_functions = ''
  let code_calc = ''

  let { weights, biases, activations } = network
  let layer_size = weights.length

  let counter = 0

  // function code -> name
  let function_cache = new Map<string, string>()

  function get_function_name(fn: Function): string {
    let code = fn.toString()
    let name = function_cache.get(code)
    if (!name) {
      name = 'f_' + (function_cache.size + 1)
      function_cache.set(code, name)
      code_functions += `
let ${name} = (${code})`
    }
    return name
  }

  let inputs: string[] = new Array(weights[0][0].length)
  for (let i = 0; i < inputs.length; i++) {
    inputs[i] = `inputs[${i}]`
  }
  for (let l = 0; l < layer_size; l++) {
    let bias = biases[l]
    let activation_fn_name = get_function_name(activations[l])
    let input_size = weights[l][0].length
    let output_size = bias.length
    let outputs = new Array(output_size)

    if (l == layer_size - 1) {
      code_calc += `
/* output layer */`
    } else {
      code_calc += `
/* layer ${l + 1} */`
    }

    for (let o = 0; o < output_size; o++) {
      counter++
      let output = `v_${counter}`
      code_calc += `
let ${output} = ${activation_fn_name}(0`
      let weight = weights[l][o]
      for (let i = 0; i < input_size; i++) {
        code_calc += ` + ${weight[i]} * ${inputs[i]}`
      }
      code_calc += ` + ${bias[o]})`
      outputs[o] = output
    }

    code_calc += `
`

    inputs = outputs
  }

  return new Function(
    'inputs',
    `
return function inference(inputs) {
/**************************
 ** activation functions **
 **************************/
${code_functions}

/************
 ** layers **
 ************/
${code_calc}
return [${inputs}]
}
`
      .trim()
      .split('\n')
      .join('\n  ')
      .replace(/  }$/, '}'),
  )() as CompiledNetwork
}

export type NetworkJSON = ReturnType<typeof to_json>

export function to_json(network: Network) {
  let fn_names = new Map(
    Object.entries(fn).map(([name, activation]) => [
      activation,
      name as keyof typeof fn,
    ]),
  )
  return {
    weights: network.weights,
    biases: network.biases,
    activations: network.activations.map((activation, index) => {
      let name = fn_names.get(activation)
      if (!name) {
        throw new Error(
          `unknown activation function, layer index: ${index}, function: ${activation}`,
        )
      }
      return name
    }),
  }
}

export function from_json(json: NetworkJSON): Network {
  return {
    weights: json.weights,
    biases: json.biases,
    activations: json.activations.map((name, index) => {
      let activation = fn[name]
      if (!name) {
        throw new Error(
          `unknown activation function, layer index: ${index}, name: ${name}`,
        )
      }
      return activation
    }),
  }
}

// TODO auto increase mutation_amount when the best fitness is not improving continuously
// TODO calculate diversity for doesABeatB
export function create_ga(args: {
  spec: NetworkSpec
  inputs: number[][]
  targets: number[][]
  /** @example 0.2 */
  mutation_amount: number
  /**
   * @description should be even number
   * @default 100
   * */
  population_size?: number
}) {
  let { spec, inputs, targets, mutation_amount } = args
  let { layers } = spec
  return new GaIsland<Network>({
    populationSize: args.population_size,
    crossover(aParent, bParent, child) {
      for (let l = 0; l < layers.length - 1; l++) {
        let input_size = layers[l].size
        let output_size = layers[l + 1].size
        for (let o = 0; o < output_size; o++) {
          for (let i = 0; i < input_size; i++) {
            let r = random_between(0.4, 0.6)
            child.weights[l][o][i] =
              aParent.weights[l][o][i] * r + bParent.weights[l][o][i] * (1 - r)
          }
          let r = random_between(0.4, 0.6)
          child.biases[l][o] =
            aParent.biases[l][o] * r + bParent.biases[l][o] * (1 - r)
        }
      }
    },
    mutate(input, output) {
      for (let l = 0; l < layers.length - 1; l++) {
        let input_size = layers[l].size
        let output_size = layers[l + 1].size
        for (let o = 0; o < output_size; o++) {
          for (let i = 0; i < input_size; i++) {
            output.weights[l][o][i] =
              input.weights[l][o][i] + random_around_zero(mutation_amount)
          }
          output.biases[l][o] =
            input.biases[l][o] + random_around_zero(mutation_amount)
        }
      }
    },
    fitness(network) {
      let output_size = layers[layers.length - 1].size
      let sample_size = inputs.length
      let error_acc = 0
      for (let i = 0; i < sample_size; i++) {
        let input = inputs[i]
        let target = targets[i]
        let output = forward(network, input)
        for (let o = 0; o < output_size; o++) {
          let e = target[o] - output[o]
          error_acc += e * e
        }
      }
      let mse = error_acc / output_size / sample_size
      return -mse
    },
    randomIndividual() {
      return random_network(spec)
    },
  })
}
