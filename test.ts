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
  to_json,
} from './index'
import { writeFileSync } from 'fs'
