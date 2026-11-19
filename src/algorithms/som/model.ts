/**
 * Self-organizing map (Kohonen, 1982).
 *
 * A grid of neurons, each with a weight vector living in the data space.
 * For each sample x:
 *   1. find the best matching unit (BMU) c, the neuron closest to x
 *   2. pull every neuron j toward x, more strongly the closer j is to c ON THE GRID:
 *        wⱼ ← wⱼ + η(t) · h(j, c, t) · (x - wⱼ),   h = exp(-d_grid(j, c)² / (2σ(t)²))
 * Both the learning rate η and the neighborhood width σ shrink over time.
 * Neighbors on the grid end up neighbors in the data: the map unfolds over
 * the data while keeping its topology.
 */
import { randInt, uniform } from '../../core/random';
import { sqDist } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface SomParams {
  size: number;
  steps: number;
  lr: number;
}

export interface SomState {
  size: number;
  /** w[r * size + c] = position of the neuron at row r, column c. */
  w: { x: number; y: number }[];
  t: number;
  bmu: number;
  sample: number;
  eta: number;
  sigma: number;
}

export function bmu(s: SomState, x: number, y: number): number {
  let best = 0, bd = Infinity;
  s.w.forEach((n, j) => { const d = sqDist(n.x, n.y, x, y); if (d < bd) { bd = d; best = j; } });
  return best;
}

/** Average distance from each point to its BMU: how well the map covers the data. */
export function quantizationError(s: SomState, data: Point[]): number {
  if (!data.length) return NaN;
  let e = 0;
  for (const p of data) { const n = s.w[bmu(s, p.x, p.y)]; e += Math.hypot(n.x - p.x, n.y - p.y); }
  return e / data.length;
}

export function* train(data: Point[], P: SomParams, s: SomState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  const N = s.size, sigma0 = N / 2;
  for (let t = 1; t <= P.steps; t++) {                                  // line 0
    const frac = t / P.steps;
    s.eta = P.lr * (1 - frac) + 0.01;                                   // line 3
    s.sigma = sigma0 * (1 - frac) + 0.5;
    const i = randInt(n);                                               // line 1
    const p = data[i];
    const c = bmu(s, p.x, p.y);                                         // line 2
    const cr = Math.floor(c / N), cc = c % N;
    s.w.forEach((wj, j) => {                                            // lines 4 and 5
      const dr = Math.floor(j / N) - cr, dc = (j % N) - cc;
      const h = Math.exp(-(dr * dr + dc * dc) / (2 * s.sigma * s.sigma));
      wj.x += s.eta * h * (p.x - wj.x);
      wj.y += s.eta * h * (p.y - wj.y);
    });
    s.t = t; s.bmu = c; s.sample = i;
    yield { line: 5, message: `Step ${t}: neuron (${cr}, ${cc}) is closest; it and its grid neighbors move toward the sample. η = ${s.eta.toFixed(3)}, σ = ${s.sigma.toFixed(2)}.` };
  }
  s.sample = -1;
  return { line: 0, message: 'The map has unfolded over the data.' };
}

export function createSom(data: Point[], P: SomParams): Trainer<SomState> {
  const N = Number(P.size);
  const w = Array.from({ length: N * N }, () => ({ x: uniform(-0.1, 0.1), y: uniform(-0.1, 0.1) }));
  const state: SomState = { size: N, w, t: 0, bmu: -1, sample: -1, eta: P.lr, sigma: N / 2 };
  return { state, steps: train(data, P, state) };
}
