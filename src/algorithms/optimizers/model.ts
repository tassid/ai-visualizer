/**
 * Gradient descent and its variants racing on the same 2D function.
 *
 *   SGD:       θ ← θ - η g
 *   Momentum:  v ← β v + g ;  θ ← θ - η v
 *   RMSProp:   s ← ρ s + (1 - ρ) g² ;  θ ← θ - η g / (sqrt(s) + ε)
 *   Adam:      m ← β₁ m + (1 - β₁) g ;  v ← β₂ v + (1 - β₂) g²
 *              θ ← θ - η m̂ / (sqrt(v̂) + ε),  with m̂ = m / (1 - β₁ᵗ), v̂ = v / (1 - β₂ᵗ)
 *
 * Here g is the exact gradient; in deep learning it would be a noisy
 * estimate from a minibatch, but the geometry of each method is the same.
 */
import type { Point, StepInfo, Trainer } from '../../core/types';

export type Surface = 'narrow valley' | 'Rosenbrock' | 'two minima';

export interface OptParams {
  surface: Surface;
  lr: number;
  iters: number;
}

export interface Runner {
  name: 'SGD' | 'Momentum' | 'RMSProp' | 'Adam';
  x: number;
  y: number;
  path: [number, number][];
  m: [number, number];
  v: [number, number];
}

export interface OptState {
  surface: Surface;
  start: { x: number; y: number };
  runners: Runner[];
  t: number;
}

/** Each surface: value and gradient, defined directly on the plot's [-1, 1] square. */
export const SURFACES: Record<Surface, { f(x: number, y: number): number; grad(x: number, y: number): [number, number]; start: [number, number] }> = {
  // An ill-conditioned bowl: curvature 12 times larger along y than along x.
  'narrow valley': {
    f: (x, y) => 0.5 * (x * x + 12 * y * y),
    grad: (x, y) => [x, 12 * y],
    start: [-0.85, 0.55],
  },
  // A scaled Rosenbrock function: the minimum (0.5, 0.25) lies in a curved, flat-bottomed valley.
  Rosenbrock: {
    f: (x, y) => (0.5 - x) ** 2 + 5 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (0.5 - x) - 20 * x * (y - x * x), 10 * (y - x * x)],
    start: [-0.8, 0.85],
  },
  // A tilted double well: a shallow minimum on the right, a deeper one on the left.
  'two minima': {
    f: (x, y) => (x * x - 0.4) ** 2 + 0.5 * y * y + 0.15 * x,
    grad: (x, y) => [4 * x * (x * x - 0.4) + 0.15, y],
    start: [0.9, 0.75],
  },
};

const NAMES: Runner['name'][] = ['SGD', 'Momentum', 'RMSProp', 'Adam'];

export function resetRunners(s: OptState) {
  s.runners = NAMES.map((name) => ({ name, x: s.start.x, y: s.start.y, path: [[s.start.x, s.start.y]], m: [0, 0], v: [0, 0] }));
  s.t = 0;
}

/** One update of one optimizer. */
export function stepRunner(r: Runner, g: [number, number], lr: number, t: number) {
  const b1 = 0.9, b2 = 0.999, rho = 0.9, eps = 1e-8;
  for (let k = 0; k < 2; k++) {
    let delta: number;
    if (r.name === 'SGD') delta = lr * g[k];
    else if (r.name === 'Momentum') { r.m[k] = 0.9 * r.m[k] + g[k]; delta = lr * r.m[k]; }
    else if (r.name === 'RMSProp') { r.v[k] = rho * r.v[k] + (1 - rho) * g[k] * g[k]; delta = (lr * g[k]) / (Math.sqrt(r.v[k]) + eps); }
    else {
      r.m[k] = b1 * r.m[k] + (1 - b1) * g[k];
      r.v[k] = b2 * r.v[k] + (1 - b2) * g[k] * g[k];
      delta = (lr * (r.m[k] / (1 - b1 ** t))) / (Math.sqrt(r.v[k] / (1 - b2 ** t)) + eps);
    }
    if (k === 0) r.x -= delta; else r.y -= delta;
  }
  // keep runaway runs drawable
  r.x = Math.max(-3, Math.min(3, r.x));
  r.y = Math.max(-3, Math.min(3, r.y));
  r.path.push([r.x, r.y]);
}

export function* train(_data: Point[], P: OptParams, s: OptState): Generator<StepInfo, StepInfo> {
  const S = SURFACES[P.surface];
  for (let i = 1; i <= P.iters; i++) {                                  // line 0
    s.t++;
    for (const r of s.runners) stepRunner(r, S.grad(r.x, r.y), P.lr, s.t); // lines 1 to 4
    const best = s.runners.reduce((a, b) => (S.f(a.x, a.y) < S.f(b.x, b.y) ? a : b));
    yield { line: 1, message: `Step ${s.t}: ${best.name} is lowest, at f = ${S.f(best.x, best.y).toFixed(4)}.` };
  }
  return { line: 0, message: `${P.iters} steps. Drag the white start point to race again from somewhere else.` };
}

export function createOpt(_data: Point[], P: OptParams): Trainer<OptState> {
  const [sx, sy] = SURFACES[P.surface].start;
  const s: OptState = { surface: P.surface, start: { x: sx, y: sy }, runners: [], t: 0 };
  resetRunners(s);
  return { state: s, steps: train(_data, P, s) };
}
