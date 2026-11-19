/**
 * Kernel SVM trained with kernelized Pegasos (Shalev-Shwartz et al., 2011).
 *
 * The decision function lives in the span of the training points:
 *   f(x) = (1 / (λ t)) Σⱼ αⱼ yⱼ K(xⱼ, x)
 * where αⱼ counts how many times point j violated the margin when it was
 * drawn. Each step draws one point i at random and, if y_i f(x_i) < 1,
 * increments α_i. The points with α > 0 are the support vectors.
 *
 * Kernel: RBF, K(a, b) = exp(-γ |a - b|²). It measures similarity, and the
 * implicit feature space is infinite dimensional.
 */
import { randInt } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface KsvmParams {
  gamma: number;
  lambda: number;
  iters: number;
}

export interface KsvmState {
  alpha: Uint32Array;
  t: number;
  current: number;
  violated: boolean;
  gamma: number;
  lambda: number;
}

export const rbf = (ax: number, ay: number, bx: number, by: number, gamma: number) => Math.exp(-gamma * ((ax - bx) ** 2 + (ay - by) ** 2));

export function decision(s: KsvmState, data: Point[], x: number, y: number, t = s.t): number {
  if (t === 0) return 0;
  let sum = 0;
  for (let j = 0; j < data.length; j++) {
    const a = s.alpha[j];
    if (a) sum += a * (data[j].c ? 1 : -1) * rbf(data[j].x, data[j].y, x, y, s.gamma);
  }
  return sum / (s.lambda * t);
}

export const supportCount = (s: KsvmState) => s.alpha.reduce((n, a) => n + (a > 0 ? 1 : 0), 0);

export function* train(data: Point[], P: KsvmParams, s: KsvmState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  for (let t = 1; t <= P.iters; t++) {                                  // line 0
    const i = randInt(n);                                               // line 1
    s.current = i;
    const p = data[i], y = p.c ? 1 : -1;
    const margin = y * decision(s, data, p.x, p.y, t);                  // line 2
    s.t = t;
    s.violated = margin < 1;
    if (s.violated) {                                                   // line 3
      s.alpha[i]++;                                                     // line 4
      yield { line: 4, message: `Step ${t}: y·f = ${margin.toFixed(2)} < 1, so this point gains weight (α = ${s.alpha[i]}).` };
    } else {
      yield { line: 2, message: `Step ${t}: y·f = ${margin.toFixed(2)} ≥ 1, outside the margin, nothing changes.` };
    }
  }
  return { line: 5, message: `Done: ${supportCount(s)} support vectors out of ${n} points.` };
}

export function createKsvm(data: Point[], P: KsvmParams): Trainer<KsvmState> {
  const state: KsvmState = { alpha: new Uint32Array(data.length), t: 0, current: -1, violated: false, gamma: P.gamma, lambda: P.lambda };
  return { state, steps: train(data, P, state) };
}
