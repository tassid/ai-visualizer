/**
 * Linear SVM trained by subgradient descent (Pegasos style, full batch).
 *
 * Objective (primal form):
 *   min  λ/2 · |w|²  +  (1/n) Σ max(0, 1 - yᵢ(w·xᵢ + b))
 *        ─────────      ─────────────────────────────────
 *        wide margin    hinge loss: penalizes points inside the margin
 *
 * The geometric margin is 2 / |w|. Shrinking |w| (regularization) widens
 * the margin; the hinge loss pushes the line away from the points.
 *
 * The hinge loss has a kink at y·f = 1, so we use a subgradient and a
 * learning rate that decays over time.
 */
import { uniform } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface SvmParams {
  lambda: number;
  lr: number;
  iters: number;
}

export interface SvmState {
  w: [number, number];
  b: number;
  it: number;
  loss: number;
  history: number[];
}

export const decision = (s: SvmState, x: number, y: number) => s.w[0] * x + s.w[1] * y + s.b;

/** Support vectors: points on or inside the margin (y·f ≤ 1). */
export const isSupport = (s: SvmState, p: Point) => (p.c ? 1 : -1) * decision(s, p.x, p.y) <= 1 + 1e-4;

export const margin = (s: SvmState) => 2 / (Math.hypot(...s.w) || 1e-9);

export function hingeObjective(s: SvmState, data: Point[], lambda: number): number {
  let loss = (lambda / 2) * (s.w[0] ** 2 + s.w[1] ** 2);
  for (const p of data) loss += Math.max(0, 1 - (p.c ? 1 : -1) * decision(s, p.x, p.y)) / data.length;
  return loss;
}

export function* train(data: Point[], P: SvmParams, s: SvmState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };

  for (let it = 1; it <= P.iters; it++) {                               // line 0
    let gw1 = P.lambda * s.w[0], gw2 = P.lambda * s.w[1], gb = 0;       // line 1
    let violators = 0;
    for (const p of data) {                                             // line 2
      const y = p.c ? 1 : -1;
      if (y * decision(s, p.x, p.y) < 1) {                              // line 3
        gw1 -= (y * p.x) / n;                                           // line 4
        gw2 -= (y * p.y) / n;
        gb -= y / n;                                                    // line 5
        violators++;
      }
    }
    const lr = P.lr / (1 + it * 0.003);                                 // decaying step size
    s.w[0] -= lr * gw1;                                                 // line 6
    s.w[1] -= lr * gw2;
    s.b -= lr * gb;
    s.it = it;
    s.loss = hingeObjective(s, data, P.lambda);
    s.history.push(s.loss);
    yield { line: 6, message: `Iteration ${it}: ${violators} point(s) inside the margin, margin width ${margin(s).toFixed(3)}.` };
  }
  return { line: 0, message: `Finished ${P.iters} iterations.` };
}

export function createSvm(data: Point[], P: SvmParams): Trainer<SvmState> {
  const state: SvmState = { w: [uniform(-1, 1), uniform(-1, 1)], b: 0, it: 0, loss: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
