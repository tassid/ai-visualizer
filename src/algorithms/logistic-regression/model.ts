/**
 * Logistic regression (binary classification).
 *
 *   p(y = 1 | x) = σ(w · x + b)
 *   loss = mean cross-entropy
 *   ∂loss/∂w = mean of (p - y) · x       ← the same shape as linear regression
 *
 * The loss is convex, so gradient descent always reaches the global minimum
 * (unless the learning rate is too large). On perfectly separable data,
 * though, the weights grow without bound: the model gets ever more
 * confident, which shows as an ever sharper boundary.
 */
import { binaryCrossEntropy, sigmoid } from '../../core/math';
import { uniform } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface LogisticParams {
  lr: number;
  iters: number;
  l2: number;
}

export interface LogisticState {
  w: [number, number];
  b: number;
  it: number;
  loss: number;
  gradNorm: number;
  history: number[];
}

export const probability = (s: LogisticState, x: number, y: number) => sigmoid(s.w[0] * x + s.w[1] * y + s.b);

export function logLoss(s: LogisticState, data: Point[], l2 = 0): number {
  let loss = 0;
  for (const p of data) loss += binaryCrossEntropy(probability(s, p.x, p.y), p.c) / data.length;
  return loss + (l2 / 2) * (s.w[0] ** 2 + s.w[1] ** 2);
}

export function* train(data: Point[], P: LogisticParams, s: LogisticState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };

  for (let it = 1; it <= P.iters; it++) {                               // line 1
    let g1 = 0, g2 = 0, gb = 0;
    for (const p of data) {
      const e = probability(s, p.x, p.y) - p.c;                         // line 2
      g1 += (e * p.x) / n;                                              // line 3
      g2 += (e * p.y) / n;
      gb += e / n;                                                      // line 4
    }
    // L2 regularization (the bias is not penalized)
    g1 += P.l2 * s.w[0];
    g2 += P.l2 * s.w[1];

    s.w[0] -= P.lr * g1;                                                // line 5
    s.w[1] -= P.lr * g2;
    s.b -= P.lr * gb;
    s.it = it;
    s.loss = logLoss(s, data, P.l2);
    s.gradNorm = Math.hypot(g1, g2, gb);
    s.history.push(s.loss);
    yield { line: 5, message: `Iteration ${it}: loss ${s.loss.toFixed(4)}, gradient length ${s.gradNorm.toFixed(4)}.` };
    if (s.gradNorm < 1e-4) return { line: 6, message: `The gradient is nearly zero: minimum reached at iteration ${it}.` };
  }
  return { line: 1, message: `Finished ${P.iters} iterations.` };
}

export function createLogistic(data: Point[], P: LogisticParams): Trainer<LogisticState> {
  const state: LogisticState = { w: [uniform(-1, 1), uniform(-1, 1)], b: 0, it: 0, loss: NaN, gradNorm: NaN, history: [] };
  state.loss = data.length ? logLoss(state, data, P.l2) : NaN;
  return { state, steps: train(data, P, state) };
}
