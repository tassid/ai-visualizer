/**
 * Linear (and polynomial) regression by gradient descent.
 *
 *   ŷ = b + w₁·x + w₂·x² + ... + w_d·x^d
 *   MSE = (1/n) Σ (ŷ - y)²
 *   ∂MSE/∂wₖ = (2/n) Σ (ŷ - y) · x^k
 *
 * With degree > 1 the model is still "linear" in its parameters: we only
 * replace x with polynomial features. That is what keeps the problem convex.
 *
 * There is also a closed form (the normal equation), w = (XᵀX)⁻¹ Xᵀy,
 * implemented in `normalEquation` to check the result in the tests.
 */
import { randn } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface LinRegParams {
  deg: number;
  lr: number;
  iters: number;
}

export interface LinRegState {
  w: number[];
  b: number;
  it: number;
  loss: number;
  /** Path of (w₁, b), drawn on the loss landscape (degree 1). */
  path: [number, number][];
  history: number[];
}

/** Evaluates the polynomial. Horner's rule would be faster; this spells out each power. */
export function evaluate(s: Pick<LinRegState, 'w' | 'b'>, x: number): number {
  let v = s.b, power = 1;
  for (let k = 0; k < s.w.length; k++) {
    power *= x;
    v += s.w[k] * power;
  }
  return v;
}

export function mse(s: Pick<LinRegState, 'w' | 'b'>, data: Point[]): number {
  if (!data.length) return 0;
  let sum = 0;
  for (const p of data) sum += (evaluate(s, p.x) - p.y) ** 2;
  return sum / data.length;
}

export function* train(data: Point[], P: LinRegParams, s: LinRegState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  const d = s.w.length;

  for (let it = 1; it <= P.iters; it++) {                               // line 1
    const g = new Array<number>(d).fill(0);
    let gb = 0;
    for (const p of data) {
      const e = evaluate(s, p.x) - p.y;                                 // line 2: residual
      gb += (2 * e) / n;                                                // line 4
      let power = 1;
      for (let k = 0; k < d; k++) {
        power *= p.x;
        g[k] += (2 * e * power) / n;                                    // line 3
      }
    }
    for (let k = 0; k < d; k++) s.w[k] -= P.lr * g[k];                  // line 5
    s.b -= P.lr * gb;                                                   // line 6
    s.it = it;
    s.loss = mse(s, data);
    s.history.push(s.loss);
    s.path.push([s.w[0], s.b]);
    const gn = Math.hypot(gb, ...g);
    yield { line: 5, message: `Iteration ${it}: mean squared error ${s.loss.toFixed(5)}.` };
    if (gn < 1e-6) return { line: 7, message: `The gradient vanished: minimum found at iteration ${it}.` };
  }
  return { line: 1, message: `Finished ${P.iters} iterations.` };
}

export function createLinReg(data: Point[], P: LinRegParams): Trainer<LinRegState> {
  // For degree 1 start at a fixed point far from the minimum, so the path on the landscape is visible.
  const w = P.deg === 1 ? [-1.6] : Array.from({ length: P.deg }, () => randn() * 0.1);
  const b = P.deg === 1 ? 0.8 : 0;
  const state: LinRegState = { w, b, it: 0, loss: NaN, path: [[w[0], b]], history: [] };
  state.loss = mse(state, data);
  return { state, steps: train(data, P, state) };
}

/** Exact least-squares answer for degree 1: w = cov(x, y) / var(x). */
export function normalEquation(data: Point[]): { w: number; b: number } {
  const n = data.length;
  const mx = data.reduce((s, p) => s + p.x, 0) / n;
  const my = data.reduce((s, p) => s + p.y, 0) / n;
  let cov = 0, vx = 0;
  for (const p of data) { cov += (p.x - mx) * (p.y - my); vx += (p.x - mx) ** 2; }
  const w = cov / vx;
  return { w, b: my - w * mx };
}
