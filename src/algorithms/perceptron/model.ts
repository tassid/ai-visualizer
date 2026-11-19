/**
 * Rosenblatt's perceptron (1958).
 *
 * Model: f(x) = w · x + b ; class = sign(f(x))
 * Learning rule: it only learns when it is wrong.
 *   if y · f(x) ≤ 0:  w ← w + η·y·x ;  b ← b + η·y
 *
 * Convergence theorem (Novikoff, 1962): if a line separates the data with
 * margin γ and every point has norm ≤ R, the perceptron makes at most
 * (R/γ)² updates. If the data is NOT separable (XOR), it keeps correcting
 * forever.
 */
import { range, shuffle, uniform } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface PerceptronParams {
  lr: number;
  epochs: number;
}

export interface PerceptronState {
  w: [number, number];
  b: number;
  epoch: number;
  /** Mistakes in the current epoch. */
  errors: number;
  /** Index of the point being examined. */
  current: number;
  /** Did the last step change the weights? */
  updated: boolean;
  /** Position inside the epoch. */
  seen: number;
  /** Mistakes at the end of each epoch. */
  history: number[];
}

/** Raw score f(x) = w · x + b. */
export const score = (s: PerceptronState, x: number, y: number) => s.w[0] * x + s.w[1] * y + s.b;

export const predict = (s: PerceptronState, x: number, y: number) => (score(s, x, y) >= 0 ? 1 : 0);

export function initialState(): PerceptronState {
  return { w: [uniform(-0.3, 0.3), uniform(-0.3, 0.3)], b: 0, epoch: 0, errors: 0, current: -1, updated: false, seen: 0, history: [] };
}

/**
 * The algorithm, line by line. Each `yield` becomes one animation step, and
 * `line` points at the pseudocode line (see index.tsx).
 */
export function* train(data: Point[], P: PerceptronParams, s: PerceptronState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'No data. Click the plane to add points.' };
  const order = range(data.length);

  for (let epoch = 0; epoch < P.epochs; epoch++) {                      // line 1
    s.epoch = epoch;
    s.errors = 0;                                                       // line 2
    shuffle(order);                                                     // line 3
    for (let k = 0; k < order.length; k++) {
      const i = order[k];
      const p = data[i];
      s.current = i;
      s.seen = k + 1;
      const y = p.c ? 1 : -1;                                           // labels in {-1, +1}
      const f = score(s, p.x, p.y);                                     // line 4

      if (y * f <= 0) {                                                 // line 5: wrong (or on the line)
        s.w[0] += P.lr * y * p.x;                                       // line 6
        s.w[1] += P.lr * y * p.y;
        s.b += P.lr * y;                                                // line 7
        s.errors++;                                                     // line 8
        s.updated = true;
        yield { line: 6, message: `Wrong side: y·f = ${(y * f).toFixed(2)} ≤ 0, so w moves ${y > 0 ? 'toward' : 'away from'} this point.` };
      } else {
        s.updated = false;
        yield { line: 4, message: `Correct: y·f = ${(y * f).toFixed(2)} > 0, nothing changes.` };
      }
    }
    s.history.push(s.errors);
    if (s.errors === 0) return { line: 9, message: `Converged in epoch ${epoch + 1}: no mistakes, the line separates the data.` };
    yield { line: 9, message: `End of epoch ${epoch + 1}: ${s.errors} mistake(s). Another pass.` };
  }
  return { line: 1, message: `Stopped after ${P.epochs} epochs without converging. The data may not be linearly separable.` };
}

export function createPerceptron(data: Point[], P: PerceptronParams): Trainer<PerceptronState> {
  const state = initialState();
  return { state, steps: train(data, P, state) };
}
