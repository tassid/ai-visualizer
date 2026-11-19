/**
 * Gaussian naive Bayes.
 *
 * By Bayes' theorem:  P(k | x) ∝ P(k) · P(x | k)
 * The "naive" assumption is that features are independent within a class:
 *   P(x | k) = Πⱼ N(xⱼ; μₖⱼ, σ²ₖⱼ)
 *
 * Training is just estimating means and variances per class. Here we do it
 * ONLINE, one point at a time, with Welford's algorithm: it updates the mean
 * and variance without storing the data and is numerically stable.
 *
 *   n ← n + 1
 *   δ ← x - μ
 *   μ ← μ + δ / n
 *   M ← M + δ · (x - μ)          (variance = M / n)
 */
import { gaussianPdf } from '../../core/math';
import { range, shuffle } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface NaiveBayesParams {
  smoothing: number;
}

export interface ClassStats {
  n: number;
  mean: [number, number];
  m2: [number, number];
}

export interface NaiveBayesState {
  stats: [ClassStats, ClassStats];
  seen: number;
  current: number;
  smoothing: number;
}

const emptyStats = (): ClassStats => ({ n: 0, mean: [0, 0], m2: [0, 0] });

export const variance = (st: ClassStats, j: number, smoothing: number) => (st.n > 1 ? st.m2[j] / st.n : 0.05) + smoothing;

/** Prior times likelihood, P(k) · p(x | k), for one class. */
export function joint(s: NaiveBayesState, k: 0 | 1, x: number, y: number): number {
  const total = s.stats[0].n + s.stats[1].n;
  const st = s.stats[k];
  if (!total || !st.n) return 0;
  return (st.n / total) * gaussianPdf(x, st.mean[0], variance(st, 0, s.smoothing)) * gaussianPdf(y, st.mean[1], variance(st, 1, s.smoothing));
}

/** P(class B | x, y). Stays at 0.5 until both classes have data. */
export function posterior(s: NaiveBayesState, x: number, y: number): number {
  if (s.stats[0].n < 1 || s.stats[1].n < 1) return 0.5;
  const a = joint(s, 0, x, y), b = joint(s, 1, x, y);
  return a + b === 0 ? 0.5 : b / (a + b);
}

/** One step of Welford's algorithm. */
export function welford(st: ClassStats, x: number, y: number) {
  st.n++;
  const v = [x, y];
  for (let j = 0; j < 2; j++) {
    const delta = v[j] - st.mean[j];
    st.mean[j] += delta / st.n;
    st.m2[j] += delta * (v[j] - st.mean[j]);
  }
}

export function* train(data: Point[], _P: NaiveBayesParams, s: NaiveBayesState): Generator<StepInfo, StepInfo> {
  for (const i of shuffle(range(data.length))) {                        // line 1
    const p = data[i];
    s.current = i;
    welford(s.stats[p.c], p.x, p.y);                                    // lines 2 to 4
    s.seen++;
    const st = s.stats[p.c];
    yield { line: 3, message: `Class ${p.c ? 'B' : 'A'}: n = ${st.n}, mean (${st.mean[0].toFixed(2)}, ${st.mean[1].toFixed(2)}).` };
  }
  s.current = -1;
  return { line: 6, message: 'Means and variances estimated. The background shows P(B | x) from Bayes’ theorem.' };
}

export function createNaiveBayes(data: Point[], P: NaiveBayesParams): Trainer<NaiveBayesState> {
  const state: NaiveBayesState = { stats: [emptyStats(), emptyStats()], seen: 0, current: -1, smoothing: P.smoothing };
  return { state, steps: train(data, P, state) };
}
