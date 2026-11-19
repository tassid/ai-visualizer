/**
 * Linear and quadratic discriminant analysis.
 *
 * Both model each class as a Gaussian, p(x | k) = N(x; μₖ, Σₖ), and classify
 * with Bayes' theorem. The difference is the covariance:
 *   QDA: each class has its own Σₖ  → quadratic boundary
 *   LDA: all classes share one Σ    → the quadratic terms cancel, linear boundary
 *
 * Unlike naive Bayes, Σ is a full matrix, so correlated features are fine.
 * Means and covariances are estimated online with Welford's update for
 * covariance (the 2D version of the one in naive-bayes/model.ts).
 */
import { inv2, mahalanobis2, det2, type Sym2 } from '../../core/linalg';
import { range, shuffle } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface DaParams {
  model: 'LDA' | 'QDA';
}

export interface Moments {
  n: number;
  mean: [number, number];
  /** Running sum of outer products of deviations; covariance = C / n. */
  C: Sym2;
}

export interface DaState {
  moments: [Moments, Moments];
  seen: number;
  current: number;
  model: 'LDA' | 'QDA';
}

const REG = 1e-3; // small ridge so a covariance estimated from few points stays invertible
const empty = (): Moments => ({ n: 0, mean: [0, 0], C: [0, 0, 0] });

/** Welford's update for the mean and the covariance of 2D points. */
export function update(m: Moments, x: number, y: number) {
  m.n++;
  const dx = x - m.mean[0], dy = y - m.mean[1];
  m.mean[0] += dx / m.n;
  m.mean[1] += dy / m.n;
  const ex = x - m.mean[0], ey = y - m.mean[1];
  m.C[0] += dx * ex;
  m.C[1] += dx * ey;
  m.C[2] += dy * ey;
}

export const covOf = (m: Moments): Sym2 => (m.n > 1 ? [m.C[0] / m.n + REG, m.C[1] / m.n, m.C[2] / m.n + REG] : [0.05, 0, 0.05]);

export function pooled(s: DaState): Sym2 {
  const [a, b] = s.moments, n = a.n + b.n;
  if (n < 3) return [0.05, 0, 0.05];
  return [(a.C[0] + b.C[0]) / n + REG, (a.C[1] + b.C[1]) / n, (a.C[2] + b.C[2]) / n + REG];
}

/** Covariance used for class k under the current model. */
export const sigmaFor = (s: DaState, k: 0 | 1): Sym2 => (s.model === 'LDA' ? pooled(s) : covOf(s.moments[k]));

/**
 * Discriminant δₖ(x) = log πₖ - ½ log|Σₖ| - ½ (x - μₖ)ᵀ Σₖ⁻¹ (x - μₖ).
 * The class with the larger δ wins.
 */
export function discriminant(s: DaState, k: 0 | 1, x: number, y: number): number {
  const m = s.moments[k], total = s.moments[0].n + s.moments[1].n;
  const S = sigmaFor(s, k);
  return Math.log(m.n / total) - 0.5 * Math.log(det2(S)) - 0.5 * mahalanobis2(x, y, m.mean, S);
}

export function posterior(s: DaState, x: number, y: number): number {
  if (s.moments[0].n < 2 || s.moments[1].n < 2) return 0.5;
  const d = discriminant(s, 1, x, y) - discriminant(s, 0, x, y);
  return 1 / (1 + Math.exp(-d));
}

/**
 * For LDA the boundary is the line w · x + b = 0 with
 *   w = Σ⁻¹ (μ₁ - μ₀),  b = -½ (μ₁ + μ₀)ᵀ Σ⁻¹ (μ₁ - μ₀) + log(π₁ / π₀).
 */
export function ldaLine(s: DaState): { w: [number, number]; b: number } | null {
  const [m0, m1] = s.moments;
  if (m0.n < 2 || m1.n < 2) return null;
  const [ia, ib, ic] = inv2(pooled(s));
  const dx = m1.mean[0] - m0.mean[0], dy = m1.mean[1] - m0.mean[1];
  const w: [number, number] = [ia * dx + ib * dy, ib * dx + ic * dy];
  const mx = (m1.mean[0] + m0.mean[0]) / 2, my = (m1.mean[1] + m0.mean[1]) / 2;
  return { w, b: -(w[0] * mx + w[1] * my) + Math.log(m1.n / m0.n) };
}

export function* train(data: Point[], _P: DaParams, s: DaState): Generator<StepInfo, StepInfo> {
  for (const i of shuffle(range(data.length))) {                        // line 1
    const p = data[i];
    s.current = i;
    update(s.moments[p.c], p.x, p.y);                                   // lines 2 to 4
    s.seen++;
    const m = s.moments[p.c];
    yield { line: 3, message: `Class ${p.c ? 'B' : 'A'}: n = ${m.n}, mean (${m.mean[0].toFixed(2)}, ${m.mean[1].toFixed(2)}).` };
  }
  s.current = -1;
  return { line: s.model === 'LDA' ? 5 : 6, message: s.model === 'LDA' ? 'Shared covariance estimated: the boundary is a straight line.' : 'One covariance per class: the boundary is a conic.' };
}

export function createDa(data: Point[], P: DaParams): Trainer<DaState> {
  const state: DaState = { moments: [empty(), empty()], seen: 0, current: -1, model: P.model };
  return { state, steps: train(data, P, state) };
}
