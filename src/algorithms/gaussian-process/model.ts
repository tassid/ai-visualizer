/**
 * Gaussian process regression.
 *
 * A Gaussian process is a distribution over functions: any finite set of
 * function values is jointly Gaussian, with covariance given by a kernel.
 * Here k(a, b) = σ_f² exp(-(a - b)² / (2ℓ²)), the squared exponential.
 *
 * Conditioning on observations (X, y) with noise σ_n² gives, at a test point x*:
 *   mean      μ(x*) = k*ᵀ (K + σ_n² I)⁻¹ y
 *   variance  σ²(x*) = k(x*, x*) - k*ᵀ (K + σ_n² I)⁻¹ k*
 * The animation reveals the observations one at a time and redraws the
 * posterior after each, so you can watch the uncertainty collapse near data.
 */
import { backSolve, cholesky, forwardSolve } from '../../core/linalg';
import { linspace } from '../../core/math';
import { randn } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface GpParams {
  lengthscale: number;
  noise: number;
  signal: number;
}

export const XS = linspace(-1, 1, 81);
const SAMPLES = 3;

export interface GpState {
  /** How many observations have been revealed. */
  seen: number;
  order: number[];
  mean: number[];
  std: number[];
  /** A few functions drawn from the posterior, sharing fixed random seeds so they move smoothly. */
  samples: number[][];
  z: number[][];
  logMarginal: number;
}

export const kernel = (a: number, b: number, P: GpParams) => P.signal * Math.exp(-((a - b) ** 2) / (2 * P.lengthscale ** 2));

/** Posterior over the grid XS given the first `m` observations. */
export function posterior(data: Point[], order: number[], m: number, P: GpParams, z: number[][]) {
  const obs = order.slice(0, m).map((i) => data[i]);
  const G = XS.length;
  if (!m) {
    const std = XS.map(() => Math.sqrt(P.signal));
    // prior samples: Cholesky of the prior covariance on the grid
    const Kss = XS.map((a) => XS.map((b) => kernel(a, b, P) + (a === b ? 1e-6 : 0)));
    const Ls = cholesky(Kss);
    const samples = z.map((zz) => XS.map((_, i) => { let v = 0; for (let k = 0; k <= i; k++) v += Ls[i][k] * zz[k]; return v; }));
    return { mean: XS.map(() => 0), std, samples, logMarginal: 0 };
  }
  // K + σ_n² I  =  L Lᵀ
  const K = obs.map((a, i) => obs.map((b, j) => kernel(a.x, b.x, P) + (i === j ? P.noise ** 2 : 0)));
  const L = cholesky(K);
  const y = obs.map((p) => p.y);
  const alpha = backSolve(L, forwardSolve(L, y));                       // (K + σ_n² I)⁻¹ y
  const mean: number[] = [], std: number[] = [];
  const V: number[][] = [];                                             // V = L⁻¹ K*
  for (let g = 0; g < G; g++) {
    const ks = obs.map((p) => kernel(XS[g], p.x, P));
    mean.push(ks.reduce((s, k, i) => s + k * alpha[i], 0));
    const v = forwardSolve(L, ks);
    V.push(v);
    std.push(Math.sqrt(Math.max(P.signal - v.reduce((s, a) => s + a * a, 0), 1e-12)));
  }
  // posterior covariance on the grid: K** - Vᵀ V, then samples = mean + chol · z
  const C = XS.map((a, i) => XS.map((b, j) => kernel(a, b, P) - V[i].reduce((s, vi, k) => s + vi * V[j][k], 0) + (i === j ? 1e-6 : 0)));
  const Lc = cholesky(C);
  const samples = z.map((zz) => mean.map((mu, i) => { let v = mu; for (let k = 0; k <= i; k++) v += Lc[i][k] * zz[k]; return v; }));
  // log marginal likelihood: -½ yᵀα - Σ log Lᵢᵢ - (n/2) log 2π
  const logMarginal = -0.5 * y.reduce((s, yi, i) => s + yi * alpha[i], 0) - L.reduce((s, row, i) => s + Math.log(row[i]), 0) - (m / 2) * Math.log(2 * Math.PI);
  return { mean, std, samples, logMarginal };
}

export function* train(data: Point[], P: GpParams, s: GpState): Generator<StepInfo, StepInfo> {
  for (let m = 1; m <= data.length; m++) {                              // line 1
    const p = data[s.order[m - 1]];
    s.seen = m;
    Object.assign(s, posterior(data, s.order, m, P, s.z));              // lines 2 to 5
    yield { line: 4, message: `Observed y = ${p.y.toFixed(2)} at x = ${p.x.toFixed(2)}. The band narrows around it.` };
  }
  return { line: 5, message: `Conditioned on all ${data.length} observations. Log marginal likelihood ${s.logMarginal.toFixed(2)}.` };
}

export function createGp(data: Point[], P: GpParams): Trainer<GpState> {
  const order = data.map((_, i) => i);
  const z = Array.from({ length: SAMPLES }, () => XS.map(() => randn()));
  const state: GpState = { seen: 0, order, z, ...posterior(data, order, 0, P, z) };
  return { state, steps: train(data, P, state) };
}
