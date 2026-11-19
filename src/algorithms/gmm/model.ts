/**
 * Gaussian mixture model fitted by expectation-maximization (EM).
 *
 *   p(x) = Σₖ πₖ N(x; μₖ, Σₖ)
 *
 * E step: responsibilities, the posterior probability that point i came from
 *         component k:  γᵢₖ = πₖ N(xᵢ; μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ; μⱼ, Σⱼ)
 * M step: re-estimate each component from the points, weighted by γ:
 *         Nₖ = Σᵢ γᵢₖ,  πₖ = Nₖ / n,  μₖ = Σᵢ γᵢₖ xᵢ / Nₖ,
 *         Σₖ = Σᵢ γᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ / Nₖ
 * Each EM iteration never decreases the log-likelihood.
 */
import { gauss2, type Sym2 } from '../../core/linalg';
import { initCentroids } from '../kmeans/model';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface GmmParams {
  k: number;
  iters: number;
}

export interface Component {
  pi: number;
  mu: [number, number];
  cov: Sym2;
}

export interface GmmState {
  comps: Component[];
  /** gamma[i][k] */
  gamma: Float64Array[];
  phase: 'E' | 'M';
  iter: number;
  logLik: number;
  history: number[];
}

const REG = 1e-4;

export function logLikelihood(data: Point[], comps: Component[]): number {
  let ll = 0;
  for (const p of data) {
    let s = 0;
    for (const c of comps) s += c.pi * gauss2(p.x, p.y, c.mu, c.cov);
    ll += Math.log(s + 1e-300);
  }
  return ll;
}

export function eStep(data: Point[], s: GmmState) {
  data.forEach((p, i) => {
    let total = 0;
    s.comps.forEach((c, k) => { s.gamma[i][k] = c.pi * gauss2(p.x, p.y, c.mu, c.cov); total += s.gamma[i][k]; });
    for (let k = 0; k < s.comps.length; k++) s.gamma[i][k] = total > 0 ? s.gamma[i][k] / total : 1 / s.comps.length;
  });
}

export function mStep(data: Point[], s: GmmState) {
  const n = data.length;
  s.comps.forEach((c, k) => {
    let Nk = 0, mx = 0, my = 0;
    data.forEach((p, i) => { const g = s.gamma[i][k]; Nk += g; mx += g * p.x; my += g * p.y; });
    if (Nk < 1e-8) return; // an empty component stays where it is
    mx /= Nk; my /= Nk;
    let a = 0, b = 0, cc = 0;
    data.forEach((p, i) => { const g = s.gamma[i][k], dx = p.x - mx, dy = p.y - my; a += g * dx * dx; b += g * dx * dy; cc += g * dy * dy; });
    c.pi = Nk / n;
    c.mu = [mx, my];
    c.cov = [a / Nk + REG, b / Nk, cc / Nk + REG];
  });
}

export function* train(data: Point[], P: GmmParams, s: GmmState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'No data. Click the plane to add points.' };
  let prev = -Infinity;
  for (let it = 1; it <= P.iters; it++) {                               // line 1
    s.phase = 'E';
    eStep(data, s);                                                     // line 2
    s.logLik = logLikelihood(data, s.comps);
    yield { line: 2, message: `E step ${it}: each point splits its membership among the components.` };
    s.phase = 'M';
    mStep(data, s);                                                     // lines 3 to 5
    s.iter = it;
    s.logLik = logLikelihood(data, s.comps);
    s.history.push(s.logLik);
    yield { line: 5, message: `M step ${it}: components move to their weighted points. Log-likelihood ${s.logLik.toFixed(2)}.` };
    if (Math.abs(s.logLik - prev) < 1e-4) return { line: 6, message: `Converged after ${it} iterations: the log-likelihood stopped improving.` };
    prev = s.logLik;
  }
  return { line: 1, message: `Stopped after ${P.iters} iterations.` };
}

export function createGmm(data: Point[], P: GmmParams): Trainer<GmmState> {
  const k = Math.max(1, Math.min(P.k, data.length || 1));
  // Start the means with k-means++ seeding: spread out, which avoids most bad local maxima.
  const seeds = data.length ? initCentroids(data, k, 'k-means++') : [{ x: 0, y: 0 }];
  const comps: Component[] = seeds.map((c) => ({ pi: 1 / k, mu: [c.x, c.y], cov: [0.04, 0, 0.04] }));
  const state: GmmState = { comps, gamma: data.map(() => new Float64Array(k).fill(1 / k)), phase: 'E', iter: 0, logLik: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
