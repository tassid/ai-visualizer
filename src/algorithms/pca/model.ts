/**
 * Principal component analysis (PCA) in 2D.
 *
 * The first component is the direction u (|u| = 1) that maximizes the
 * variance of the projected data:  Var(u) = uᵀ Σ u.
 * By the Rayleigh quotient, the maximum is the largest eigenvalue of Σ,
 * reached at its eigenvector.
 *
 * The animation does it by brute force: it turns u from 0° to 180° and
 * measures uᵀ Σ u. At the end it compares with the exact eigen solution
 * (core/math.ts: eigen2).
 */
import { covariance2, eigen2 } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface PcaParams {
  stepDeg: number;
  projections: 'on' | 'off';
}

export interface PcaState {
  mx: number;
  my: number;
  cov: [number, number, number];
  l1: number;
  l2: number;
  /** Exact angle of PC1, in radians, in [0, π). */
  pc1: number;
  angle: number;
  curve: [number, number][];
  best: { v: number; angle: number };
  finished: boolean;
}

/** Variance of the projection onto the direction at angle θ: uᵀ Σ u with u = (cos θ, sin θ). */
export function projectedVariance(cov: [number, number, number], theta: number): number {
  const [a, b, c] = cov, co = Math.cos(theta), si = Math.sin(theta);
  return a * co * co + 2 * b * co * si + c * si * si;
}

export function* train(_data: Point[], P: PcaParams, s: PcaState): Generator<StepInfo, StepInfo> {
  const step = (P.stepDeg * Math.PI) / 180;
  for (let theta = 0; theta < Math.PI - 1e-9; theta += step) {
    s.angle = theta;
    const v = projectedVariance(s.cov, theta);                          // lines 3 and 4
    s.curve.push([theta, v]);
    if (v > s.best.v) s.best = { v, angle: theta };
    yield { line: 4, message: `θ = ${((theta * 180) / Math.PI).toFixed(0)}°: projected variance ${v.toFixed(4)}${v === s.best.v ? ', the largest so far' : ''}.` };
  }
  s.finished = true;
  s.angle = s.pc1;
  return {
    line: 5,
    message: `The exact PC1 points at ${((s.pc1 * 180) / Math.PI).toFixed(1)}°; the sweep found ${((s.best.angle * 180) / Math.PI).toFixed(0)}°. It explains ${((s.l1 / (s.l1 + s.l2 || 1)) * 100).toFixed(1)}% of the variance.`,
  };
}

export function createPca(data: Point[], P: PcaParams): Trainer<PcaState> {
  const c = covariance2(data);                                          // lines 0 to 1
  const { l1, l2, angle } = eigen2(c.xx, c.xy, c.yy);
  const state: PcaState = {
    mx: c.mx, my: c.my, cov: [c.xx, c.xy, c.yy], l1, l2, pc1: angle,
    angle: 0, curve: [], best: { v: -1, angle: 0 }, finished: false,
  };
  return { state, steps: train(data, P, state) };
}
