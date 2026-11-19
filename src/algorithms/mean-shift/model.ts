/**
 * Mean shift (Fukunaga and Hostetler, 1975; Comaniciu and Meer, 2002).
 *
 * Every point climbs the kernel density estimate until it reaches a peak:
 *   m(x) = Σⱼ K(xⱼ - x) xⱼ / Σⱼ K(xⱼ - x),   K(u) = exp(-|u|² / (2h²))
 * Moving x to m(x) is a step uphill along the gradient of the density, with
 * an automatic step size. Points that end at the same peak form a cluster,
 * so the number of clusters is not chosen in advance; the bandwidth h is.
 */
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface MsParams {
  bandwidth: number;
  iters: number;
}

export interface MsState {
  /** Current position of each point's climber. */
  modes: { x: number; y: number }[];
  iter: number;
  maxShift: number;
  /** Cluster of each point once converged; -1 before. */
  labels: Int32Array;
  peaks: { x: number; y: number }[];
}

export const kernel = (dx: number, dy: number, h: number) => Math.exp(-(dx * dx + dy * dy) / (2 * h * h));

/** Kernel density estimate (unnormalized) at (x, y). */
export function density(data: Point[], x: number, y: number, h: number): number {
  let s = 0;
  for (const p of data) s += kernel(p.x - x, p.y - y, h);
  return s;
}

export function shift(data: Point[], x: number, y: number, h: number): { x: number; y: number } {
  let sx = 0, sy = 0, sw = 0;
  for (const p of data) {
    const w = kernel(p.x - x, p.y - y, h);
    sx += w * p.x; sy += w * p.y; sw += w;
  }
  return sw > 0 ? { x: sx / sw, y: sy / sw } : { x, y };
}

export function* train(data: Point[], P: MsParams, s: MsState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'No data. Click the plane to add points.' };
  for (let it = 1; it <= P.iters; it++) {                               // line 1
    let maxShift = 0;
    s.modes = s.modes.map((m) => {                                      // lines 2 and 3
      const next = shift(data, m.x, m.y, P.bandwidth);
      maxShift = Math.max(maxShift, Math.hypot(next.x - m.x, next.y - m.y));
      return next;
    });
    s.iter = it;
    s.maxShift = maxShift;
    yield { line: 3, message: `Iteration ${it}: every climber moves to the weighted mean of its neighborhood. Largest move ${maxShift.toFixed(4)}.` };
    if (maxShift < 1e-4) break;                                         // line 4
  }
  // Group climbers that ended at the same peak.
  const tol = P.bandwidth / 2;
  s.modes.forEach((m, i) => {                                           // line 5
    let k = s.peaks.findIndex((p) => Math.hypot(p.x - m.x, p.y - m.y) < tol);
    if (k < 0) { s.peaks.push({ ...m }); k = s.peaks.length - 1; }
    s.labels[i] = k;
  });
  return { line: 5, message: `Converged after ${s.iter} iterations onto ${s.peaks.length} peak(s) of the density.` };
}

export function createMs(data: Point[], P: MsParams): Trainer<MsState> {
  const state: MsState = { modes: data.map((p) => ({ x: p.x, y: p.y })), iter: 0, maxShift: NaN, labels: new Int32Array(data.length).fill(-1), peaks: [] };
  return { state, steps: train(data, P, state) };
}
