/**
 * Math helpers used by the algorithms. Written by hand, without libraries,
 * so every formula can be read.
 */
import type { Point } from './types';

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Logistic function σ(z) = 1 / (1 + e^(-z)). Maps any real number into (0, 1). */
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Dot product a · b. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Squared Euclidean distance (skips the square root when we only compare). */
export const sqDist = (ax: number, ay: number, bx: number, by: number) => (ax - bx) ** 2 + (ay - by) ** 2;

export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

export function argmax(xs: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}

/** n evenly spaced values from a to b, inclusive. */
export const linspace = (a: number, b: number, n: number) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / Math.max(1, n - 1));

/** Binary cross-entropy: -[y·log(p) + (1-y)·log(1-p)]. The epsilon avoids log(0). */
export const binaryCrossEntropy = (p: number, y: number, eps = 1e-9) =>
  -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));

/** Accuracy of a probability predictor (class 1 when p > 0.5). */
export function accuracy(data: Point[], predict: (x: number, y: number) => number): number {
  if (!data.length) return 0;
  let ok = 0;
  for (const p of data) if ((predict(p.x, p.y) > 0.5 ? 1 : 0) === p.c) ok++;
  return ok / data.length;
}

export interface Covariance2 {
  mx: number;
  my: number;
  xx: number;
  xy: number;
  yy: number;
}

/** Mean and 2x2 covariance matrix of a set of points. */
export function covariance2(points: { x: number; y: number }[]): Covariance2 {
  const n = points.length || 1;
  let mx = 0, my = 0;
  for (const p of points) { mx += p.x; my += p.y; }
  mx /= n; my /= n;
  let xx = 0, xy = 0, yy = 0;
  for (const p of points) {
    const dx = p.x - mx, dy = p.y - my;
    xx += dx * dx; xy += dx * dy; yy += dy * dy;
  }
  return { mx, my, xx: xx / n, xy: xy / n, yy: yy / n };
}

/**
 * Eigenvalues and principal eigenvector of a symmetric 2x2 matrix [[a, b], [b, c]].
 *   λ = (a + c)/2 ± sqrt(((a - c)/2)² + b²)
 *   angle of the principal eigenvector: θ = ½ · atan2(2b, a - c)
 */
export function eigen2(a: number, b: number, c: number) {
  const tr = (a + c) / 2;
  const d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  let angle = 0.5 * Math.atan2(2 * b, a - c);
  if (angle < 0) angle += Math.PI;
  return { l1: tr + d, l2: Math.max(0, tr - d), angle };
}

/** Density of N(μ, σ²) at x. */
export const gaussianPdf = (x: number, mu: number, variance: number) =>
  Math.exp(-((x - mu) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);

export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : 'n/a');
export const pct = (v: number) => (Number.isFinite(v) ? (v * 100).toFixed(1) + '%' : 'n/a');

/** Number formatted for TeX, with a real minus sign. */
export const tex = (v: number, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : '?');

/** A signed term for TeX, e.g. "+ 0.33 x_1" or "- 0.26 x_2". */
export const texTerm = (v: number, symbol: string, digits = 2, color?: string) => {
  const num = Math.abs(v).toFixed(digits);
  const body = color ? `{\\color{${color}}${num}}` : num;
  return `${v < 0 ? '-' : '+'} ${body}\\,${symbol}`;
};
