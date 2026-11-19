/**
 * Funções matemáticas usadas pelos algoritmos.
 * Escritas à mão (sem bibliotecas) para que dê para ler cada fórmula.
 */
import type { Point } from './types';

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Função logística σ(z) = 1 / (1 + e^(-z)). Mapeia qualquer real em (0, 1). */
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Produto escalar a · b. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Distância euclidiana ao quadrado (evita a raiz quando só precisamos comparar). */
export const sqDist = (ax: number, ay: number, bx: number, by: number) => (ax - bx) ** 2 + (ay - by) ** 2;

export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/** Índice do maior elemento. */
export function argmax(xs: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}

/** n valores igualmente espaçados entre a e b (inclusive). */
export const linspace = (a: number, b: number, n: number) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / Math.max(1, n - 1));

/**
 * Entropia cruzada binária: -[y·log(p) + (1-y)·log(1-p)].
 * O epsilon evita log(0).
 */
export const binaryCrossEntropy = (p: number, y: number, eps = 1e-9) =>
  -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));

/** Acurácia de um preditor de probabilidade (classe 1 se p > 0,5). */
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

/** Média e matriz de covariância 2x2 de um conjunto de pontos. */
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
 * Autovalores e autovetor principal de uma matriz simétrica 2x2 [[a, b], [b, c]].
 *
 * Para 2x2 existe fórmula fechada:
 *   λ = (a + c)/2 ± sqrt(((a - c)/2)² + b²)
 * e o ângulo do autovetor principal é θ = ½ · atan2(2b, a - c).
 */
export function eigen2(a: number, b: number, c: number) {
  const tr = (a + c) / 2;
  const d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  let angle = 0.5 * Math.atan2(2 * b, a - c);
  if (angle < 0) angle += Math.PI;
  return { l1: tr + d, l2: Math.max(0, tr - d), angle };
}

/** Densidade da normal N(μ, σ²) em x. */
export const gaussianPdf = (x: number, mu: number, variance: number) =>
  Math.exp(-((x - mu) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);

export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : '...');
export const pct = (v: number) => (Number.isFinite(v) ? (v * 100).toFixed(1) + '%' : '...');
