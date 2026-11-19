/**
 * Análise de componentes principais (PCA) em 2D.
 *
 * O primeiro componente é a direção u (|u| = 1) que maximiza a variância
 * dos dados projetados:  Var(u) = uᵀ Σ u.
 * Pelo quociente de Rayleigh, o máximo é o maior autovalor de Σ, atingido
 * no autovetor correspondente.
 *
 * A animação faz isso "na força bruta": gira u de 0° a 180° e mede uᵀ Σ u.
 * No fim, compara com a solução exata por autovalores (core/math.ts: eigen2).
 */
import { covariance2, eigen2 } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface PcaParams {
  stepDeg: number;
  projections: 'sim' | 'não';
}

export interface PcaState {
  mx: number;
  my: number;
  cov: [number, number, number];
  l1: number;
  l2: number;
  /** Ângulo exato do PC1, em radianos, em [0, π). */
  pc1: number;
  angle: number;
  curve: [number, number][];
  best: { v: number; angle: number };
  finished: boolean;
}

/** Variância da projeção na direção de ângulo θ: uᵀ Σ u com u = (cos θ, sin θ). */
export function projectedVariance(cov: [number, number, number], theta: number): number {
  const [a, b, c] = cov, co = Math.cos(theta), si = Math.sin(theta);
  return a * co * co + 2 * b * co * si + c * si * si;
}

export function* train(_data: Point[], P: PcaParams, s: PcaState): Generator<StepInfo, StepInfo> {
  const step = (P.stepDeg * Math.PI) / 180;
  for (let theta = 0; theta < Math.PI - 1e-9; theta += step) {
    s.angle = theta;
    const v = projectedVariance(s.cov, theta);                          // linhas 3 e 4
    s.curve.push([theta, v]);
    if (v > s.best.v) s.best = { v, angle: theta };
    yield { line: 4, message: `θ = ${((theta * 180) / Math.PI).toFixed(0)}°: variância projetada ${v.toFixed(4)}${v === s.best.v ? ' (nova melhor)' : ''}.` };
  }
  s.finished = true;
  s.angle = s.pc1;
  const err = Math.abs(((s.best.angle - s.pc1 + Math.PI / 2 + Math.PI) % Math.PI) - Math.PI / 2) * (180 / Math.PI);
  return {
    line: 5,
    message: `PC1 exato em ${((s.pc1 * 180) / Math.PI).toFixed(1)}° (a busca achou ${((s.best.angle * 180) / Math.PI).toFixed(0)}°, erro ${err.toFixed(1)}°). Explica ${((s.l1 / (s.l1 + s.l2 || 1)) * 100).toFixed(1)}% da variância.`,
  };
}

export function createPca(data: Point[], P: PcaParams): Trainer<PcaState> {
  const c = covariance2(data);                                          // linhas 0 a 2
  const { l1, l2, angle } = eigen2(c.xx, c.xy, c.yy);
  const state: PcaState = {
    mx: c.mx, my: c.my, cov: [c.xx, c.xy, c.yy], l1, l2, pc1: angle,
    angle: 0, curve: [], best: { v: -1, angle: 0 }, finished: false,
  };
  return { state, steps: train(data, P, state) };
}
