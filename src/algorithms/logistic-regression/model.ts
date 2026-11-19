/**
 * Regressão logística (classificação binária).
 *
 *   p(y = 1 | x) = σ(w · x + b)
 *   perda = média da entropia cruzada
 *   ∂perda/∂w = média de (p - y) · x       ← a mesma forma da regressão linear!
 *
 * A função de perda é convexa, então a descida de gradiente sempre chega ao
 * mínimo global (se a taxa de aprendizado não for grande demais). Em dados
 * perfeitamente separáveis, porém, os pesos crescem sem limite: o modelo fica
 * cada vez mais "confiante". Isso aparece como uma fronteira cada vez mais nítida.
 */
import { binaryCrossEntropy, sigmoid } from '../../core/math';
import { uniform } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface LogisticParams {
  lr: number;
  iters: number;
  l2: number;
}

export interface LogisticState {
  w: [number, number];
  b: number;
  it: number;
  loss: number;
  gradNorm: number;
  /** Direção de descida (-gradiente) normalizada, para desenhar. */
  descent: [number, number];
  history: number[];
}

export const probability = (s: LogisticState, x: number, y: number) => sigmoid(s.w[0] * x + s.w[1] * y + s.b);

export function* train(data: Point[], P: LogisticParams, s: LogisticState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'Sem dados.' };

  for (let it = 1; it <= P.iters; it++) {                               // linha 1
    let g1 = 0, g2 = 0, gb = 0, loss = 0;
    for (const p of data) {
      const pr = probability(s, p.x, p.y);                              // linha 2
      const e = pr - p.c;
      g1 += (e * p.x) / n;                                              // linha 3
      g2 += (e * p.y) / n;
      gb += e / n;                                                      // linha 4
      loss += binaryCrossEntropy(pr, p.c) / n;
    }
    // regularização L2 (não se aplica ao viés)
    g1 += P.l2 * s.w[0];
    g2 += P.l2 * s.w[1];
    loss += (P.l2 / 2) * (s.w[0] ** 2 + s.w[1] ** 2);

    s.w[0] -= P.lr * g1;                                                // linha 5
    s.w[1] -= P.lr * g2;
    s.b -= P.lr * gb;
    s.it = it;
    s.loss = loss;
    s.gradNorm = Math.hypot(g1, g2, gb);
    const gn = Math.hypot(g1, g2) || 1;
    s.descent = [-g1 / gn, -g2 / gn];
    s.history.push(loss);
    yield { line: 5, message: `Iteração ${it}: perda ${loss.toFixed(4)}, |∇| = ${s.gradNorm.toFixed(4)}.` };
    if (s.gradNorm < 1e-4) return { line: 6, message: `Gradiente quase zero: chegou ao mínimo na iteração ${it}.` };
  }
  return { line: 1, message: `Fim das ${P.iters} iterações.` };
}

export function createLogistic(data: Point[], P: LogisticParams): Trainer<LogisticState> {
  const state: LogisticState = { w: [uniform(-1, 1), uniform(-1, 1)], b: 0, it: 0, loss: NaN, gradNorm: NaN, descent: [0, 0], history: [] };
  return { state, steps: train(data, P, state) };
}
