/**
 * SVM linear treinada por subgradiente (estilo Pegasos, em lote completo).
 *
 * Objetivo (forma primal):
 *   min  λ/2 · |w|²  +  (1/n) Σ max(0, 1 - yᵢ(w·xᵢ + b))
 *        ─────────      ─────────────────────────────────
 *        margem larga   perda hinge: pune pontos dentro da margem
 *
 * A margem geométrica vale 2 / |w|. Diminuir |w| (regularização) alarga a
 * margem; a perda hinge empurra a reta para longe dos pontos.
 *
 * A perda hinge não é diferenciável em y·f = 1, por isso usamos um
 * subgradiente e uma taxa de aprendizado que decai com o tempo.
 */
import { uniform } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface SvmParams {
  lambda: number;
  lr: number;
  iters: number;
}

export interface SvmState {
  w: [number, number];
  b: number;
  it: number;
  loss: number;
  history: number[];
}

export const decision = (s: SvmState, x: number, y: number) => s.w[0] * x + s.w[1] * y + s.b;

/** Vetores de suporte: pontos sobre a margem ou dentro dela (y·f ≤ 1). */
export const isSupport = (s: SvmState, p: Point) => (p.c ? 1 : -1) * decision(s, p.x, p.y) <= 1 + 1e-4;

export const margin = (s: SvmState) => 2 / (Math.hypot(...s.w) || 1e-9);

export function* train(data: Point[], P: SvmParams, s: SvmState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'Sem dados.' };

  for (let it = 1; it <= P.iters; it++) {                               // linha 0
    let gw1 = P.lambda * s.w[0], gw2 = P.lambda * s.w[1], gb = 0;       // linha 1
    let loss = (P.lambda / 2) * (s.w[0] ** 2 + s.w[1] ** 2);
    let violators = 0;
    for (const p of data) {                                             // linha 2
      const y = p.c ? 1 : -1;
      const m = y * decision(s, p.x, p.y);
      if (m < 1) {                                                      // linha 3
        gw1 -= (y * p.x) / n;                                           // linha 4
        gw2 -= (y * p.y) / n;
        gb -= y / n;                                                    // linha 5
        loss += (1 - m) / n;
        violators++;
      }
    }
    const lr = P.lr / (1 + it * 0.003);                                 // decaimento da taxa
    s.w[0] -= lr * gw1;                                                 // linha 6
    s.w[1] -= lr * gw2;
    s.b -= lr * gb;
    s.it = it;
    s.loss = loss;
    s.history.push(loss);
    yield { line: 6, message: `Iteração ${it}: ${violators} ponto(s) violam a margem; margem = ${margin(s).toFixed(3)}.` };
  }
  return { line: 0, message: `Fim das ${P.iters} iterações.` };
}

export function createSvm(data: Point[], P: SvmParams): Trainer<SvmState> {
  const state: SvmState = { w: [uniform(-1, 1), uniform(-1, 1)], b: 0, it: 0, loss: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
