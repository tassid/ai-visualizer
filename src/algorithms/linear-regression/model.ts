/**
 * Regressão linear (e polinomial) por descida de gradiente.
 *
 *   ŷ = b + w₁·x + w₂·x² + ... + w_d·x^d
 *   MSE = (1/n) Σ (ŷ - y)²
 *   ∂MSE/∂wₖ = (2/n) Σ (ŷ - y) · x^k
 *
 * Com grau > 1 o modelo continua "linear" nos parâmetros: só trocamos x por
 * atributos polinomiais. Isso é o que torna o problema convexo.
 *
 * Existe também a solução fechada (equação normal) w = (XᵀX)⁻¹ Xᵀy,
 * implementada em `normalEquation` para comparação nos testes.
 */
import { randn } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface LinRegParams {
  deg: number;
  lr: number;
  iters: number;
}

export interface LinRegState {
  w: number[];
  b: number;
  it: number;
  loss: number;
  /** Caminho (w₁, b) percorrido, para a paisagem de perda (grau 1). */
  path: [number, number][];
  history: number[];
}

/** Avalia o polinômio pela regra de Horner seria mais rápido; aqui fica explícito para leitura. */
export function evaluate(s: Pick<LinRegState, 'w' | 'b'>, x: number): number {
  let v = s.b, power = 1;
  for (let k = 0; k < s.w.length; k++) {
    power *= x;
    v += s.w[k] * power;
  }
  return v;
}

export function mse(s: Pick<LinRegState, 'w' | 'b'>, data: Point[]): number {
  if (!data.length) return 0;
  let sum = 0;
  for (const p of data) sum += (evaluate(s, p.x) - p.y) ** 2;
  return sum / data.length;
}

export function* train(data: Point[], P: LinRegParams, s: LinRegState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'Sem dados.' };
  const d = s.w.length;

  for (let it = 1; it <= P.iters; it++) {                               // linha 1
    const g = new Array<number>(d).fill(0);
    let gb = 0;
    for (const p of data) {
      const e = evaluate(s, p.x) - p.y;                                 // linha 2: resíduo
      gb += (2 * e) / n;                                                // linha 4
      let power = 1;
      for (let k = 0; k < d; k++) {
        power *= p.x;
        g[k] += (2 * e * power) / n;                                    // linha 3
      }
    }
    for (let k = 0; k < d; k++) s.w[k] -= P.lr * g[k];                  // linha 5
    s.b -= P.lr * gb;                                                   // linha 6
    s.it = it;
    s.loss = mse(s, data);
    s.history.push(s.loss);
    s.path.push([s.w[0], s.b]);
    const gn = Math.hypot(gb, ...g);
    yield { line: 5, message: `Iteração ${it}: MSE = ${s.loss.toFixed(5)}, |∇| = ${gn.toFixed(5)}.` };
    if (gn < 1e-6) return { line: 7, message: `Gradiente nulo: mínimo encontrado na iteração ${it}.` };
  }
  return { line: 1, message: `Fim das ${P.iters} iterações.` };
}

export function createLinReg(data: Point[], P: LinRegParams): Trainer<LinRegState> {
  // Para grau 1 começamos num ponto fixo longe do mínimo: deixa o caminho na paisagem mais visível.
  const w = P.deg === 1 ? [-1.6] : Array.from({ length: P.deg }, () => randn() * 0.1);
  const b = P.deg === 1 ? 0.8 : 0;
  const state: LinRegState = { w, b, it: 0, loss: NaN, path: [[w[0], b]], history: [] };
  state.loss = mse(state, data);
  return { state, steps: train(data, P, state) };
}

/** Solução exata para grau 1 por mínimos quadrados: w = cov(x, y) / var(x). */
export function normalEquation(data: Point[]): { w: number; b: number } {
  const n = data.length;
  const mx = data.reduce((s, p) => s + p.x, 0) / n;
  const my = data.reduce((s, p) => s + p.y, 0) / n;
  let cov = 0, vx = 0;
  for (const p of data) { cov += (p.x - mx) * (p.y - my); vx += (p.x - mx) ** 2; }
  const w = cov / vx;
  return { w, b: my - w * mx };
}
