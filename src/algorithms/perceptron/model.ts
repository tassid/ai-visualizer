/**
 * Perceptron de Rosenblatt (1958).
 *
 * Modelo: f(x) = w · x + b ; classe = sinal(f(x))
 * Regra de aprendizado: só aprende quando erra.
 *   se y · f(x) ≤ 0:  w ← w + η·y·x ;  b ← b + η·y
 *
 * Teorema de convergência (Novikoff, 1962): se existe uma reta que separa os
 * dados com margem γ e todos os pontos têm norma ≤ R, o perceptron faz no
 * máximo (R/γ)² atualizações. Se os dados NÃO forem separáveis (XOR), ele
 * fica corrigindo para sempre.
 */
import { range, shuffle, uniform } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface PerceptronParams {
  lr: number;
  epochs: number;
}

export interface PerceptronState {
  w: [number, number];
  b: number;
  epoch: number;
  /** Erros na época em andamento. */
  errors: number;
  /** Índice do ponto sendo examinado. */
  current: number;
  /** O último passo atualizou os pesos? */
  updated: boolean;
  /** Posição dentro da época. */
  seen: number;
  /** Erros ao final de cada época. */
  history: number[];
}

/** Valor bruto f(x) = w · x + b. */
export const score = (s: PerceptronState, x: number, y: number) => s.w[0] * x + s.w[1] * y + s.b;

export const predict = (s: PerceptronState, x: number, y: number) => (score(s, x, y) >= 0 ? 1 : 0);

export function initialState(): PerceptronState {
  return { w: [uniform(-0.3, 0.3), uniform(-0.3, 0.3)], b: 0, epoch: 0, errors: 0, current: -1, updated: false, seen: 0, history: [] };
}

/**
 * O algoritmo, linha a linha. Cada `yield` vira um passo da animação, e o
 * número `line` aponta para a linha do pseudocódigo (ver index.ts).
 */
export function* train(data: Point[], P: PerceptronParams, s: PerceptronState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'Sem dados.' };
  const order = range(data.length);

  for (let epoch = 0; epoch < P.epochs; epoch++) {                      // linha 1
    s.epoch = epoch;
    s.errors = 0;                                                       // linha 2
    shuffle(order);                                                     // linha 3
    for (let k = 0; k < order.length; k++) {
      const i = order[k];
      const p = data[i];
      s.current = i;
      s.seen = k + 1;
      const y = p.c ? 1 : -1;                                           // rótulos em {-1, +1}
      const f = score(s, p.x, p.y);                                     // linha 4

      if (y * f <= 0) {                                                 // linha 5: errou (ou está na fronteira)
        s.w[0] += P.lr * y * p.x;                                       // linha 6
        s.w[1] += P.lr * y * p.y;
        s.b += P.lr * y;                                                // linha 7
        s.errors++;                                                     // linha 8
        s.updated = true;
        yield { line: 6, message: `Errou: y·f = ${(y * f).toFixed(2)} ≤ 0, então w e b andam ${y > 0 ? 'a favor' : 'contra'} o ponto.` };
      } else {
        s.updated = false;
        yield { line: 4, message: `Acertou: y·f = ${(y * f).toFixed(2)} > 0, nada muda.` };
      }
    }
    s.history.push(s.errors);
    if (s.errors === 0) return { line: 9, message: `Convergiu na época ${epoch + 1}: nenhum erro, a reta separa os dados.` };
    yield { line: 9, message: `Fim da época ${epoch + 1}: ${s.errors} erro(s). Começa outra época.` };
  }
  return { line: 1, message: `Limite de ${P.epochs} épocas atingido sem convergir. Os dados podem não ser linearmente separáveis.` };
}

export function createPerceptron(data: Point[], P: PerceptronParams): Trainer<PerceptronState> {
  const state = initialState();
  return { state, steps: train(data, P, state) };
}
