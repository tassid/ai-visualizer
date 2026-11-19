/**
 * Treino da MLP com descida de gradiente em lotes (mini-batch).
 * Cada passo da animação é uma época inteira sobre os dados.
 */
import { range, shuffle } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';
import { Network, type ActivationName } from './network';
import { createOptimizer, type OptimizerName } from './optimizers';

export interface MlpParams {
  arch: string;          // ex.: "8-8" = duas camadas ocultas de 8 neurônios
  act: ActivationName;
  opt: OptimizerName;
  lr: number;
  batch: string;         // "completo" ou um número
  epochs: number;
}

export interface MlpState {
  net: Network;
  epoch: number;
  loss: number;
  history: number[];
}

export const parseArch = (arch: string) => String(arch).split('-').map(Number).filter((n) => n > 0);

export function* train(data: Point[], P: MlpParams, s: MlpState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'Sem dados.' };
  const net = s.net;
  const opt = createOptimizer(P.opt, net, P.lr);
  const batchSize = P.batch === 'completo' ? n : Math.min(n, Number(P.batch));
  const order = range(n);

  for (let epoch = 1; epoch <= P.epochs; epoch++) {                     // linha 1
    shuffle(order);
    let total = 0;
    for (let start = 0; start < n; start += batchSize) {                // linha 2
      const batch = order.slice(start, start + batchSize);
      const grad = net.zerosLike();
      for (const i of batch) {                                          // linhas 3 a 9
        const p = data[i];
        total += net.backward(p.x, p.y, p.c, grad);
      }
      // média do gradiente no lote
      for (const m of grad.W) for (const r of m) for (let k = 0; k < r.length; k++) r[k] /= batch.length;
      for (const b of grad.B) for (let k = 0; k < b.length; k++) b[k] /= batch.length;
      opt.step(net, grad);                                              // linha 10
    }
    s.epoch = epoch;
    s.loss = total / n;
    s.history.push(s.loss);
    yield { line: 10, message: `Época ${epoch}: perda média ${s.loss.toFixed(4)}.` };
    if (s.loss < 1e-4) return { line: 11, message: `Perda abaixo de 0,0001 na época ${epoch}. A rede separou os dados.` };
  }
  return { line: 1, message: `Treino concluído após ${P.epochs} épocas.` };
}

export function createMlp(data: Point[], P: MlpParams): Trainer<MlpState> {
  const net = new Network([2, ...parseArch(P.arch), 1], P.act);
  const state: MlpState = { net, epoch: 0, loss: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
