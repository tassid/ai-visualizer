/**
 * Training the MLP with mini-batch gradient descent.
 * Each animation step is one full epoch over the data.
 */
import { range, shuffle } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';
import { Network, type ActivationName } from './network';
import { createOptimizer, type OptimizerName } from './optimizers';

export interface MlpParams {
  arch: string;          // e.g. "8-8" = two hidden layers of 8 neurons
  act: ActivationName;
  opt: OptimizerName;
  lr: number;
  batch: string;         // "full" or a number
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
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  const net = s.net;
  const opt = createOptimizer(P.opt, net, P.lr);
  const batchSize = P.batch === 'full' ? n : Math.min(n, Number(P.batch));
  const order = range(n);

  for (let epoch = 1; epoch <= P.epochs; epoch++) {                     // line 1
    shuffle(order);
    let total = 0;
    for (let start = 0; start < n; start += batchSize) {                // line 2
      const batch = order.slice(start, start + batchSize);
      const grad = net.zerosLike();
      for (const i of batch) {                                          // lines 3 to 9
        const p = data[i];
        total += net.backward(p.x, p.y, p.c, grad);
      }
      // average gradient over the batch
      for (const m of grad.W) for (const r of m) for (let k = 0; k < r.length; k++) r[k] /= batch.length;
      for (const b of grad.B) for (let k = 0; k < b.length; k++) b[k] /= batch.length;
      opt.step(net, grad);                                              // line 10
    }
    s.epoch = epoch;
    s.loss = total / n;
    s.history.push(s.loss);
    yield { line: 10, message: `Epoch ${epoch}: mean loss ${s.loss.toFixed(4)}.` };
    if (s.loss < 1e-4) return { line: 11, message: `Loss below 0.0001 at epoch ${epoch}. The network separates the data.` };
  }
  return { line: 1, message: `Finished ${P.epochs} epochs.` };
}

export function createMlp(data: Point[], P: MlpParams): Trainer<MlpState> {
  const net = new Network([2, ...parseArch(P.arch), 1], P.act);
  const state: MlpState = { net, epoch: 0, loss: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
