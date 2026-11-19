import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { accuracy } from '../../core/math';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createMlp, type MlpParams } from './model';
import { Network } from './network';

const base: MlpParams = { arch: '8-8', act: 'tanh', opt: 'Adam', lr: 0.03, batch: 'full', epochs: 600 };

describe('MLP', () => {
  beforeEach(() => seed(3));

  it('learns XOR (which the perceptron cannot)', () => {
    const data = DATASETS.xor.generate();
    const t = createMlp(data, base);
    runToEnd(t);
    expect(accuracy(data, (x, y) => t.state.net.predict(x, y))).toBeGreaterThan(0.95);
  });

  it('the loss goes down during training', () => {
    const data = DATASETS.circles.generate();
    const t = createMlp(data, { ...base, epochs: 200 });
    runToEnd(t);
    const h = t.state.history;
    expect(h[h.length - 1]).toBeLessThan(h[0] / 2);
  });

  it('the backpropagation gradient matches the finite difference', () => {
    const net = new Network([2, 3, 1], 'tanh');
    const grad = net.zerosLike();
    const [x, y, target] = [0.3, -0.6, 1];
    net.backward(x, y, target, grad);
    const lossAt = () => {
      const p = net.predict(x, y);
      return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
    };
    const h = 1e-6;
    const w = net.params.W[0][1];
    const original = w[0];
    w[0] = original + h; const up = lossAt();
    w[0] = original - h; const down = lossAt();
    w[0] = original;
    expect(grad.W[0][1][0]).toBeCloseTo((up - down) / (2 * h), 5);
  });

  it.each(['SGD', 'Momentum', 'Adam'] as const)('optimizer %s lowers the loss', (opt) => {
    const data = DATASETS.blobs.generate();
    const t = createMlp(data, { ...base, opt, lr: opt === 'SGD' ? 0.3 : 0.03, epochs: 150 });
    runToEnd(t);
    const h = t.state.history;
    expect(h[h.length - 1]).toBeLessThan(h[0]);
  });
});
