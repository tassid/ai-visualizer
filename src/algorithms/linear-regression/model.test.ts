import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createLinReg, mse, normalEquation } from './model';

describe('regressão linear', () => {
  beforeEach(() => seed(1));

  it('a descida de gradiente chega à solução da equação normal', () => {
    const data = DATASETS.line.generate();
    const t = createLinReg(data, { deg: 1, lr: 0.3, iters: 3000 });
    runToEnd(t);
    const exact = normalEquation(data);
    expect(t.state.w[0]).toBeCloseTo(exact.w, 3);
    expect(t.state.b).toBeCloseTo(exact.b, 3);
  });

  it('o caminho registra cada iteração', () => {
    const data = DATASETS.line.generate();
    const t = createLinReg(data, { deg: 1, lr: 0.1, iters: 50 });
    runToEnd(t);
    expect(t.state.path).toHaveLength(51);
  });

  it('grau maior ajusta melhor a curva seno', () => {
    const data = DATASETS.sine.generate();
    const linear = createLinReg(data, { deg: 1, lr: 0.3, iters: 2000 });
    const cubic = createLinReg(data, { deg: 3, lr: 0.6, iters: 2000 });
    runToEnd(linear);
    runToEnd(cubic);
    expect(mse(cubic.state, data)).toBeLessThan(mse(linear.state, data));
  });
});
