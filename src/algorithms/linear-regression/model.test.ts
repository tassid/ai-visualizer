import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createLinReg, mse, normalEquation } from './model';

describe('linear regression', () => {
  beforeEach(() => seed(1));

  it('gradient descent reaches the normal equation solution', () => {
    const data = DATASETS.line.generate();
    const t = createLinReg(data, { deg: 1, lr: 0.3, iters: 3000 });
    runToEnd(t);
    const exact = normalEquation(data);
    expect(t.state.w[0]).toBeCloseTo(exact.w, 3);
    expect(t.state.b).toBeCloseTo(exact.b, 3);
  });

  it('the path records every iteration', () => {
    const data = DATASETS.line.generate();
    const t = createLinReg(data, { deg: 1, lr: 0.1, iters: 50 });
    runToEnd(t);
    expect(t.state.path).toHaveLength(51);
  });

  it('a higher degree fits the sine curve better', () => {
    const data = DATASETS.sine.generate();
    const linear = createLinReg(data, { deg: 1, lr: 0.3, iters: 2000 });
    const cubic = createLinReg(data, { deg: 3, lr: 0.6, iters: 2000 });
    runToEnd(linear);
    runToEnd(cubic);
    expect(mse(cubic.state, data)).toBeLessThan(mse(linear.state, data));
  });
});
