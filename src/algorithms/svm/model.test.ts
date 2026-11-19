import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createSvm, decision, isSupport, margin } from './model';

describe('linear SVM', () => {
  beforeEach(() => seed(5));

  it('separates linearly separable data', () => {
    const data = DATASETS.linear.generate();
    const t = createSvm(data, { lambda: 0.001, lr: 0.5, iters: 1500 });
    runToEnd(t);
    const errors = data.filter((p) => (decision(t.state, p.x, p.y) >= 0 ? 1 : 0) !== p.c).length;
    expect(errors / data.length).toBeLessThan(0.03);
  });

  it('a larger λ gives a wider margin', () => {
    const data = DATASETS.blobs.generate();
    const small = createSvm(data, { lambda: 0.001, lr: 0.5, iters: 800 });
    const big = createSvm(data, { lambda: 0.5, lr: 0.5, iters: 800 });
    runToEnd(small);
    runToEnd(big);
    expect(margin(big.state)).toBeGreaterThan(margin(small.state));
  });

  it('only some points are support vectors', () => {
    const data = DATASETS.linear.generate();
    const t = createSvm(data, { lambda: 0.01, lr: 0.5, iters: 800 });
    runToEnd(t);
    const sv = data.filter((p) => isSupport(t.state, p)).length;
    expect(sv).toBeGreaterThan(0);
    expect(sv).toBeLessThan(data.length / 2);
  });
});
