import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { accuracy } from '../../core/math';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createLogistic, probability } from './model';

describe('logistic regression', () => {
  beforeEach(() => seed(11));

  it('the loss never goes up with a small rate (convex problem)', () => {
    const data = DATASETS.blobs.generate();
    const t = createLogistic(data, { lr: 0.1, iters: 200, l2: 0 });
    runToEnd(t);
    const h = t.state.history;
    for (let i = 1; i < h.length; i++) expect(h[i]).toBeLessThanOrEqual(h[i - 1] + 1e-12);
  });

  it('classifies two blobs well', () => {
    const data = DATASETS.blobs.generate();
    const t = createLogistic(data, { lr: 0.5, iters: 500, l2: 0 });
    runToEnd(t);
    expect(accuracy(data, (x, y) => probability(t.state, x, y))).toBeGreaterThan(0.9);
  });

  it('L2 regularization keeps the weights smaller', () => {
    const data = DATASETS.linear.generate();
    const a = createLogistic(data, { lr: 0.5, iters: 500, l2: 0 });
    const b = createLogistic(data, { lr: 0.5, iters: 500, l2: 0.1 });
    b.state.w = [...a.state.w];
    runToEnd(a);
    runToEnd(b);
    expect(Math.hypot(...b.state.w)).toBeLessThan(Math.hypot(...a.state.w));
  });
});
