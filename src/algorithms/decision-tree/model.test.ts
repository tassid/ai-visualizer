import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { accuracy } from '../../core/math';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { bestSplit, gini, predictTree, treeDepth } from '../shared/tree';
import { createTree } from './model';

describe('decision tree', () => {
  beforeEach(() => seed(4));

  it('Gini is 0 on pure nodes and at most 0.5', () => {
    expect(gini(0)).toBe(0);
    expect(gini(1)).toBe(0);
    expect(gini(0.5)).toBe(0.5);
  });

  it('finds the perfect split on data separable along one axis', () => {
    const data = [-0.8, -0.5, -0.2, 0.3, 0.6, 0.9].map((x) => ({ x, y: 0, c: x > 0 ? 1 : 0 }));
    const s = bestSplit(data, data.map((_, i) => i), [0, 1])!;
    expect(s.feature).toBe(0);
    expect(s.threshold).toBeCloseTo(0.05, 10);
    expect(s.gini).toBe(0);
  });

  it('with no depth limit, it memorizes the training set', () => {
    const data = DATASETS.xor.generate();
    const t = createTree(data, { depth: 20, minSplit: 2 });
    runToEnd(t);
    expect(accuracy(data, (x, y) => predictTree(t.state.root, x, y))).toBe(1);
  });

  it('respects the maximum depth', () => {
    const data = DATASETS.spiral.generate();
    const t = createTree(data, { depth: 3, minSplit: 2 });
    runToEnd(t);
    expect(treeDepth(t.state.root)).toBeLessThanOrEqual(3);
  });
});
