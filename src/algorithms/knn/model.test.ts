import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createKnn, leaveOneOut, neighbors } from './model';

describe('KNN', () => {
  beforeEach(() => seed(2));

  it('finds exactly the k nearest', () => {
    const data = [0, 0.1, 0.2, 0.5, 0.9].map((x) => ({ x, y: 0, c: 0 }));
    const nb = neighbors(data, 0.12, 0, 2);
    expect(nb.idx.sort()).toEqual([1, 2]);
    expect(nb.radius).toBeCloseTo(0.08, 10); // distance to the second neighbor (x = 0.2)
  });

  it('k = 1 with leave-one-out still does well on the moons', () => {
    const data = DATASETS.moons.generate();
    expect(leaveOneOut(data, 1, 'uniform')).toBeGreaterThan(0.9);
  });

  it('the sweep classifies every cell', () => {
    const data = DATASETS.blobs.generate();
    const t = createKnn(data, { k: 5, res: 20, weighting: 'uniform' });
    const { steps } = runToEnd(t);
    expect(steps).toBe(400);
    expect(Array.from(t.state.cells).every((v) => v >= 0 && v <= 1)).toBe(true);
  });
});
