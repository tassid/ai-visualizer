import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createForest } from './model';

describe('random forest', () => {
  beforeEach(() => seed(8));

  it('about 1/e of the points are left out of each bootstrap sample', () => {
    const data = DATASETS.moons.generate();
    const t = createForest(data, { trees: 60, depth: 4, maxFeatures: '1 at random' });
    runToEnd(t);
    const outFraction = t.state.inBag.map((b) => 1 - b.reduce((a, v) => a + v, 0) / data.length);
    const mean = outFraction.reduce((a, b) => a + b, 0) / outFraction.length;
    expect(mean).toBeGreaterThan(0.32);
    expect(mean).toBeLessThan(0.42);
  });

  it('reaches good out-of-bag accuracy on the moons', () => {
    const data = DATASETS.moons.generate();
    const t = createForest(data, { trees: 40, depth: 7, maxFeatures: '1 at random' });
    runToEnd(t);
    expect(t.state.oobAccuracy).toBeGreaterThan(0.9);
  });
});
