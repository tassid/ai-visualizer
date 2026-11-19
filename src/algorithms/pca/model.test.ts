import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createPca, projectedVariance } from './model';

describe('PCA', () => {
  beforeEach(() => seed(10));

  it('the angle sweep finds the same PC1 as the eigenvalues', () => {
    const data = DATASETS.elongated.generate();
    const t = createPca(data, { stepDeg: 0.5, projections: 'off' });
    runToEnd(t);
    const s = t.state;
    // angular difference modulo 180°
    const diff = Math.abs(((s.best.angle - s.pc1 + Math.PI * 1.5) % Math.PI) - Math.PI / 2);
    expect(diff).toBeLessThan((1 * Math.PI) / 180);
    expect(s.best.v).toBeCloseTo(s.l1, 3);
  });

  it('the projected variance stays between λ₂ and λ₁', () => {
    const t = createPca(DATASETS.blobs.generate(), { stepDeg: 1, projections: 'off' });
    const s = t.state;
    for (let th = 0; th < Math.PI; th += 0.1) {
      const v = projectedVariance(s.cov, th);
      expect(v).toBeLessThanOrEqual(s.l1 + 1e-12);
      expect(v).toBeGreaterThanOrEqual(s.l2 - 1e-12);
    }
  });

  it('λ₁ + λ₂ = trace of Σ', () => {
    const t = createPca(DATASETS.clusters.generate(), { stepDeg: 1, projections: 'off' });
    const [a, , c] = t.state.cov;
    expect(t.state.l1 + t.state.l2).toBeCloseTo(a + c, 10);
  });
});
