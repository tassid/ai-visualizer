import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createPca, projectedVariance } from './model';

describe('PCA', () => {
  beforeEach(() => seed(10));

  it('a busca por ângulo encontra o mesmo PC1 que os autovalores', () => {
    const data = DATASETS.elongated.generate();
    const t = createPca(data, { stepDeg: 0.5, projections: 'não' });
    runToEnd(t);
    const s = t.state;
    // diferença angular módulo 180°
    const diff = Math.abs(((s.best.angle - s.pc1 + Math.PI * 1.5) % Math.PI) - Math.PI / 2);
    expect(diff).toBeLessThan((1 * Math.PI) / 180);
    expect(s.best.v).toBeCloseTo(s.l1, 3);
  });

  it('a variância projetada fica entre λ₂ e λ₁', () => {
    const t = createPca(DATASETS.blobs.generate(), { stepDeg: 1, projections: 'não' });
    const s = t.state;
    for (let th = 0; th < Math.PI; th += 0.1) {
      const v = projectedVariance(s.cov, th);
      expect(v).toBeLessThanOrEqual(s.l1 + 1e-12);
      expect(v).toBeGreaterThanOrEqual(s.l2 - 1e-12);
    }
  });

  it('λ₁ + λ₂ = traço de Σ', () => {
    const t = createPca(DATASETS.clusters.generate(), { stepDeg: 1, projections: 'não' });
    const [a, , c] = t.state.cov;
    expect(t.state.l1 + t.state.l2).toBeCloseTo(a + c, 10);
  });
});
