import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { accuracy, covariance2 } from '../../core/math';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createNaiveBayes, posterior, welford } from './model';

describe('Naive Bayes gaussiano', () => {
  beforeEach(() => seed(9));

  it('Welford calcula a mesma média e variância que a fórmula direta', () => {
    const data = DATASETS.blobs.generate().filter((p) => p.c === 1);
    const st = { n: 0, mean: [0, 0] as [number, number], m2: [0, 0] as [number, number] };
    for (const p of data) welford(st, p.x, p.y);
    const cov = covariance2(data);
    expect(st.mean[0]).toBeCloseTo(cov.mx, 10);
    expect(st.m2[0] / st.n).toBeCloseTo(cov.xx, 10);
    expect(st.m2[1] / st.n).toBeCloseTo(cov.yy, 10);
  });

  it('classifica bem duas nuvens gaussianas', () => {
    const data = DATASETS.blobs.generate();
    const t = createNaiveBayes(data, { smoothing: 0.001 });
    runToEnd(t);
    expect(accuracy(data, (x, y) => posterior(t.state, x, y))).toBeGreaterThan(0.9);
  });

  it('a posteriori fica entre 0 e 1', () => {
    const t = createNaiveBayes(DATASETS.circles.generate(), { smoothing: 0.001 });
    runToEnd(t);
    for (const [x, y] of [[0, 0], [0.9, 0.9], [-0.5, 0.2]]) {
      const p = posterior(t.state, x, y);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
});
