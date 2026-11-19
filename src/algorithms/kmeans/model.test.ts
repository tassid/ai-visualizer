import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createKMeans } from './model';

describe('K-Means', () => {
  beforeEach(() => seed(6));

  it('a inércia nunca aumenta (atribuição e atualização são monotônicas)', () => {
    const data = DATASETS.clusters.generate();
    const t = createKMeans(data, { k: 4, init: 'aleatória', animateAssign: 'não' });
    runToEnd(t);
    const h = t.state.history;
    for (let i = 1; i < h.length; i++) expect(h[i]).toBeLessThanOrEqual(h[i - 1] + 1e-12);
  });

  it('converge e termina', () => {
    const data = DATASETS.clusters.generate();
    const t = createKMeans(data, { k: 4, init: 'k-means++', animateAssign: 'não' });
    const { last } = runToEnd(t, 10_000);
    expect(last?.message).toMatch(/Convergiu/);
  });

  it('com k-means++ encontra os 4 grupos reais', () => {
    const data = DATASETS.clusters.generate();
    const t = createKMeans(data, { k: 4, init: 'k-means++', animateAssign: 'não' });
    runToEnd(t);
    // pontos gerados com o mesmo centro (i % 4) devem cair majoritariamente no mesmo grupo
    for (let g = 0; g < 4; g++) {
      const labels = data.map((_, i) => i).filter((i) => i % 4 === g).map((i) => t.state.assign[i]);
      const counts = new Map<number, number>();
      labels.forEach((l) => counts.set(l, (counts.get(l) ?? 0) + 1));
      expect(Math.max(...counts.values()) / labels.length).toBeGreaterThan(0.85);
    }
  });
});
