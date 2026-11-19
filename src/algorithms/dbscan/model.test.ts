import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createDbscan, NOISE, UNSEEN } from './model';

describe('DBSCAN', () => {
  beforeEach(() => seed(12));

  it('separa as duas luas', () => {
    const data = DATASETS.moons.generate();
    const t = createDbscan(data, { eps: 0.15, minPts: 4 });
    runToEnd(t);
    expect(t.state.clusters).toBe(2);
    // cada lua (classe original) fica num único grupo
    for (const c of [0, 1]) {
      const labels = new Set(data.map((p, i) => (p.c === c ? t.state.label[i] : null)).filter((l) => l !== null && l !== NOISE));
      expect(labels.size).toBe(1);
    }
  });

  it('marca pontos isolados como ruído', () => {
    const data = [
      ...Array.from({ length: 10 }, (_, i) => ({ x: i * 0.01, y: 0, c: 0 })),
      { x: 0.9, y: 0.9, c: 0 },
    ];
    const t = createDbscan(data, { eps: 0.05, minPts: 3 });
    runToEnd(t);
    expect(t.state.label[10]).toBe(NOISE);
    expect(t.state.clusters).toBe(1);
  });

  it('todos os pontos terminam com rótulo', () => {
    const data = DATASETS.circles.generate();
    const t = createDbscan(data, { eps: 0.12, minPts: 4 });
    runToEnd(t);
    expect(Array.from(t.state.label).includes(UNSEEN)).toBe(false);
  });
});
