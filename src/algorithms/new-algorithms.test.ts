/**
 * Tests for the algorithms added in the second round. Each checks a property
 * that follows from the math, as in the per-folder tests.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../core/datasets';
import { accuracy } from '../core/math';
import { seed } from '../core/random';
import { runToEnd } from '../engine/useTrainer';
import { createAda, score as adaScore } from './adaboost/model';
import { createDa, posterior as daPosterior } from './discriminant-analysis/model';
import { createGp, XS } from './gaussian-process/model';
import { createGmm } from './gmm/model';
import { createGb, scoreAt } from './gradient-boosting/model';
import { createHc } from './hierarchical/model';
import { createKsvm, decision } from './kernel-svm/model';
import { createMs } from './mean-shift/model';
import { createOpt, SURFACES } from './optimizers/model';
import { createQ, GOAL, maxQ, START } from './q-learning/model';
import { createSom, quantizationError } from './som/model';
import { cholesky } from '../core/linalg';

beforeEach(() => seed(21));

describe('discriminant analysis', () => {
  it('QDA separates the circles, LDA cannot', () => {
    const data = DATASETS.circles.generate();
    const q = createDa(data, { model: 'QDA' });
    const l = createDa(data, { model: 'LDA' });
    runToEnd(q); runToEnd(l);
    expect(accuracy(data, (x, y) => daPosterior(q.state, x, y))).toBeGreaterThan(0.95);
    expect(accuracy(data, (x, y) => daPosterior(l.state, x, y))).toBeLessThan(0.7);
  });
});

describe('kernel SVM', () => {
  it('an RBF kernel solves the circles', () => {
    const data = DATASETS.circles.generate();
    const t = createKsvm(data, { gamma: 10, lambda: 0.01, iters: 3000 });
    runToEnd(t);
    expect(accuracy(data, (x, y) => (decision(t.state, data, x, y) >= 0 ? 1 : 0))).toBeGreaterThan(0.95);
  });
});

describe('AdaBoost', () => {
  it('drives the training error down on the moons', () => {
    const data = DATASETS.moons.generate();
    const t = createAda(data, { rounds: 150 });
    runToEnd(t);
    expect(accuracy(data, (x, y) => (adaScore(t.state, x, y) >= 0 ? 1 : 0))).toBeGreaterThan(0.95);
    expect(t.state.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });
});

describe('gradient boosting', () => {
  it('the log loss decreases round after round', () => {
    const data = DATASETS.xor.generate();
    const t = createGb(data, { rounds: 60, depth: 2, lr: 0.3 });
    runToEnd(t);
    const h = t.state.history;
    expect(h[h.length - 1]).toBeLessThan(h[0] / 2);
    expect(accuracy(data, (x, y) => 1 / (1 + Math.exp(-scoreAt(t.state, 0.3, x, y))))).toBeGreaterThan(0.95);
  });
});

describe('Gaussian process', () => {
  it('Cholesky reconstructs the matrix', () => {
    const A = [[4, 2, 0.4], [2, 5, 1], [0.4, 1, 3]];
    const L = cholesky(A);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += L[i][k] * L[j][k];
      expect(s).toBeCloseTo(A[i][j], 10);
    }
  });
  it('with little noise the mean passes near each observation and the band is narrow there', () => {
    const data = DATASETS.few.generate();
    const t = createGp(data, { lengthscale: 0.2, noise: 0.01, signal: 0.25 });
    runToEnd(t);
    for (const p of data) {
      const i = Math.round(((p.x + 1) / 2) * (XS.length - 1));
      if (Math.abs(XS[i] - p.x) > 0.01) continue;
      expect(Math.abs(t.state.mean[i] - p.y)).toBeLessThan(0.08);
      expect(t.state.std[i]).toBeLessThan(0.1);
    }
  });
});

describe('Gaussian mixture', () => {
  it('EM never decreases the log-likelihood', () => {
    const data = DATASETS.clusters.generate();
    const t = createGmm(data, { k: 4, iters: 60 });
    runToEnd(t);
    const h = t.state.history;
    for (let i = 1; i < h.length; i++) expect(h[i]).toBeGreaterThanOrEqual(h[i - 1] - 1e-6);
  });
});

describe('hierarchical clustering', () => {
  it('makes n - 1 merges with non-decreasing heights (Ward)', () => {
    const data = DATASETS.clusters.generate().slice(0, 80);
    const t = createHc(data, { linkage: 'ward', k: 4 });
    runToEnd(t);
    expect(t.state.merges).toHaveLength(79);
    for (let i = 1; i < 79; i++) expect(t.state.merges[i].height).toBeGreaterThanOrEqual(t.state.merges[i - 1].height - 1e-9);
    expect(new Set(t.state.cut).size).toBe(4);
  });
  it('single linkage separates the two moons', () => {
    const data = DATASETS.moons.generate();
    const t = createHc(data, { linkage: 'single', k: 2 });
    runToEnd(t);
    const agree = data.filter((p, i) => t.state.cut![i] === t.state.cut![0] ? p.c === data[0].c : p.c !== data[0].c).length;
    expect(agree / data.length).toBeGreaterThan(0.95);
  });
});

describe('mean shift', () => {
  it('finds the four clusters with a suitable bandwidth', () => {
    const data = DATASETS.clusters.generate();
    const t = createMs(data, { bandwidth: 0.15, iters: 200 });
    runToEnd(t);
    expect(t.state.peaks.length).toBe(4);
  });
});

describe('self-organizing map', () => {
  it('training lowers the quantization error', () => {
    const data = DATASETS.moons.generate();
    const t = createSom(data, { size: 8, steps: 3000, lr: 0.5 });
    const before = quantizationError(t.state, data);
    runToEnd(t);
    expect(quantizationError(t.state, data)).toBeLessThan(before / 3);
  });
});

describe('Q-learning', () => {
  it('learns a positive value at the start and reaches the goal greedily', () => {
    const t = createQ([], { alpha: 0.5, gamma: 0.95, epsilon: 0.1, episodes: 400 });
    runToEnd(t, 1_000_000);
    expect(maxQ(t.state.Q, START)).toBeGreaterThan(0);
    // follow the greedy policy from the start
    let s = START;
    const { greedyPath } = { greedyPath: [] as number[] };
    for (let k = 0; k < 64 && s !== GOAL; k++) {
      let a = 0;
      for (let i = 1; i < 4; i++) if (t.state.Q[s * 4 + i] > t.state.Q[s * 4 + a]) a = i;
      greedyPath.push(s);
      const r = Math.floor(s / 8) + [-1, 0, 1, 0][a], c = (s % 8) + [0, 1, 0, -1][a];
      if (r >= 0 && r < 8 && c >= 0 && c < 8 && !t.state.walls[r * 8 + c]) s = r * 8 + c;
    }
    expect(s).toBe(GOAL);
  });
});

describe('optimizers', () => {
  it('all four get close to the minimum of the narrow valley', () => {
    // Adam and RMSProp keep steps of about η near the minimum, so they hover within ~η of it.
    const t = createOpt([], { surface: 'narrow valley', lr: 0.03, iters: 1500 });
    runToEnd(t);
    for (const r of t.state.runners) expect(SURFACES['narrow valley'].f(r.x, r.y)).toBeLessThan(5e-3);
  });
  it('SGD diverges on the narrow valley when η > 2 / 12', () => {
    const t = createOpt([], { surface: 'narrow valley', lr: 0.2, iters: 50 });
    runToEnd(t);
    const sgd = t.state.runners.find((r) => r.name === 'SGD')!;
    expect(Math.abs(sgd.y)).toBeGreaterThan(1);
  });
});
