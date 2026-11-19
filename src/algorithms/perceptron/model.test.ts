import { beforeEach, describe, expect, it } from 'vitest';
import { DATASETS } from '../../core/datasets';
import { accuracy } from '../../core/math';
import { seed } from '../../core/random';
import { runToEnd } from '../../engine/useTrainer';
import { createPerceptron, predict } from './model';

describe('perceptron', () => {
  beforeEach(() => seed(7));

  it('converge em dados linearmente separáveis', () => {
    const data = DATASETS.linear.generate();
    const t = createPerceptron(data, { lr: 0.1, epochs: 100 });
    const { last } = runToEnd(t);
    expect(last?.message).toMatch(/Convergiu/);
    expect(accuracy(data, (x, y) => predict(t.state, x, y))).toBe(1);
  });

  it('não converge no XOR', () => {
    const data = DATASETS.xor.generate();
    const t = createPerceptron(data, { lr: 0.1, epochs: 20 });
    const { last } = runToEnd(t);
    expect(last?.message).toMatch(/Limite/);
    expect(t.state.history).toHaveLength(20);
    expect(Math.min(...t.state.history)).toBeGreaterThan(0);
  });

  it('só altera os pesos quando erra', () => {
    const data = [{ x: 0.5, y: 0.5, c: 1 }];
    const t = createPerceptron(data, { lr: 1, epochs: 5 });
    t.state.w = [1, 1];
    t.state.b = 0;
    t.steps.next();
    expect(t.state.updated).toBe(false);
    expect(t.state.w).toEqual([1, 1]);
  });
});
