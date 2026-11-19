/**
 * Gradient boosting for classification (Friedman, 2001), with log loss.
 *
 * The model is a sum of small regression trees on the log-odds scale:
 *   F(x) = F₀ + ν Σₘ hₘ(x),   p(x) = σ(F(x))
 * Each round fits a tree to the negative gradient of the loss, which for log
 * loss is the residual rᵢ = yᵢ - pᵢ, and sets each leaf to a Newton step:
 *   leaf value = Σ rᵢ / Σ pᵢ(1 - pᵢ)
 * ν (the learning rate, or shrinkage) keeps each step small.
 */
import { binaryCrossEntropy, sigmoid } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';
import type { Box } from '../shared/tree';

export interface GbParams {
  rounds: number;
  depth: number;
  lr: number;
}

export interface RegNode {
  box: Box;
  value: number;
  feature?: 0 | 1;
  threshold?: number;
  left?: RegNode;
  right?: RegNode;
}

export interface GbState {
  f0: number;
  trees: RegNode[];
  /** Current scores F(xᵢ) on the training points. */
  F: Float64Array;
  residuals: Float64Array;
  loss: number;
  history: number[];
}

/**
 * Grows a regression tree on targets r with Hessian weights h, choosing the
 * split that most increases (Σr)²/n on each side, which is the same as most
 * reducing the squared error.
 */
export function growRegTree(data: Point[], idx: number[], r: Float64Array, h: Float64Array, depth: number, box: Box): RegNode {
  let sr = 0, sh = 0;
  for (const i of idx) { sr += r[i]; sh += h[i]; }
  const node: RegNode = { box, value: sr / (sh + 1e-9) };
  if (depth === 0 || idx.length < 4) return node;
  const n = idx.length;
  let best: { f: 0 | 1; t: number; gain: number } | null = null;
  for (const f of [0, 1] as const) {
    const key = (i: number) => (f === 0 ? data[i].x : data[i].y);
    const order = idx.slice().sort((a, b) => key(a) - key(b));
    let left = 0;
    for (let k = 1; k < n; k++) {
      left += r[order[k - 1]];
      const v0 = key(order[k - 1]), v1 = key(order[k]);
      if (v0 === v1) continue;
      const gain = (left * left) / k + ((sr - left) ** 2) / (n - k) - (sr * sr) / n;
      if (!best || gain > best.gain) best = { f, t: (v0 + v1) / 2, gain };
    }
  }
  if (!best || best.gain <= 1e-12) return node;
  const L: number[] = [], R: number[] = [];
  for (const i of idx) ((best.f === 0 ? data[i].x : data[i].y) < best.t ? L : R).push(i);
  const [x0, x1, y0, y1] = box, t = best.t;
  node.feature = best.f;
  node.threshold = t;
  node.left = growRegTree(data, L, r, h, depth - 1, best.f === 0 ? [x0, t, y0, y1] : [x0, x1, y0, t]);
  node.right = growRegTree(data, R, r, h, depth - 1, best.f === 0 ? [t, x1, y0, y1] : [x0, x1, t, y1]);
  return node;
}

export function regPredict(n: RegNode, x: number, y: number): number {
  while (n.left && n.right) n = (n.feature === 0 ? x : y) < n.threshold! ? n.left : n.right;
  return n.value;
}

export function regSegments(n: RegNode, out: [number, number, number, number][] = []) {
  if (!n.left || !n.right) return out;
  const [x0, x1, y0, y1] = n.box, t = n.threshold!;
  out.push(n.feature === 0 ? [t, y0, t, y1] : [x0, t, x1, t]);
  regSegments(n.left, out);
  regSegments(n.right, out);
  return out;
}

export function scoreAt(s: GbState, lr: number, x: number, y: number): number {
  let F = s.f0;
  for (const t of s.trees) F += lr * regPredict(t, x, y);
  return F;
}

export function* train(data: Point[], P: GbParams, s: GbState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  const h = new Float64Array(n);
  const all = data.map((_, i) => i);
  for (let m = 1; m <= P.rounds; m++) {                                 // line 1
    for (let i = 0; i < n; i++) {
      const p = sigmoid(s.F[i]);
      s.residuals[i] = data[i].c - p;                                   // line 2
      h[i] = p * (1 - p);
    }
    const tree = growRegTree(data, all, s.residuals, h, P.depth, [-1, 1, -1, 1]); // lines 3 and 4
    s.trees.push(tree);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      s.F[i] += P.lr * regPredict(tree, data[i].x, data[i].y);          // line 5
      loss += binaryCrossEntropy(sigmoid(s.F[i]), data[i].c) / n;
    }
    s.loss = loss;
    s.history.push(loss);
    yield { line: 5, message: `Round ${m}: a depth-${P.depth} tree fit to the residuals. Log loss ${loss.toFixed(4)}.` };
  }
  return { line: 6, message: `${P.rounds} trees added. Each one corrected what the previous ones got wrong.` };
}

export function createGb(data: Point[], P: GbParams): Trainer<GbState> {
  const n = data.length;
  const pos = data.reduce((a, p) => a + p.c, 0);
  const prior = Math.min(Math.max(pos / Math.max(1, n), 1e-3), 1 - 1e-3);
  const f0 = Math.log(prior / (1 - prior)); // F₀ = log-odds of the base rate
  const state: GbState = { f0, trees: [], F: new Float64Array(n).fill(f0), residuals: new Float64Array(n), loss: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
