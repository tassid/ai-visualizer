/**
 * Random forest (Breiman, 2001).
 *
 * Two sources of randomness make the trees differ:
 *   1. bagging: each tree trains on a bootstrap sample (n draws with replacement);
 *   2. random subspace: each split may only use a random subset of features.
 *
 * Averaging models whose errors are weakly correlated lowers the variance.
 * That is why the forest's boundary is smoother than a single tree's.
 *
 * About 36.8% of the points are left out of each bootstrap sample (the
 * chance of never being drawn is (1 - 1/n)ⁿ ≈ 1/e). Those out-of-bag points
 * are a free validation set.
 */
import { randInt, random } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';
import { applySplit, bestSplit, makeNode, predictTree, type TreeNode } from '../shared/tree';

export interface ForestParams {
  trees: number;
  depth: number;
  maxFeatures: 'all' | '1 at random';
}

export interface ForestState {
  trees: TreeNode[];
  /** inBag[t][i] = 1 if point i was in tree t's sample. */
  inBag: Uint8Array[];
  oobAccuracy: number;
  history: number[];
}

/** Grows one full tree (no animation) on a sample. */
export function growTree(data: Point[], idx: number[], P: ForestParams): TreeNode {
  const root = makeNode(data, idx, 0, [-1, 1, -1, 1]);
  const stack = [root];
  while (stack.length) {
    const node = stack.pop()!;
    if (node.depth >= P.depth || node.n < 2 || node.prob === 0 || node.prob === 1) { node.leaf = true; continue; }
    let split;
    if (P.maxFeatures === 'all') split = bestSplit(data, node.idx, [0, 1]);
    else {
      const f: 0 | 1 = random() < 0.5 ? 0 : 1;
      split = bestSplit(data, node.idx, [f]) ?? bestSplit(data, node.idx, [f === 0 ? 1 : 0]);
    }
    if (!split) { node.leaf = true; continue; }
    stack.push(...applySplit(data, node, split));
  }
  return root;
}

export const forestPredict = (s: ForestState, x: number, y: number) => {
  if (!s.trees.length) return 0.5;
  let sum = 0;
  for (const t of s.trees) sum += predictTree(t, x, y);
  return sum / s.trees.length;
};

/** Out-of-bag accuracy: each point is judged only by the trees that never saw it. */
export function outOfBag(data: Point[], s: ForestState): number {
  let ok = 0, counted = 0;
  data.forEach((p, i) => {
    let sum = 0, c = 0;
    s.trees.forEach((t, k) => { if (!s.inBag[k][i]) { sum += predictTree(t, p.x, p.y); c++; } });
    if (c) { counted++; if ((sum / c > 0.5 ? 1 : 0) === p.c) ok++; }
  });
  return counted ? ok / counted : NaN;
}

export function* train(data: Point[], P: ForestParams, s: ForestState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  for (let t = 1; t <= P.trees; t++) {                                  // line 1
    const idx: number[] = [];
    const bag = new Uint8Array(n);
    for (let k = 0; k < n; k++) { const i = randInt(n); idx.push(i); bag[i] = 1; } // line 2
    s.trees.push(growTree(data, idx, P));                               // line 3
    s.inBag.push(bag);                                                  // line 4
    s.oobAccuracy = outOfBag(data, s);
    s.history.push(1 - s.oobAccuracy);
    const unique = bag.reduce((a, b) => a + b, 0);
    yield { line: 3, message: `Tree ${t}: its sample has ${unique} distinct points (${((1 - unique / n) * 100).toFixed(0)}% left out). Out-of-bag accuracy ${(s.oobAccuracy * 100).toFixed(1)}%.` };
  }
  return { line: 6, message: `A forest of ${P.trees} trees. The prediction is their average.` };
}

export function createForest(data: Point[], P: ForestParams): Trainer<ForestState> {
  const state: ForestState = { trees: [], inBag: [], oobAccuracy: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
