/**
 * Decision tree (CART with Gini), built breadth first.
 *
 * The usual recursive version builds a whole branch before the other one.
 * Here a queue makes the animation grow the tree level by level, which is
 * easier to follow.
 */
import type { Point, StepInfo, Trainer } from '../../core/types';
import { applySplit, bestSplit, makeNode, type TreeNode } from '../shared/tree';

export interface TreeParams {
  depth: number;
  minSplit: number;
}

export interface TreeState {
  root: TreeNode;
  queue: TreeNode[];
  current: TreeNode | null;
  leaves: number;
}

export function* train(data: Point[], P: TreeParams, s: TreeState): Generator<StepInfo, StepInfo> {
  while (s.queue.length) {
    const node = s.queue.shift()!;                                      // line 1
    s.current = node;
    const pure = node.prob === 0 || node.prob === 1;
    if (pure || node.depth >= P.depth || node.n < P.minSplit) {         // line 2
      node.leaf = true;
      s.leaves++;
      const why = pure ? 'is pure' : node.depth >= P.depth ? 'is at the maximum depth' : 'has too few points';
      yield { line: 3, message: `A node with ${node.n} points ${why}, so it becomes a leaf (P(B) = ${node.prob.toFixed(2)}).` };
      continue;
    }
    const split = bestSplit(data, node.idx, [0, 1]);                    // lines 4 to 6
    if (!split) {
      node.leaf = true;
      s.leaves++;
      yield { line: 3, message: 'No split lowers the impurity, so the node becomes a leaf.' };
      continue;
    }
    const [l, r] = applySplit(data, node, split);                       // line 7
    s.queue.push(l, r);                                                 // line 8
    yield {
      line: 7,
      message: `Split on ${split.feature === 0 ? 'x' : 'y'} < ${split.threshold.toFixed(2)}: Gini drops by ${split.gain.toFixed(3)} (${l.n} left, ${r.n} right).`,
    };
  }
  s.current = null;
  return { line: 0, message: `Tree complete with ${s.leaves} leaves.` };
}

export function createTree(data: Point[], P: TreeParams): Trainer<TreeState> {
  const root = makeNode(data, data.map((_, i) => i), 0, [-1, 1, -1, 1]);
  const state: TreeState = { root, queue: [root], current: null, leaves: 0 };
  return { state, steps: train(data, P, state) };
}
