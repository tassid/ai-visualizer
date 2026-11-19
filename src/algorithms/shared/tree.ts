/**
 * Decision tree structures and helpers, shared by the single tree and the
 * random forest.
 *
 * Impurity criterion: Gini.  G(p) = 1 - p² - (1-p)² = 2p(1-p)
 *   G = 0   → pure node (one class only)
 *   G = 0.5 → half of each class (maximum uncertainty)
 *
 * A split is good when the weighted average Gini of the children is much
 * lower than the parent's. The difference is the impurity gain.
 */
import type { DiagramNode } from '../../components/charts/TreeDiagram';
import type { Point } from '../../core/types';

/** The rectangle [x0, x1, y0, y1] a node covers on the plane. */
export type Box = [number, number, number, number];

export interface TreeNode {
  id: number;
  idx: number[];
  depth: number;
  box: Box;
  /** Share of class 1 among the node's examples. */
  prob: number;
  n: number;
  leaf: boolean;
  /** Split feature: 0 = x, 1 = y. */
  feature?: 0 | 1;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface Split {
  feature: 0 | 1;
  threshold: number;
  /** Weighted Gini of the children. */
  gini: number;
  gain: number;
}

export const gini = (p: number) => 2 * p * (1 - p);

let nextId = 0;

export function makeNode(data: Point[], idx: number[], depth: number, box: Box): TreeNode {
  let pos = 0;
  for (const i of idx) pos += data[i].c;
  return { id: nextId++, idx, depth, box, prob: idx.length ? pos / idx.length : 0.5, n: idx.length, leaf: false };
}

const value = (p: Point, f: 0 | 1) => (f === 0 ? p.x : p.y);

/**
 * Best split among the allowed features.
 *
 * The trick that makes this O(n log n) per feature instead of O(n²): sort the
 * examples by the feature and sweep left to right, keeping a running count of
 * class 1. Each candidate threshold then costs O(1).
 */
export function bestSplit(data: Point[], idx: number[], features: (0 | 1)[]): Split | null {
  const n = idx.length;
  if (n < 2) return null;
  let total = 0;
  for (const i of idx) total += data[i].c;
  const parent = gini(total / n);
  let best: Split | null = null;

  for (const f of features) {
    const sorted = idx.slice().sort((a, b) => value(data[a], f) - value(data[b], f));
    let left = 0;
    for (let k = 1; k < n; k++) {
      left += data[sorted[k - 1]].c;
      const v0 = value(data[sorted[k - 1]], f), v1 = value(data[sorted[k]], f);
      if (v1 === v0) continue; // equal values cannot be separated
      const g = (k * gini(left / k) + (n - k) * gini((total - left) / (n - k))) / n;
      const gain = parent - g;
      if (gain > 1e-9 && (!best || g < best.gini)) best = { feature: f, threshold: (v0 + v1) / 2, gini: g, gain };
    }
  }
  return best;
}

/** Splits a node into two children, cutting its rectangle too. */
export function applySplit(data: Point[], node: TreeNode, split: Split): [TreeNode, TreeNode] {
  node.feature = split.feature;
  node.threshold = split.threshold;
  const L: number[] = [], R: number[] = [];
  for (const i of node.idx) (value(data[i], split.feature) < split.threshold ? L : R).push(i);
  const [x0, x1, y0, y1] = node.box, t = split.threshold;
  const boxL: Box = split.feature === 0 ? [x0, t, y0, y1] : [x0, x1, y0, t];
  const boxR: Box = split.feature === 0 ? [t, x1, y0, y1] : [x0, x1, t, y1];
  node.left = makeNode(data, L, node.depth + 1, boxL);
  node.right = makeNode(data, R, node.depth + 1, boxR);
  return [node.left, node.right];
}

/** Nodes visited from the root to the leaf (or unsplit node) that contains (x, y). */
export function pathTo(node: TreeNode, x: number, y: number): TreeNode[] {
  const out = [node];
  let n = node;
  while (n.left && n.right) {
    n = (n.feature === 0 ? x : y) < n.threshold! ? n.left : n.right;
    out.push(n);
  }
  return out;
}

export const predictTree = (node: TreeNode, x: number, y: number): number => {
  const p = pathTo(node, x, y);
  return p[p.length - 1].prob;
};

export function countNodes(n: TreeNode | undefined): number {
  return n ? 1 + countNodes(n.left) + countNodes(n.right) : 0;
}

export function treeDepth(n: TreeNode): number {
  return n.left && n.right ? Math.max(treeDepth(n.left), treeDepth(n.right)) : n.depth;
}

/** The splits as segments on the plane: [x0, y0, x1, y1]. */
export function splitSegments(n: TreeNode, out: [number, number, number, number][] = []) {
  if (!n.left || !n.right) return out;
  const [x0, x1, y0, y1] = n.box, t = n.threshold!;
  out.push(n.feature === 0 ? [t, y0, t, y1] : [x0, t, x1, t]);
  splitSegments(n.left, out);
  splitSegments(n.right, out);
  return out;
}

/** Converts to the TreeDiagram component's format. */
export function toDiagram(n: TreeNode): DiagramNode {
  return {
    id: n.id,
    prob: n.prob,
    pending: !n.leaf && !n.left,
    label: n.left ? `${n.feature === 0 ? 'x' : 'y'} < ${n.threshold!.toFixed(2)}` : undefined,
    children: n.left && n.right ? [toDiagram(n.left), toDiagram(n.right)] : undefined,
  };
}
