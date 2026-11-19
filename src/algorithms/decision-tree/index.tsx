import { theory } from './theory';
import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, pct, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { TreeDiagram } from '../../components/charts/TreeDiagram';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { countNodes, gini, pathTo, predictTree, splitSegments, toDiagram, treeDepth, type TreeNode } from '../shared/tree';
import { createTree, type TreeParams, type TreeState } from './model';

const ease = { duration: 0.35, ease: [0.45, 0, 0.55, 1] } as const;

/** Splits grow out from their midpoint, like Manim's Create animation. */
export function SplitLines({ root, color = 'rgba(255,255,255,.75)', width = 1.5 }: { root: TreeNode; color?: string; width?: number }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {splitSegments(root).map(([x0, y0, x1, y1]) => {
        const key = `${x0.toFixed(4)}-${y0.toFixed(4)}-${x1.toFixed(4)}-${y1.toFixed(4)}`;
        const cx = x((x0 + x1) / 2), cy = y((y0 + y1) / 2);
        return <motion.line key={key} initial={{ x1: cx, y1: cy, x2: cx, y2: cy }} animate={{ x1: x(x0), y1: y(y0), x2: x(x1), y2: y(y1) }} transition={ease} stroke={color} strokeWidth={width} />;
      })}
    </g>
  );
}

function BoxOutline({ node, color, dash, width = 1 }: { node: TreeNode; color: string; dash?: string; width?: number }) {
  const { x, y } = usePlot();
  const [x0, x1, y0, y1] = node.box;
  return (
    <motion.rect
      initial={false}
      animate={{ x: x(x0) + 1, y: y(y1) + 1, width: Math.max(0, x(x1) - x(x0) - 2), height: Math.max(0, y(y0) - y(y1) - 2) }}
      transition={{ duration: 0.15 }}
      fill="none" stroke={color} strokeWidth={width} strokeDasharray={dash} pointerEvents="none"
    />
  );
}

function TreeVisual({ state: s, data, version, hover, compact }: VisualProps<TreeState>) {
  const leaf = hover ? pathTo(s.root, hover.x, hover.y).at(-1)! : null;
  return (
    <Plot heatmap={{ version, resolution: 112, color: (x, y) => binaryColor(predictTree(s.root, x, y)) }}>
      <SplitLines root={s.root} />
      {!compact && s.queue.map((q) => <BoxOutline key={q.id} node={q} color="rgba(255,255,255,.3)" dash="4 5" />)}
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && leaf && <BoxOutline node={leaf} color={COLORS.yellow} width={2} />}
      {!compact && leaf && <Ring at={hover} r={6} color={COLORS.yellow} />}
      {!compact && !leaf && s.current && <BoxOutline node={s.current} color={COLORS.yellow} width={2.5} />}
    </Plot>
  );
}

function TreeAux({ state: s, hover }: VisualProps<TreeState>) {
  const path = hover ? new Set(pathTo(s.root, hover.x, hover.y).map((n) => n.id)) : undefined;
  return <TreeDiagram root={toDiagram(s.root)} currentId={hover ? undefined : s.current?.id} path={path} width={560} height={250} />;
}

export const decisionTree: AlgorithmDef<TreeState> = {
  id: 'decision-tree',
  name: 'Decision tree',
  group: 'Neighbors, trees and ensembles',
  kind: 'classification',
  datasets: ['moons', 'xor', 'circles', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 3,
  thumbnailSteps: 500,
  hint: 'Hover the plane to trace a point’s path through the tree below.',
  params: [
    { key: 'depth', label: 'Max depth', type: 'range', min: 1, max: 12, step: 1, value: 6 },
    { key: 'minSplit', label: 'Min points to split', type: 'range', min: 2, max: 30, step: 1, value: 4 },
  ],
  create: (data, params) => createTree(data, params as unknown as TreeParams),
  Visual: TreeVisual,
  aux: { title: 'Tree', Component: TreeAux },
  equation: (s) => {
    const n = s.current;
    const head = `G(p) = 2p\\,(1-p)`;
    if (!n || !n.left || !n.right) return head;
    const g = (n.left.n * gini(n.left.prob) + n.right.n * gini(n.right.prob)) / n.n;
    return `${head} \\qquad ${tex(gini(n.prob), 3)} \\ \\to\\ \\tfrac{${n.left.n}}{${n.n}}\\,${tex(gini(n.left.prob), 3)} + \\tfrac{${n.right.n}}{${n.n}}\\,${tex(gini(n.right.prob), 3)} = {\\color{${COLORS.yellow}}${tex(g, 3)}}`;
  },
  metrics: (s, data) => [
    { label: 'Nodes', value: countNodes(s.root) },
    { label: 'Leaves', value: s.leaves },
    { label: 'Waiting', value: s.queue.length },
    { label: 'Depth', value: treeDepth(s.root) },
    { label: 'Training accuracy', value: pct(accuracy(data, (x, y) => predictTree(s.root, x, y))) },
  ],
  docs: {
    ...theory,
    description:
      'A decision tree cuts the plane recursively with axis-aligned splits. At each node it tries every possible split on every feature and keeps the one that lowers the Gini impurity the most. The animation builds the tree level by level: each step decides one node (outlined in yellow), either splitting it or turning it into a leaf. Dashed boxes are still waiting. Deep trees memorize the data, as the training accuracy reaching 100% shows; shallow trees generalize better.',
    steps: [
      'Start with all the data at the root.',
      'For each feature, sort the values and try every midpoint as a threshold.',
      'Keep the split with the lowest weighted Gini, where $G = 2p(1-p)$.',
      'Create two children and repeat until a node is pure, too small or at the maximum depth.',
    ],
    complexity: [['Training', 'O(d\\,n\\log n)\\ \\text{per level}'], ['Prediction', 'O(h)'], ['Memory', 'O(\\text{nodes})']],
    complexityNote: 'h is the depth of the tree. In the worst case h reaches n.',
    pseudocode: [
      'queue ← [root with all the data]',
      'while the queue is not empty: node ← queue.pop()',
      '    if node is pure, too deep or too small:',
      '        node becomes a leaf ; continue',
      '    for each feature f and threshold t:',
      '        g ← weighted Gini of the split (f < t)',
      '    (f*, t*) ← split with the lowest g',
      '    split node into left (f* < t*) and right',
      '    queue.add(left, right)',
    ],
    exercises: [
      'With depth 12 and min 2, training accuracy hits 100%. Add a few points of the wrong class and watch the tree carve regions just for them.',
      'Replace Gini with entropy, -p log p - (1-p) log(1-p), in shared/tree.ts. Do the trees change much?',
      'Implement cost-complexity pruning.',
      'Why do trees struggle with diagonal boundaries, like the linearly separable data?',
    ],
  },
};
