import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, pct, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { countNodes, predictTree } from '../shared/tree';
import { SplitLines } from '../decision-tree';
import { createForest, forestPredict, type ForestParams, type ForestState } from './model';

function ForestVisual({ state: s, data, version, hover, compact }: VisualProps<ForestState>) {
  const last = s.trees[s.trees.length - 1];
  const bag = s.inBag[s.inBag.length - 1];
  return (
    <Plot heatmap={s.trees.length ? { version, resolution: 96, color: (x, y) => binaryColor(forestPredict(s, x, y)) } : null}>
      {last && !compact && <SplitLines key={s.trees.length} root={last} color="rgba(255,255,0,.5)" width={1.1} />}
      <Points data={data} color={(p) => classColor(p.c)} radius={(_, i) => (!bag || bag[i] ? 4.6 : 2.6)} />
      {!compact && hover && <Ring at={hover} r={6} color={COLORS.yellow} />}
    </Plot>
  );
}

export const randomForest: AlgorithmDef<ForestState> = {
  id: 'random-forest',
  name: 'Random forest',
  group: 'Instance and tree methods',
  kind: 'classification',
  datasets: ['moons', 'circles', 'xor', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 3,
  thumbnailSteps: 20,
  hint: 'Hover the plane to count how the trees vote at that point.',
  params: [
    { key: 'trees', label: 'Trees', type: 'range', min: 1, max: 80, step: 1, value: 40 },
    { key: 'depth', label: 'Max depth', type: 'range', min: 1, max: 12, step: 1, value: 7 },
    { key: 'maxFeatures', label: 'Features per split', type: 'select', options: ['1 at random', 'all'], value: '1 at random', help: 'With "all", only the bootstrap sample makes the trees differ.' },
  ],
  create: (data, params) => createForest(data, params as unknown as ForestParams),
  Visual: ForestVisual,
  history: { label: 'Out-of-bag error', get: (s) => s.history },
  equation: (s, _data, _params, hover) => {
    const head = `\\hat p(\\mathbf{x}) = \\frac{1}{T}\\sum_{t=1}^{T} h_t(\\mathbf{x})`;
    if (!hover || !s.trees.length) return `${head} \\qquad T = ${s.trees.length}`;
    const votesB = s.trees.filter((t) => predictTree(t, hover.x, hover.y) > 0.5).length;
    return `${head} = ${tex(forestPredict(s, hover.x, hover.y))} \\qquad {\\color{${COLORS.blue}}${votesB}} \\text{ of } ${s.trees.length} \\text{ trees vote B}`;
  },
  metrics: (s, data) => [
    { label: 'Trees', value: s.trees.length },
    { label: 'Training accuracy', value: pct(accuracy(data, (x, y) => forestPredict(s, x, y))) },
    { label: 'Out-of-bag accuracy', value: pct(s.oobAccuracy) },
    { label: 'Nodes in the last tree', value: s.trees.length ? countNodes(s.trees[s.trees.length - 1]) : 0 },
  ],
  docs: {
    description:
      'A random forest averages many decision trees. Each tree trains on a bootstrap sample (drawn with replacement) and each split may only use one randomly chosen feature. The trees make different mistakes, and averaging them lowers the variance, so the boundary is smoother and steadier than a single tree’s. The yellow lines are the splits of the newest tree, and the small points were left out of its sample. The out-of-bag error, charted on the side, estimates the test error without holding out any data.',
    steps: [
      'Draw $n$ examples with replacement (a bootstrap sample).',
      'Grow a tree on that sample, picking at random which features each split may use.',
      'Repeat for $T$ trees.',
      'Predict with the average of all trees, $\\hat p(\\mathbf{x}) = \\frac{1}{T}\\sum_t h_t(\\mathbf{x})$.',
    ],
    complexity: [['Training', 'O(T\\,d\\,n\\log n\\,h)'], ['Prediction', 'O(T\\,h)'], ['Memory', 'O(T \\cdot \\text{nodes})']],
    complexityNote: 'T is the number of trees and h their depth. The trees are independent, so training parallelizes perfectly.',
    pseudocode: [
      'forest ← []',
      'for t = 1 to T:',
      '    sample ← draw n examples with replacement',
      '    tree ← build(sample, random features per split)',
      '    forest.add(tree)',
      '',
      'predict(x) = mean of tree(x) over the forest',
    ],
    exercises: [
      'Compare a forest of 1 tree with one of 80. What happens to the out-of-bag accuracy?',
      'Switch "1 at random" to "all". Do the trees look more alike? How does that change the error?',
      'Implement feature importance by mean decrease in impurity.',
      'Implement gradient boosting with shallow trees (depth 1 to 3) and compare it with the forest.',
    ],
  },
};
