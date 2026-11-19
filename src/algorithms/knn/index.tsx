import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { createKnn, neighbors, type KnnParams, type KnnState, type Neighborhood, type Weighting } from './model';

/** Cells classified so far, as SVG squares. */
function Cells({ s }: { s: KnnState }) {
  const { size } = usePlot();
  const R = s.res, cs = size / R;
  const rects = [];
  for (let c = 0; c < s.done; c++) {
    const [r, g, b, a] = binaryColor(s.cells[c]);
    rects.push(<rect key={c} x={(c % R) * cs} y={Math.floor(c / R) * cs} width={cs + 0.6} height={cs + 0.6} fill={`rgba(${r | 0},${g | 0},${b | 0},${a + 0.06})`} />);
  }
  return <g pointerEvents="none">{rects}</g>;
}

/** Lines to the neighbors, the radius that contains them, and the query marker. */
function Query({ q, data }: { q: Neighborhood; data: VisualProps<KnnState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {q.idx.map((i) => (
        <motion.line key={i} initial={false} animate={{ x1: x(q.x), y1: y(q.y), x2: x(data[i].x), y2: y(data[i].y) }} transition={{ duration: 0.08 }} stroke="rgba(255,255,255,.5)" strokeWidth={1} />
      ))}
      <Ring at={q} worldRadius={q.radius} color={COLORS.yellow} width={1.2} dash="5 5" />
      <motion.rect initial={false} animate={{ x: x(q.x) - 6, y: y(q.y) - 6 }} transition={{ duration: 0.08 }} width={12} height={12} fill={q.p > 0.5 ? COLORS.blue : q.p < 0.5 ? COLORS.red : '#fff'} stroke={COLORS.yellow} strokeWidth={2} />
    </g>
  );
}

function KnnVisual({ state: s, data, params, hover, compact }: VisualProps<KnnState>) {
  const probe = hover && data.length ? neighbors(data, hover.x, hover.y, Number(params.k), params.weighting as Weighting) : null;
  const q = compact ? null : probe ?? s.current;
  const near = new Set(q?.idx ?? []);
  return (
    <Plot>
      <Cells s={s} />
      <Points data={data} color={(p) => classColor(p.c)} radius={(_, i) => (near.has(i) ? 6.5 : 4.4)} />
      {q && <Query q={q} data={data} />}
    </Plot>
  );
}

export const knn: AlgorithmDef<KnnState> = {
  id: 'knn',
  name: 'k-nearest neighbors',
  group: 'Instance and tree methods',
  kind: 'classification',
  datasets: ['moons', 'circles', 'xor', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 8,
  thumbnailSteps: 2000,
  hint: 'Hover the plane to query any point and see the vote.',
  params: [
    { key: 'k', label: 'k', type: 'range', min: 1, max: 31, step: 2, value: 7, help: 'An odd k avoids ties between two classes.' },
    { key: 'weighting', label: 'Vote weight', type: 'select', options: ['uniform', 'distance'], value: 'uniform' },
    { key: 'res', label: 'Sweep resolution', type: 'select', options: [20, 30, 40], value: 30 },
  ],
  create: (data, params) => createKnn(data, params as unknown as KnnParams),
  Visual: KnnVisual,
  equation: (s, data, params, hover) => {
    const q = hover && data.length ? neighbors(data, hover.x, hover.y, Number(params.k), params.weighting as Weighting) : s.current;
    const head = `\\hat y(\\mathbf{q}) = \\operatorname{majority}\\{\\,y_i : i \\in N_k(\\mathbf{q})\\,\\}`;
    if (!q) return head;
    const b = Math.round(q.p * q.idx.length), a = q.idx.length - b;
    return `${head} \\qquad {\\color{${COLORS.red}}${a}\\ \\text{A}} \\ \\text{vs}\\ {\\color{${COLORS.blue}}${b}\\ \\text{B}}`;
  },
  metrics: (s, data) => [
    { label: 'Cells', value: `${s.done} of ${s.res * s.res}` },
    { label: 'Leave-one-out accuracy', value: pct(s.looAccuracy) },
    { label: 'Stored points', value: data.length },
  ],
  docs: {
    description:
      'KNN has no training phase: it just stores the data. To classify a new point, it measures the distance to every example, takes the $k$ closest and lets them vote. The animation sweeps the plane cell by cell, showing the neighbors behind each decision and the radius that contains them. A small $k$ makes jagged boundaries that memorize the noise; a large $k$ smooths too much. Leave-one-out accuracy classifies each point using all the others, an honest way to measure error.',
    steps: [
      'Store all training examples.',
      'For a query point, compute the distance to each example.',
      'Sort and keep the $k$ closest.',
      'Predict the class with the most votes (optionally weighted by $1/\\text{distance}$).',
    ],
    complexity: [['Training', 'O(1)'], ['Prediction (brute force)', 'O(n\\,d)'], ['Prediction (k-d tree)', '\\approx O(\\log n)'], ['Memory', 'O(n\\,d)']],
    complexityNote: 'In high dimensions all distances look alike (the curse of dimensionality) and k-d trees lose their advantage.',
    pseudocode: [
      'function classify(q):',
      '    D ← []',
      '    for each (x, y) in the data:',
      '        D.add( (distance(q, x), y) )',
      '    sort D by distance ; V ← first k',
      '    return the most common class in V',
    ],
    exercises: [
      'Move k from 1 to 31 on the moons and note the leave-one-out accuracy. Which k is best?',
      'Drop a single point of the wrong class inside a cluster. How far does its influence reach with k = 1? With k = 15?',
      'Swap Euclidean distance for Manhattan distance in model.ts. Does the boundary change shape?',
      'Build a k-d tree and compare prediction time against brute force for 10,000 points.',
    ],
  },
};
