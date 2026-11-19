import { theory } from './theory';
import { AnimatePresence, motion } from 'framer-motion';
import { CLUSTER_PALETTE, COLORS } from '../../core/colors';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { createDbscan, NOISE, regionAt, UNSEEN, type DbscanParams, type DbscanState } from './model';

/** Points waiting in the expansion queue get a ring that grows in and fades out. */
function QueueRings({ s, data }: { s: DbscanState; data: VisualProps<DbscanState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <AnimatePresence>
      {s.queue.map((j) => (
        <motion.circle key={j} cx={x(data[j].x)} cy={y(data[j].y)} initial={{ r: 2, opacity: 0 }} animate={{ r: 7.5, opacity: 0.8 }} exit={{ r: 12, opacity: 0 }} transition={{ duration: 0.25 }} fill="none" stroke={COLORS.yellow} strokeWidth={1.2} pointerEvents="none" />
      ))}
    </AnimatePresence>
  );
}

function DbscanVisual({ state: s, data, params, hover, compact }: VisualProps<DbscanState>) {
  const eps = Number(params.eps);
  const center = hover ?? (s.current >= 0 ? data[s.current] : null);
  const inside = hover ? new Set(regionAt(data, hover.x, hover.y, eps)) : null;
  const color = (i: number) => (s.label[i] === UNSEEN ? COLORS.grey : s.label[i] === NOISE ? COLORS.greyDark : CLUSTER_PALETTE[s.label[i] % CLUSTER_PALETTE.length]);
  return (
    <Plot>
      {!compact && <QueueRings s={s} data={data} />}
      <Points data={data} color={(_, i) => color(i)} radius={(_, i) => (inside?.has(i) ? 6.5 : s.core[i] ? 5.4 : s.label[i] === NOISE ? 3.4 : 3.9)} />
      {!compact && center && (
        <>
          <Ring at={center} worldRadius={eps} color="rgba(255,255,255,.8)" width={1.2} dash="5 5" />
          <Ring at={center} r={hover ? 4 : 8} color={COLORS.yellow} width={2.5} />
        </>
      )}
    </Plot>
  );
}

export const dbscan: AlgorithmDef<DbscanState> = {
  id: 'dbscan',
  name: 'DBSCAN',
  group: 'Clustering',
  kind: 'clustering',
  datasets: ['moons', 'circles', 'clusters', 'spiral'],
  defaultDataset: 'moons',
  defaultSpeed: 6,
  thumbnailSteps: 2000,
  hint: 'Hover the plane to see the ε-ball there and whether it would be a core point.',
  params: [
    { key: 'eps', label: 'Radius ε', type: 'range', min: 0.03, max: 0.3, step: 0.01, value: 0.15 },
    { key: 'minPts', label: 'minPts', type: 'range', min: 2, max: 12, step: 1, value: 4, help: 'Neighbors needed, the point itself included, to be a core point.' },
  ],
  create: (data, params) => createDbscan(data, params as unknown as DbscanParams),
  Visual: DbscanVisual,
  equation: (s, data, params, hover) => {
    const eps = Number(params.eps), m = Number(params.minPts);
    const center = hover ?? (s.current >= 0 ? data[s.current] : null);
    const head = `\\big|N_\\varepsilon(\\mathbf{p})\\big| \\ge \\text{minPts}`;
    if (!center) return head;
    const n = regionAt(data, center.x, center.y, eps).length;
    return `${head} \\qquad ${n} ${n >= m ? '\\ge' : '<'} ${m} \\ \\Rightarrow\\ \\text{${n >= m ? 'core' : 'not core'}}`;
  },
  metrics: (s, data) => {
    let noise = 0, core = 0, visited = 0;
    s.label.forEach((l, i) => { if (l === NOISE) noise++; if (s.core[i]) core++; if (s.visited[i]) visited++; });
    return [
      { label: 'Clusters', value: s.clusters },
      { label: 'Core points', value: core },
      { label: 'Noise', value: noise },
      { label: 'Queue', value: s.queue.length },
      { label: 'Visited', value: `${visited} of ${data.length}` },
    ];
  },
  docs: {
    ...theory,
    description:
      'DBSCAN groups points by density. A point is a core point (drawn larger) if at least minPts points lie within radius $\\varepsilon$. Clusters grow by linking neighboring core points, like a breadth-first search: the points waiting in the queue have yellow rings. Points that belong to no dense region stay as noise, in dark grey. Unlike k-means, you do not choose the number of clusters, and it finds arbitrary shapes such as the moons. The dashed circle is the $\\varepsilon$-ball of the point being examined.',
    steps: [
      'Take an unvisited point and count its neighbors within $\\varepsilon$.',
      'If there are fewer than minPts, mark it as noise (it may become a border point later).',
      'If there are at least minPts, start a new cluster and queue the neighbors.',
      'Expand the queue: neighbors that are core points bring in their own neighbors.',
      'Repeat until every point is visited.',
    ],
    complexity: [['Training (brute force)', 'O(n^2)'], ['Training (spatial index)', 'O(n\\log n)'], ['Prediction', '\\text{n/a}'], ['Memory', 'O(n)']],
    pseudocode: [
      'C ← 0',
      'for each unvisited point p:',
      '    mark p visited',
      '    N ← neighbors(p, ε)',
      '    if |N| < minPts: p ← noise ; continue',
      '',
      '    C ← C + 1 ;  p ← cluster C ;  queue ← N',
      '    for each q in the queue:',
      '        if q unvisited: mark q',
      "            N' ← neighbors(q, ε)",
      "            if |N'| ≥ minPts: queue ← queue ∪ N'",
      '        if q has no cluster: q ← cluster C',
    ],
    exercises: [
      'On the moons, find the range of ε that gives exactly 2 clusters. What happens below and above it?',
      'Drag points to build a thin bridge between the two moons. How many points does it take to merge them?',
      'Add a spatial grid (cells of side ε) to region() and measure the speedup.',
      'Plot the k-distance graph (distance to the k-th neighbor, sorted) to choose ε.',
    ],
  },
};
