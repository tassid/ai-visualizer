import { CLUSTER_PALETTE, COLORS } from '../../core/colors';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points } from '../../components/plot/marks';
import { ESL, ISLR, paper } from '../books';
import { createHc, type HcParams, type HcState, type Merge } from './model';

/** Color of a live cluster: stable per cluster id; singletons stay grey. */
const clusterColor = (id: number, size: number) => (size < 2 ? COLORS.grey : CLUSTER_PALETTE[(id * 7) % CLUSTER_PALETTE.length]);

function HcVisual({ state: s, data, compact }: VisualProps<HcState>) {
  const color = (i: number) => {
    if (s.cut) return CLUSTER_PALETTE[s.cut[i] % CLUSTER_PALETTE.length];
    const id = s.owner[i];
    return clusterColor(id, s.members.get(id)?.length ?? 1);
  };
  return (
    <Plot>
      {!compact && s.last && <MergeHull s={s} data={data} />}
      <Points data={data} color={(_, i) => color(i)} radius={(_, i) => (s.last && s.owner[i] === s.last.id ? 6 : 4.2)} />
    </Plot>
  );
}

/** The link that was just made: a line between the two clusters' centers. */
function MergeHull({ s, data }: { s: HcState; data: VisualProps<HcState>['data'] }) {
  const { x, y } = usePlot();
  const m = s.last!;
  const pts = s.members.get(m.id) ?? [];
  const cx = pts.reduce((a, i) => a + data[i].x, 0) / pts.length, cy = pts.reduce((a, i) => a + data[i].y, 0) / pts.length;
  let r = 0;
  for (const i of pts) r = Math.max(r, Math.hypot(data[i].x - cx, data[i].y - cy));
  return <circle cx={x(cx)} cy={y(cy)} r={(r * 280) + 10} fill="none" stroke={COLORS.yellow} strokeWidth={1.5} strokeDasharray="4 4" pointerEvents="none" />;
}

/** Dendrogram drawn with the merges done so far. Leaves are ordered by the final tree. */
function Dendrogram({ state: s, data, params }: VisualProps<HcState>) {
  const W = 560, H = 240, pad = 16;
  const n = data.length;
  if (n < 2 || !s.merges.length) return <svg viewBox={`0 0 ${W} ${H}`} className="aux-svg" />;
  // order leaves by a depth-first walk of the tree built so far (unmerged leaves keep their index order)
  const children = new Map<number, [number, number]>();
  s.merges.forEach((m) => children.set(m.id, [m.a, m.b]));
  const roots = [...s.members.keys()].sort((a, b) => Math.min(...(s.members.get(a) ?? [a])) - Math.min(...(s.members.get(b) ?? [b])));
  const order: number[] = [];
  const walk = (id: number) => { const c = children.get(id); if (c) { walk(c[0]); walk(c[1]); } else order.push(id); };
  roots.forEach(walk);
  const xi = new Map<number, number>();
  order.forEach((leaf, k) => xi.set(leaf, pad + (k / Math.max(1, n - 1)) * (W - 2 * pad)));
  const maxH = Math.max(...s.merges.map((m) => m.height), 1e-9);
  const yh = (h: number) => H - pad - (h / maxH) * (H - 3 * pad);
  const pos = new Map<number, { x: number; h: number }>();
  for (let i = 0; i < n; i++) pos.set(i, { x: xi.get(i)!, h: 0 });
  const k = Number(params.k);
  const links = s.merges.map((m: Merge, idx) => {
    const a = pos.get(m.a)!, b = pos.get(m.b)!;
    pos.set(m.id, { x: (a.x + b.x) / 2, h: m.height });
    const last = idx === s.merges.length - 1 && !s.cut;
    return <path key={m.id} d={`M${a.x},${yh(a.h)}V${yh(m.height)}H${b.x}V${yh(b.h)}`} fill="none" stroke={last ? COLORS.yellow : 'rgba(236,236,236,.55)'} strokeWidth={last ? 2 : 1} />;
  });
  // cut line between the (n-k)-th and (n-k+1)-th merge heights
  const cutH = s.cut && s.merges.length >= k ? (s.merges[n - k - 1]?.height + (s.merges[n - k]?.height ?? maxH)) / 2 : null;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="aux-svg">
      {links}
      {cutH !== null && Number.isFinite(cutH) && <line x1={pad} x2={W - pad} y1={yh(cutH)} y2={yh(cutH)} stroke={COLORS.yellow} strokeDasharray="6 5" />}
    </svg>
  );
}

export const hierarchical: AlgorithmDef<HcState> = {
  id: 'hierarchical',
  name: 'Hierarchical clustering',
  group: 'Clustering',
  kind: 'clustering',
  datasets: ['clusters', 'moons', 'circles', 'blobs'],
  defaultDataset: 'clusters',
  defaultSpeed: 7,
  thumbnailSteps: 1000,
  hint: 'Try single linkage on the moons, then Ward: one follows chains, the other prefers round groups.',
  params: [
    { key: 'linkage', label: 'Linkage', type: 'select', options: ['ward', 'average', 'complete', 'single'], value: 'ward' },
    { key: 'k', label: 'Clusters at the cut', type: 'range', min: 2, max: 8, step: 1, value: 4 },
  ],
  create: (data, params) => createHc(data, params as unknown as HcParams),
  Visual: HcVisual,
  aux: { title: 'Dendrogram', Component: Dendrogram },
  equation: (_s, _d, params) => {
    const forms: Record<string, string> = {
      single: 'd(A\\cup B, C) = \\min\\big(d(A,C),\\ d(B,C)\\big)',
      complete: 'd(A\\cup B, C) = \\max\\big(d(A,C),\\ d(B,C)\\big)',
      average: 'd(A\\cup B, C) = \\frac{|A|\\,d(A,C) + |B|\\,d(B,C)}{|A| + |B|}',
      ward: 'd(A\\cup B, C) = \\frac{(|A|+|C|)\\,d_{AC} + (|B|+|C|)\\,d_{BC} - |C|\\,d_{AB}}{|A|+|B|+|C|}',
    };
    return forms[String(params.linkage)];
  },
  metrics: (s, data) => [
    { label: 'Merges', value: `${s.merges.length} of ${Math.max(0, data.length - 1)}` },
    { label: 'Clusters left', value: s.members.size },
    { label: 'Last merge height', value: s.merges.length ? s.merges[s.merges.length - 1].height.toFixed(3) : 'n/a' },
  ],
  docs: {
    description:
      'Agglomerative clustering builds a whole hierarchy of clusterings at once. It starts with every point alone and, at each step, merges the two closest clusters. The dendrogram below records every merge at the height where it happened; cutting it with a horizontal line gives a clustering with any number of groups, without rerunning anything. The linkage defines the distance between clusters: the closest pair of points (single), the farthest pair (complete), the average pair, or Ward’s increase in within-cluster variance.',
    steps: [
      'Put each point in its own cluster and compute all pairwise distances.',
      'Find the two closest clusters and merge them.',
      'Update the distances from the new cluster to every other one with the linkage rule.',
      'Repeat until one cluster remains; cut the dendrogram to get $k$ clusters.',
    ],
    complexity: [['Training (naive)', 'O(n^3)'], ['Training (priority queue)', 'O(n^2\\log n)'], ['Training (single linkage, MST)', 'O(n^2)'], ['Memory', 'O(n^2)']],
    pseudocode: [
      'each point is its own cluster ; compute D[i, j]',
      'repeat n - 1 times:',
      '    (a, b) ← the pair of clusters with the smallest D',
      '    merge a and b into c',
      '    for every other cluster i: D[c, i] ← linkage(D[a, i], D[b, i])',
      '    record (a, b, D[a, b]) in the dendrogram',
      'cut the dendrogram to get k clusters',
    ],
    exercises: [
      'On the moons, single linkage separates them and Ward does not. On the four clusters it is the opposite. Explain both.',
      'Single linkage suffers from chaining. Add a line of points between two clusters and watch them merge early.',
      'Show that the single linkage dendrogram can be built from a minimum spanning tree of the points.',
      'Where would you cut the dendrogram? Look for the largest vertical gap between consecutive merge heights.',
    ],
    theory: [
      'A hierarchical clustering is a nested sequence of partitions, from $n$ singletons to one cluster. Agglomerative methods build it bottom up; divisive methods split top down. The result is a dendrogram, a binary tree in which the height of each internal node is the dissimilarity at which its two children merged. For single, complete and average linkage the heights increase monotonically, so the tree can be cut at any height.',
      'The linkage turns point distances into cluster distances. Single linkage, $d(A,B) = \\min_{a\\in A, b\\in B} d(a,b)$, finds elongated or curved groups but chains through bridges of points. Complete linkage, the maximum, prefers compact groups of similar diameter. Average linkage is in between. Ward’s method merges the pair that least increases the total within-cluster sum of squares, the same objective as k-means; with squared Euclidean distances the increase is',
      '$$\\Delta(A,B) = \\frac{|A|\\,|B|}{|A| + |B|}\\,\\lVert\\bar{\\mathbf{x}}_A - \\bar{\\mathbf{x}}_B\\rVert^2.$$',
      'All four are special cases of the Lance-Williams recurrence $d(A\\cup B, C) = \\alpha_A d(A,C) + \\alpha_B d(B,C) + \\beta\\,d(A,B) + \\gamma\\,|d(A,C) - d(B,C)|$, with coefficients that depend on the cluster sizes. That recurrence lets the algorithm update one row of the distance matrix per merge instead of recomputing from the points.',
      'A dendrogram should not be read too literally. Two points that sit next to each other at the bottom of the tree need not be close; what matters is the height at which they first join. And the hierarchy is imposed by the method: if the true groups are not nested, no single cut recovers them.',
    ],
    references: [
      ISLR('Section 12.4.2, Hierarchical clustering'),
      ESL('Section 14.3.12, Hierarchical clustering'),
      paper('Ward, Hierarchical grouping to optimize an objective function, Journal of the American Statistical Association 58 (1963)'),
      paper('Lance and Williams, A general theory of classificatory sorting strategies, The Computer Journal 9 (1967)'),
      paper('Müllner, Modern hierarchical, agglomerative clustering algorithms (2011)', 'https://arxiv.org/abs/1109.2378'),
    ],
  },
};
