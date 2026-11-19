import { motion } from 'framer-motion';
import { line } from 'd3-shape';
import { CLUSTER_PALETTE, COLORS, hexToRgb } from '../../core/colors';
import { fmt, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Handle, Points, Ring } from '../../components/plot/marks';
import { createKMeans, inertia, nearest, type KMeansParams, type KMeansState } from './model';

const PAL_RGB = CLUSTER_PALETTE.map(hexToRgb);
const colorOf = (k: number) => CLUSTER_PALETTE[k % CLUSTER_PALETTE.length];

/**
 * Centroids are diamonds that glide to the new mean. The dotted trail keeps
 * every earlier position. Each one is also a drag handle.
 */
function Centroids({ s, compact }: { s: KMeansState; compact?: boolean }) {
  const { x, y } = usePlot();
  const path = line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1]));
  return (
    <>
      <g pointerEvents="none">
        {s.trails.map((t, k) => <path key={`t${k}`} d={path(t) ?? ''} fill="none" stroke={colorOf(k)} strokeOpacity={0.7} strokeWidth={1.5} strokeDasharray="2 4" />)}
        {s.centroids.map((c, k) => (
          <motion.g key={k} initial={false} animate={{ x: x(c.x), y: y(c.y) }} transition={{ duration: 0.4, ease: [0.45, 0, 0.55, 1] }}>
            <rect x={-8} y={-8} width={16} height={16} transform="rotate(45)" fill={colorOf(k)} stroke="#000" strokeWidth={2} />
          </motion.g>
        ))}
      </g>
      {!compact &&
        s.centroids.map((c, k) => (
          <Handle
            key={`h${k}`}
            title={`Drag centroid ${k + 1}`}
            at={c}
            visible={false}
            onDrag={(px, py) => { c.x = px; c.y = py; s.trails[k].push([px, py]); }}
          />
        ))}
    </>
  );
}

function KMeansVisual({ state: s, data, version, compact }: VisualProps<KMeansState>) {
  return (
    <Plot heatmap={s.centroids.length ? { version, resolution: 90, color: (x, y) => [...PAL_RGB[nearest(s.centroids, x, y)[0] % PAL_RGB.length], 0.12] } : null}>
      <Points data={data} color={(_, i) => (s.assign[i] >= 0 ? colorOf(s.assign[i]) : COLORS.grey)} />
      {!compact && s.current >= 0 && <Ring at={data[s.current]} r={9} color="#fff" />}
      <Centroids s={s} compact={compact} />
    </Plot>
  );
}

export const kmeans: AlgorithmDef<KMeansState> = {
  id: 'kmeans',
  name: 'k-means',
  group: 'Unsupervised',
  kind: 'clustering',
  datasets: ['clusters', 'blobs', 'moons', 'circles'],
  defaultDataset: 'clusters',
  defaultSpeed: 1,
  thumbnailSteps: 200,
  hint: 'Drag the diamonds to choose your own starting centroids, then press play.',
  params: [
    { key: 'k', label: 'k', type: 'range', min: 2, max: 8, step: 1, value: 4 },
    { key: 'init', label: 'Initialization', type: 'select', options: ['k-means++', 'random'], value: 'k-means++' },
    { key: 'animateAssign', label: 'Assign one point per step', type: 'select', options: ['no', 'yes'], value: 'no' },
  ],
  create: (data, params) => createKMeans(data, params as unknown as KMeansParams),
  Visual: KMeansVisual,
  history: { label: 'Inertia', get: (s) => s.history },
  equation: (s, data) => {
    const J = s.assign.some((a) => a >= 0) ? tex(inertia(data, s), 3) : '?';
    return `J = \\sum_{i} \\big\\lVert \\mathbf{x}_i - {\\color{${COLORS.yellow}}\\boldsymbol{\\mu}_{c(i)}} \\big\\rVert^2 = ${J}`;
  },
  metrics: (s) => [
    { label: 'Iteration', value: s.iter },
    { label: 'Phase', value: s.phase },
    { label: 'Points that moved', value: s.changed },
    { label: 'Inertia', value: fmt(s.inertia, 3) },
  ],
  docs: {
    description:
      'k-means splits the data into $k$ clusters by alternating two steps: assign each point to its nearest centroid, then move each centroid to the mean of its points. The tinted background shows the Voronoi regions, the area closest to each centroid. Neither step can increase the inertia $J$, so the algorithm always converges, though it can stop in a local minimum. k-means++ spreads out the starting centroids and usually does better. Try the moons and the circles to see where it fails: k-means assumes round clusters of similar size.',
    steps: [
      'Pick $k$ starting centroids (at random, with k-means++, or by dragging them).',
      'Assign each point to the nearest centroid.',
      'Move each centroid to the mean of its points.',
      'Repeat until no point changes cluster.',
    ],
    complexity: [['Training', 'O(n\\,k\\,d)\\ \\text{per iteration}'], ['Prediction', 'O(k\\,d)'], ['Memory', 'O(n + k\\,d)']],
    pseudocode: [
      'pick k centroids μ₁ ... μₖ',
      'repeat:',
      '    for each point x:',
      '        c(x) ← argminⱼ |x - μⱼ|²',
      '    for each j:',
      '        μⱼ ← mean of the x with c(x) = j',
      'until no c(x) changes',
    ],
    exercises: [
      'Drag all four centroids into one corner before pressing play. Does k-means still find the four clusters?',
      'Run random initialization several times with k = 4. How often does it end in a bad local minimum?',
      'Implement the elbow method: run k = 1..8 and plot the final inertia.',
      'Implement k-medoids, which uses real points as centers and is more robust to outliers.',
    ],
  },
};
