import { motion } from 'framer-motion';
import { CLUSTER_PALETTE, COLORS } from '../../core/colors';
import { fmt, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { ESL, paper } from '../books';
import { createMs, density, shift, type MsParams, type MsState } from './model';

/** Each climber is a dot that glides to its next position, trailing a line back to its point. */
function Climbers({ s, data }: { s: MsState; data: VisualProps<MsState>['data'] }) {
  const { x, y } = usePlot();
  const done = s.peaks.length > 0;
  return (
    <g pointerEvents="none">
      {data.map((p, i) => <line key={`l${i}`} x1={x(p.x)} y1={y(p.y)} x2={x(s.modes[i].x)} y2={y(s.modes[i].y)} stroke="#fff" strokeOpacity={0.12} />)}
      {s.modes.map((m, i) => (
        <motion.circle key={i} initial={false} animate={{ cx: x(m.x), cy: y(m.y) }} transition={{ duration: 0.3, ease: [0.45, 0, 0.55, 1] }} r={3} fill={done ? CLUSTER_PALETTE[s.labels[i] % CLUSTER_PALETTE.length] : COLORS.yellow} />
      ))}
    </g>
  );
}

function MsVisual({ state: s, data, params, version, hover, compact }: VisualProps<MsState>) {
  const h = Number(params.bandwidth);
  const peak = data.length ? Math.max(...data.map((p) => density(data, p.x, p.y, h))) : 1;
  const next = hover && data.length ? shift(data, hover.x, hover.y, h) : null;
  return (
    <Plot heatmap={{ version, resolution: 40, color: (x, y) => [88, 196, 221, (0.35 * density(data, x, y, h)) / peak] }}>
      <Points data={data} color={(_, i) => (s.peaks.length ? CLUSTER_PALETTE[s.labels[i] % CLUSTER_PALETTE.length] : COLORS.grey)} radius={3.6} opacity={() => 0.55} />
      {!compact && <Climbers s={s} data={data} />}
      {!compact && hover && next && (
        <>
          <Ring at={hover} worldRadius={h} color="rgba(255,255,255,.6)" width={1} dash="4 4" />
          <ArrowTo from={hover} to={next} />
        </>
      )}
    </Plot>
  );
}

function ArrowTo({ from, to }: { from: { x: number; y: number }; to: { x: number; y: number } }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      <line x1={x(from.x)} y1={y(from.y)} x2={x(to.x)} y2={y(to.y)} stroke={COLORS.yellow} strokeWidth={2} />
      <circle cx={x(to.x)} cy={y(to.y)} r={4} fill={COLORS.yellow} />
    </g>
  );
}

export const meanShift: AlgorithmDef<MsState> = {
  id: 'mean-shift',
  name: 'Mean shift',
  group: 'Clustering',
  kind: 'clustering',
  datasets: ['clusters', 'blobs', 'moons', 'circles'],
  defaultDataset: 'clusters',
  defaultSpeed: 2,
  thumbnailSteps: 60,
  hint: 'Hover the plane: the yellow arrow is the mean shift step from that point, always uphill on the density.',
  params: [
    { key: 'bandwidth', label: 'Bandwidth h', type: 'select', options: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], value: 0.15, help: 'The kernel width. It decides how many peaks the density has, and so how many clusters.' },
    { key: 'iters', label: 'Max iterations', type: 'range', min: 10, max: 200, step: 10, value: 100 },
  ],
  create: (data, params) => createMs(data, params as unknown as MsParams),
  Visual: MsVisual,
  equation: (_s, data, params, hover) => {
    const head = `\\mathbf{m}(\\mathbf{x}) = \\frac{\\sum_j K(\\mathbf{x}_j - \\mathbf{x})\\,\\mathbf{x}_j}{\\sum_j K(\\mathbf{x}_j - \\mathbf{x})}, \\quad K(\\mathbf{u}) = e^{-\\lVert\\mathbf{u}\\rVert^2/2h^2}`;
    if (!hover || !data.length) return head;
    const m = shift(data, hover.x, hover.y, Number(params.bandwidth));
    return `${head} \\qquad {\\color{${COLORS.yellow}}\\mathbf{m}} = (${tex(m.x)},\\ ${tex(m.y)})`;
  },
  metrics: (s) => [
    { label: 'Iteration', value: s.iter },
    { label: 'Largest move', value: fmt(s.maxShift, 5) },
    { label: 'Peaks found', value: s.peaks.length || 'n/a' },
  ],
  docs: {
    description:
      'Mean shift treats the data as samples from a density and lets every point climb to the nearest peak. The blue glow is the kernel density estimate. At each iteration every yellow climber moves to the weighted mean of the data around it, which is always a step uphill. Climbers that reach the same peak belong to the same cluster. The number of clusters comes out of the data; what you choose is the bandwidth $h$, the scale at which the density is smoothed.',
    steps: [
      'Place a climber on every data point.',
      'Move each climber to $\\mathbf{m}(\\mathbf{x})$, the kernel-weighted mean of the data around it.',
      'Repeat until no climber moves more than a tiny amount.',
      'Group the climbers that ended at the same peak.',
    ],
    complexity: [['Training', 'O(T\\,n^2)'], ['Prediction', 'O(T\\,n)\\ \\text{per point}'], ['Memory', 'O(n)']],
    pseudocode: [
      'yᵢ ← xᵢ for every point',
      'repeat:',
      '    for each i:',
      '        yᵢ ← Σⱼ K(xⱼ - yᵢ) xⱼ / Σⱼ K(xⱼ - yᵢ)',
      'until no yᵢ moves',
      'points whose yᵢ coincide form one cluster',
    ],
    exercises: [
      'Change the bandwidth from 0.05 to 0.5. How does the number of peaks change? Where is it four?',
      'On the moons, can any bandwidth separate them? Compare with DBSCAN.',
      'Show that m(x) - x is proportional to the gradient of the density estimate divided by the density.',
      'Speed it up: only update climbers that have not converged, and use a grid to find neighbors.',
    ],
    theory: [
      'The kernel density estimate with a Gaussian kernel of bandwidth $h$ is $\\hat f(\\mathbf{x}) \\propto \\sum_j \\exp(-\\lVert\\mathbf{x}-\\mathbf{x}_j\\rVert^2/2h^2)$. Its gradient is',
      '$$\\nabla\\hat f(\\mathbf{x}) \\propto \\frac{1}{h^2}\\sum_j K(\\mathbf{x}_j-\\mathbf{x})\\,(\\mathbf{x}_j-\\mathbf{x}) = \\frac{1}{h^2}\\Big(\\sum_j K_j\\Big)\\big(\\mathbf{m}(\\mathbf{x})-\\mathbf{x}\\big),$$',
      'so the mean shift vector $\\mathbf{m}(\\mathbf{x}) - \\mathbf{x}$ points along the gradient, scaled by $h^2/\\hat f(\\mathbf{x})$ up to a constant. The step is large where the density is low and small near a peak, an adaptive step size that needs no learning rate. Comaniciu and Meer proved that the sequence of density values is increasing and converges, so every climber stops at a stationary point of $\\hat f$, in practice a local maximum.',
      'Clustering by modes needs no assumption about the shape of the clusters and no number of clusters. The bandwidth plays the role that $k$ plays in k-means: a small $h$ gives a bumpy density with many peaks, a large $h$ merges them. Rules such as Silverman’s, $h \\approx 1.06\\,\\hat\\sigma\\,n^{-1/5}$ in one dimension, give a starting point. Beyond clustering, mean shift is used for image segmentation and for tracking objects in video, where the density is a color histogram.',
    ],
    references: [
      paper('Fukunaga and Hostetler, The estimation of the gradient of a density function, with applications in pattern recognition, IEEE Transactions on Information Theory 21 (1975)'),
      paper('Cheng, Mean shift, mode seeking, and clustering, IEEE Transactions on Pattern Analysis and Machine Intelligence 17 (1995)'),
      paper('Comaniciu and Meer, Mean shift: a robust approach toward feature space analysis, IEEE Transactions on Pattern Analysis and Machine Intelligence 24 (2002)'),
      ESL('Section 6.6, Kernel density estimation and classification'),
    ],
  },
};
