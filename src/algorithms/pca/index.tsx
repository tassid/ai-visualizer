import { theory } from './theory';
import { scaleLinear } from 'd3-scale';
import { line } from 'd3-shape';
import { motion } from 'framer-motion';
import { COLORS } from '../../core/colors';
import { fmt, pct, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Arrow, Handle, Points } from '../../components/plot/marks';
import { createPca, projectedVariance, type PcaParams, type PcaState } from './model';

const Y = COLORS.yellow;
const deg = (r: number) => (r * 180) / Math.PI;

/** The turning line through the mean, with each point's projection onto it. */
function Projections({ s, data, show }: { s: PcaState; data: VisualProps<PcaState>['data']; show: boolean }) {
  const { x, y } = usePlot();
  const co = Math.cos(s.angle), si = Math.sin(s.angle);
  return (
    <g pointerEvents="none">
      <motion.line initial={false} animate={{ x1: x(s.mx - co * 2), y1: y(s.my - si * 2), x2: x(s.mx + co * 2), y2: y(s.my + si * 2) }} transition={{ duration: 0.08 }} stroke={Y} strokeWidth={2} />
      {show &&
        data.map((p, i) => {
          const t = (p.x - s.mx) * co + (p.y - s.my) * si; // coordinate along the line
          const qx = s.mx + t * co, qy = s.my + t * si;
          return (
            <g key={i}>
              <line x1={x(p.x)} y1={y(p.y)} x2={x(qx)} y2={y(qy)} stroke={Y} strokeOpacity={0.18} />
              <circle cx={x(qx)} cy={y(qy)} r={2.4} fill={Y} />
            </g>
          );
        })}
    </g>
  );
}

/** Final components, with lengths of two standard deviations. */
function Components({ s }: { s: PcaState }) {
  const s1 = 2 * Math.sqrt(s.l1), s2 = 2 * Math.sqrt(s.l2);
  const c = Math.cos(s.pc1), n = Math.sin(s.pc1);
  return (
    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.35 }}>
      <Arrow from={[s.mx, s.my]} to={[s.mx + c * s1, s.my + n * s1]} color={Y} width={3} />
      <Arrow from={[s.mx, s.my]} to={[s.mx - n * s2, s.my + c * s2]} color={COLORS.teal} width={3} />
    </motion.g>
  );
}

function CenterDot({ s }: { s: PcaState }) {
  const { x, y } = usePlot();
  return <circle cx={x(s.mx)} cy={y(s.my)} r={4} fill="#fff" pointerEvents="none" />;
}

function PcaVisual({ state: s, data, params, compact }: VisualProps<PcaState>) {
  const r = 0.55;
  return (
    <Plot>
      {!s.finished && <Projections s={s} data={data} show={params.projections === 'on' && !compact} />}
      <Points data={data} color={() => COLORS.blue} radius={4} />
      {s.finished && <Components s={s} />}
      <CenterDot s={s} />
      {!compact && !s.finished && (
        <Handle
          title="Drag to turn the line"
          at={{ x: s.mx + Math.cos(s.angle) * r, y: s.my + Math.sin(s.angle) * r }}
          onDrag={(px, py) => {
            let a = Math.atan2(py - s.my, px - s.mx);
            if (a < 0) a += Math.PI;
            if (a >= Math.PI) a -= Math.PI;
            s.angle = a;
          }}
        />
      )}
    </Plot>
  );
}

/** Projected variance against angle, with D3. The peak is PC1. */
function VarianceCurve({ state: s }: VisualProps<PcaState>) {
  const W = 560, H = 200, pad = 32;
  const sx = scaleLinear().domain([0, Math.PI]).range([pad, W - pad]);
  const sy = scaleLinear().domain([0, s.l1 * 1.1 || 1]).range([H - pad, pad / 2]);
  const d = line<[number, number]>().x((p) => sx(p[0])).y((p) => sy(p[1]))(s.curve) ?? '';
  const now = projectedVariance(s.cov, s.angle);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="aux-svg">
      <line x1={pad} x2={W - pad} y1={H - pad} y2={H - pad} className="chart-base" />
      <line x1={sx(s.pc1)} x2={sx(s.pc1)} y1={pad / 2} y2={H - pad} stroke={Y} strokeOpacity={0.5} strokeDasharray="4 4" />
      <path d={d} fill="none" stroke={COLORS.blue} strokeWidth={2.2} />
      <motion.circle initial={false} animate={{ cx: sx(s.angle), cy: sy(now) }} transition={{ duration: 0.08 }} r={4.5} fill={Y} />
      {[0, 45, 90, 135, 180].map((v) => (
        <text key={v} x={sx((v * Math.PI) / 180)} y={H - 10} textAnchor="middle" className="node-label muted">{v}°</text>
      ))}
    </svg>
  );
}

export const pca: AlgorithmDef<PcaState> = {
  id: 'pca',
  name: 'Principal component analysis',
  group: 'Dimensionality reduction and maps',
  kind: 'reduction',
  datasets: ['elongated', 'clusters', 'blobs'],
  defaultDataset: 'elongated',
  defaultSpeed: 5,
  thumbnailSteps: 60,
  hint: 'Drag the yellow dot to turn the line yourself and hunt for the largest variance.',
  params: [
    { key: 'stepDeg', label: 'Angle step', type: 'select', options: [0.5, 1, 1.5, 3, 5], value: 1.5 },
    { key: 'projections', label: 'Projections', type: 'select', options: ['on', 'off'], value: 'on' },
  ],
  create: (data, params) => createPca(data, params as unknown as PcaParams),
  Visual: PcaVisual,
  aux: { title: 'Projected variance by angle', Component: VarianceCurve },
  equation: (s) => {
    const [a, b, c] = s.cov;
    const th = s.angle;
    return `\\operatorname{Var}(\\theta) = {\\color{${Y}}\\mathbf{u}}^{\\!\\top} \\Sigma\\, {\\color{${Y}}\\mathbf{u}} = \\begin{bmatrix} ${tex(Math.cos(th))} & ${tex(Math.sin(th))} \\end{bmatrix} \\begin{bmatrix} ${tex(a, 3)} & ${tex(b, 3)} \\\\ ${tex(b, 3)} & ${tex(c, 3)} \\end{bmatrix} \\begin{bmatrix} ${tex(Math.cos(th))} \\\\ ${tex(Math.sin(th))} \\end{bmatrix} = ${tex(projectedVariance(s.cov, th), 4)}`;
  },
  metrics: (s) => [
    { label: 'Angle', value: `${deg(s.angle).toFixed(1)}°` },
    { label: 'Projected variance', value: fmt(projectedVariance(s.cov, s.angle), 4) },
    { label: 'λ₁', value: fmt(s.l1, 4) },
    { label: 'λ₂', value: fmt(s.l2, 4) },
    { label: 'Explained by PC1', value: pct(s.l1 / (s.l1 + s.l2 || 1)) },
  ],
  docs: {
    ...theory,
    description:
      'Principal component analysis finds the directions in which the data varies most. The animation turns a line through the mean and projects every point onto it: the spread of the projections is the variance in that direction, and the chart below records it for every angle. The direction of largest variance is the first principal component (yellow), and the second (teal) is perpendicular to it, each drawn two standard deviations long. In practice PCA gets there in one shot from the eigenvectors of the covariance matrix $\\Sigma$. It is used to reduce dimensions, compress data and plot datasets with many features.',
    steps: [
      'Center the data by subtracting the mean.',
      'Compute the covariance matrix $\\Sigma$.',
      'The eigenvectors of $\\Sigma$ are the principal directions; the eigenvalues are the variance along each.',
      'Project the data onto the first $k$ components to reduce the dimension.',
    ],
    complexity: [['Training (eigendecomposition)', 'O(n\\,d^2 + d^3)'], ['Training (truncated SVD)', 'O(n\\,d\\,k)'], ['Projection', 'O(d\\,k)\\ \\text{per point}'], ['Memory', 'O(d^2)']],
    pseudocode: [
      'μ ← mean of the data ;  X ← data - μ',
      'Σ ← (1/n) · Xᵀ X',
      'for θ from 0° to 180°:',
      '    u ← (cos θ, sin θ)',
      '    variance(θ) ← uᵀ Σ u',
      'PC1 ← eigenvector of Σ with the largest eigenvalue',
    ],
    exercises: [
      'Why is turning from 0° to 180° enough, instead of a full 360°?',
      'Show that uᵀ Σ u equals the variance of the projections u · xᵢ.',
      'Implement power iteration to find PC1 without the closed form.',
      'Apply PCA to a real high-dimensional dataset (for example 8x8 digits) and plot the first two components.',
    ],
  },
};
