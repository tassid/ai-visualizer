import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring, SMOOTH } from '../../components/plot/marks';
import { createNaiveBayes, joint, posterior, variance, type ClassStats, type NaiveBayesParams, type NaiveBayesState } from './model';

/**
 * 1σ and 2σ ellipses of each class. Because features are assumed independent,
 * the ellipses are always aligned with the axes.
 */
function ClassEllipses({ st, c, smoothing }: { st: ClassStats; c: number; smoothing: number }) {
  const { x, y, size } = usePlot();
  if (st.n < 2) return null;
  const sx = Math.sqrt(variance(st, 0, smoothing)), sy = Math.sqrt(variance(st, 1, smoothing));
  const k = size / 2;
  return (
    <g pointerEvents="none">
      {[1, 2].map((m) => (
        <motion.ellipse
          key={m}
          initial={false}
          animate={{ cx: x(st.mean[0]), cy: y(st.mean[1]), rx: m * sx * k, ry: m * sy * k }}
          transition={SMOOTH}
          fill="none"
          stroke={classColor(c)}
          strokeWidth={m === 1 ? 2 : 1}
          strokeDasharray={m === 2 ? '5 5' : undefined}
        />
      ))}
      <motion.circle initial={false} animate={{ cx: x(st.mean[0]), cy: y(st.mean[1]) }} transition={SMOOTH} r={5} fill={classColor(c)} stroke="#000" strokeWidth={1.5} />
    </g>
  );
}

function NaiveBayesVisual({ state: s, data, version, hover, compact }: VisualProps<NaiveBayesState>) {
  const cur = s.current >= 0 ? data[s.current] : null;
  return (
    <Plot heatmap={{ version, color: (x, y) => binaryColor(posterior(s, x, y)) }}>
      <Points data={data} color={(p) => classColor(p.c)} />
      <ClassEllipses st={s.stats[0]} c={0} smoothing={s.smoothing} />
      <ClassEllipses st={s.stats[1]} c={1} smoothing={s.smoothing} />
      {!compact && <Ring at={hover ?? cur} color={COLORS.yellow} width={2.5} r={hover ? 6 : 11} />}
    </Plot>
  );
}

export const naiveBayes: AlgorithmDef<NaiveBayesState> = {
  id: 'naive-bayes',
  name: 'Gaussian naive Bayes',
  group: 'Linear and probabilistic models',
  kind: 'classification',
  datasets: ['blobs', 'circles', 'linear', 'moons'],
  defaultDataset: 'blobs',
  defaultSpeed: 4,
  thumbnailSteps: 1000,
  hint: 'Hover the plane to see Bayes’ theorem evaluated at that point. Drag points and watch the ellipses follow.',
  params: [{ key: 'smoothing', label: 'Variance smoothing', type: 'select', options: [0, 0.001, 0.01, 0.05], value: 0.001, help: 'Added to every variance so a class with tightly packed points does not divide by almost zero.' }],
  create: (data, params) => createNaiveBayes(data, params as unknown as NaiveBayesParams),
  Visual: NaiveBayesVisual,
  equation: (s, _data, _params, hover) => {
    const A = COLORS.red, B = COLORS.blue;
    const general = `P({\\color{${B}}B}\\mid\\mathbf{x}) = \\frac{P({\\color{${B}}B})\\,p(\\mathbf{x}\\mid {\\color{${B}}B})}{P({\\color{${A}}A})\\,p(\\mathbf{x}\\mid {\\color{${A}}A}) + P({\\color{${B}}B})\\,p(\\mathbf{x}\\mid {\\color{${B}}B})}`;
    if (!hover) return general;
    const a = joint(s, 0, hover.x, hover.y), b = joint(s, 1, hover.x, hover.y);
    return `${general} = \\frac{${tex(b, 3)}}{${tex(a, 3)} + ${tex(b, 3)}} = ${tex(posterior(s, hover.x, hover.y))}`;
  },
  metrics: (s, data) => [
    { label: 'Points seen', value: `${s.seen} of ${data.length}` },
    { label: 'P(A), P(B)', value: `${fmt(s.stats[0].n / Math.max(1, s.seen), 2)}, ${fmt(s.stats[1].n / Math.max(1, s.seen), 2)}` },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => posterior(s, x, y))) },
  ],
  docs: {
    description:
      'Naive Bayes applies Bayes’ theorem under the assumption that features are independent within each class. In the Gaussian version, each class is described by one mean and one variance per feature, drawn here as 1σ and 2σ ellipses aligned with the axes. The animation estimates these numbers one point at a time with Welford’s algorithm, and the background shows the posterior probability of class B. Because the two classes can have different variances, the boundary is quadratic: on the circles it closes into a loop.',
    steps: [
      'For each class, estimate the share $P(k)$, the mean $\\mu_k$ and the variance $\\sigma_k^2$ of each feature.',
      'For a new point, compute $P(k)\\prod_j \\mathcal{N}(x_j;\\ \\mu_{kj},\\ \\sigma_{kj}^2)$ for each class.',
      'Normalize those numbers so they sum to 1. That is $P(k\\mid\\mathbf{x})$.',
      'Predict the class with the higher posterior.',
    ],
    complexity: [['Training', 'O(n\\,d)'], ['Prediction', 'O(k\\,d)'], ['Memory', 'O(k\\,d)']],
    complexityNote: 'k is the number of classes. One of the cheapest models there is, and it trains in a single pass.',
    pseudocode: [
      'for each class k: nₖ ← 0 ; μₖ ← 0 ; Mₖ ← 0',
      'for each (x, y):',
      '    nᵧ ← nᵧ + 1 ;  δ ← x - μᵧ',
      '    μᵧ ← μᵧ + δ / nᵧ',
      '    Mᵧ ← Mᵧ + δ · (x - μᵧ)',
      'σ²ₖ ← Mₖ / nₖ ;  P(k) ← nₖ / n',
      'predict(x) = argmaxₖ  P(k) · Πⱼ N(xⱼ; μₖⱼ, σ²ₖⱼ)',
    ],
    exercises: [
      'Why does naive Bayes do reasonably well on the circles, despite being so simple?',
      'Draw a diagonal cloud of points for one class. Where does the independence assumption break?',
      'Replace the axis-aligned ellipses with a full covariance matrix. That model is quadratic discriminant analysis.',
      'Compute everything in log space (log-likelihoods) to avoid underflow with many features.',
    ],
  },
};
