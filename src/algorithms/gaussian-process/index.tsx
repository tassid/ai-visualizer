import { area, line } from 'd3-shape';
import { motion } from 'framer-motion';
import { COLORS } from '../../core/colors';
import { fmt, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring, SMOOTH } from '../../components/plot/marks';
import { GPML, PRML, paper } from '../books';
import { createGp, XS, type GpParams, type GpState } from './model';

const SAMPLE_COLORS = [COLORS.teal, COLORS.purple, COLORS.gold];

function Posterior({ s }: { s: GpState }) {
  const { x, y } = usePlot();
  const idx = XS.map((_, i) => i);
  const band = area<number>().x((i) => x(XS[i])).y0((i) => y(s.mean[i] - 2 * s.std[i])).y1((i) => y(s.mean[i] + 2 * s.std[i]))(idx) ?? '';
  const mean = line<number>().x((i) => x(XS[i])).y((i) => y(s.mean[i]))(idx) ?? '';
  return (
    <g pointerEvents="none">
      <motion.path initial={false} animate={{ d: band }} transition={SMOOTH} fill={COLORS.blue} fillOpacity={0.16} />
      {s.samples.map((smp, k) => (
        <motion.path key={k} initial={false} animate={{ d: line<number>().x((i) => x(XS[i])).y((i) => y(smp[i]))(idx) ?? '' }} transition={SMOOTH} fill="none" stroke={SAMPLE_COLORS[k]} strokeOpacity={0.6} strokeWidth={1.3} />
      ))}
      <motion.path initial={false} animate={{ d: mean }} transition={SMOOTH} fill="none" stroke={COLORS.yellow} strokeWidth={2.5} />
    </g>
  );
}

function GpVisual({ state: s, data, hover, compact }: VisualProps<GpState>) {
  const shown = new Set(s.order.slice(0, s.seen));
  const near = hover ? Math.round(((hover.x + 1) / 2) * (XS.length - 1)) : -1;
  return (
    <Plot>
      <Posterior s={s} />
      <Points data={data} color={(_, i) => (shown.has(i) ? COLORS.blue : COLORS.grey)} radius={(_, i) => (shown.has(i) ? 5 : 3)} opacity={(_, i) => (shown.has(i) ? 1 : 0.5)} />
      {!compact && near >= 0 && near < XS.length && <Ring at={{ x: XS[near], y: s.mean[near] }} r={6} color={COLORS.yellow} />}
    </Plot>
  );
}

export const gaussianProcess: AlgorithmDef<GpState> = {
  id: 'gaussian-process',
  name: 'Gaussian process regression',
  group: 'Linear, probabilistic and kernel models',
  kind: 'regression',
  datasets: ['few', 'sine', 'line'],
  defaultDataset: 'few',
  defaultSpeed: 1,
  thumbnailSteps: 6,
  hint: 'Change the length scale and press Replay: it sets how far one observation’s influence reaches.',
  params: [
    { key: 'lengthscale', label: 'Length scale ℓ', type: 'select', options: [0.05, 0.1, 0.2, 0.4, 0.8], value: 0.2 },
    { key: 'noise', label: 'Noise σₙ', type: 'select', options: [0.01, 0.05, 0.1, 0.2], value: 0.05 },
    { key: 'signal', label: 'Signal variance σf²', type: 'select', options: [0.1, 0.25, 0.5], value: 0.25 },
  ],
  create: (data, params) => createGp(data, params as unknown as GpParams),
  Visual: GpVisual,
  equation: (s, _d, _p, hover) => {
    const head = `\\mu(x_*) = \\mathbf{k}_*^{\\top}(K + \\sigma_n^2 I)^{-1}\\mathbf{y}, \\quad \\sigma^2(x_*) = k(x_*, x_*) - \\mathbf{k}_*^{\\top}(K + \\sigma_n^2 I)^{-1}\\mathbf{k}_*`;
    if (!hover) return head;
    const i = Math.max(0, Math.min(XS.length - 1, Math.round(((hover.x + 1) / 2) * (XS.length - 1))));
    return `${head} \\qquad {\\color{${COLORS.yellow}}\\mu = ${tex(s.mean[i])}},\\ \\sigma = ${tex(s.std[i])}`;
  },
  metrics: (s, data) => [
    { label: 'Observed', value: `${s.seen} of ${data.length}` },
    { label: 'Log marginal likelihood', value: s.seen ? fmt(s.logMarginal, 2) : 'n/a' },
    { label: 'Widest band (±2σ)', value: fmt(2 * Math.max(...s.std), 3) },
  ],
  docs: {
    description:
      'A Gaussian process puts a probability distribution directly on functions. Before any data, the prior says that function values at nearby inputs are strongly correlated, and the colored curves are random functions drawn from it. Each step reveals one observation and conditions on it: the mean (yellow) passes near the point, the uncertainty band (±2σ) pinches there, and the sampled functions all bend through it. Far from the data the band reopens to the prior. The length scale $\\ell$ decides how far that pinch reaches.',
    steps: [
      'Choose a kernel $k(x, x\')$, here the squared exponential with length scale $\\ell$.',
      'Build the covariance matrix $K$ of the observed inputs and add the noise $\\sigma_n^2$ on the diagonal.',
      'For each test input, compute the cross-covariances $\\mathbf{k}_*$ with the observed inputs.',
      'The posterior mean and variance follow from one Cholesky factorization of $K + \\sigma_n^2 I$.',
    ],
    complexity: [['Training', 'O(n^3)'], ['Prediction (mean)', 'O(n)\\ \\text{per point}'], ['Prediction (variance)', 'O(n^2)\\ \\text{per point}'], ['Memory', 'O(n^2)']],
    complexityNote: 'The cubic cost of the factorization limits exact GPs to a few thousand points; sparse and inducing-point approximations go further.',
    pseudocode: [
      'K ← [k(xᵢ, xⱼ)] + σₙ² I',
      'for m = 1 to n (reveal one observation at a time):',
      '    L ← cholesky(K[1..m, 1..m])',
      '    α ← Lᵀ \\ (L \\ y)',
      '    μ(x*) ← k*ᵀ α ;  v ← L \\ k*',
      '    σ²(x*) ← k(x*, x*) - vᵀ v',
    ],
    exercises: [
      'Set ℓ = 0.05 and ℓ = 0.8 on the same data. Which one overfits and which one ignores the data?',
      'Raise the noise σₙ. Does the mean still pass through the points?',
      'The log marginal likelihood is how GPs choose ℓ. Try all values of ℓ and find the one that maximizes it.',
      'Replace the kernel with a periodic one, k = σ² exp(-2 sin²(π|a-b|/p) / ℓ²), and fit the sine curve.',
    ],
    theory: [
      'A Gaussian process $f \\sim \\mathcal{GP}(m, k)$ is a collection of random variables, one per input, such that any finite subset is jointly Gaussian with mean $m(x)$ and covariance $k(x, x\')$. The kernel encodes assumptions: the squared exponential $k(x,x\') = \\sigma_f^2\\exp(-(x-x\')^2/2\\ell^2)$ gives infinitely smooth functions that vary on a scale of about $\\ell$.',
      'With observations $y_i = f(x_i) + \\varepsilon_i$, $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma_n^2)$, the training targets and a test value are jointly Gaussian:',
      '$$\\begin{bmatrix}\\mathbf{y}\\\\ f_*\\end{bmatrix} \\sim \\mathcal{N}\\!\\left(\\mathbf{0},\\ \\begin{bmatrix}K + \\sigma_n^2 I & \\mathbf{k}_*\\\\ \\mathbf{k}_*^\\top & k(x_*, x_*)\\end{bmatrix}\\right).$$',
      'Conditioning a Gaussian on part of its coordinates gives another Gaussian, which yields the posterior mean and variance in the equation above. Note that the variance does not depend on $\\mathbf{y}$ at all, only on where the inputs are. The mean can be written $\\mu(x_*) = \\sum_i\\alpha_i k(x_i, x_*)$, a weighted sum of kernels just like a kernel SVM; in fact GP regression and kernel ridge regression give the same mean.',
      'Hyperparameters such as $\\ell$ and $\\sigma_n$ are usually set by maximizing the log marginal likelihood',
      '$$\\log p(\\mathbf{y}\\mid X) = -\\tfrac12\\mathbf{y}^\\top(K+\\sigma_n^2I)^{-1}\\mathbf{y} - \\tfrac12\\log|K+\\sigma_n^2I| - \\tfrac{n}{2}\\log 2\\pi,$$',
      'whose first term rewards fitting the data and second term penalizes complexity, an automatic Occam’s razor. Numerically everything goes through the Cholesky factor $L$: solves are two triangular substitutions and $\\log|K + \\sigma_n^2 I| = 2\\sum_i\\log L_{ii}$.',
    ],
    references: [
      GPML('Chapter 2, Regression, Chapter 4, Covariance functions, and Chapter 5, Model selection'),
      PRML('Section 6.4, Gaussian processes'),
      paper('Görtler, Kehlbeck and Deussen, A visual exploration of Gaussian processes, Distill (2019)', 'https://distill.pub/2019/visual-exploration-gaussian-processes/'),
    ],
  },
};
