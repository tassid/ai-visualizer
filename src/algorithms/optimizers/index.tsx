import { line } from 'd3-shape';
import { scaleSequential } from 'd3-scale';
import { interpolateViridis } from 'd3-scale-chromatic';
import { motion } from 'framer-motion';
import { useMemo } from 'react';
import { COLORS } from '../../core/colors';
import { fmt } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Handle } from '../../components/plot/marks';
import { ContourLines } from '../../components/plot/shapes';
import { DLBOOK, paper } from '../books';
import { createOpt, resetRunners, SURFACES, type OptParams, type OptState, type Surface } from './model';

const RUNNER_COLORS: Record<string, string> = { SGD: COLORS.red, Momentum: COLORS.yellow, RMSProp: COLORS.green, Adam: COLORS.blue };

/** Contour lines at levels spaced evenly in log scale above the minimum. */
function Landscape({ surface, compact }: { surface: Surface; compact?: boolean }) {
  const S = SURFACES[surface];
  const levels = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const v = S.f(-1 + i / 30, -1 + j / 30); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const color = scaleSequential(interpolateViridis).domain([0, 1]);
    return Array.from({ length: 14 }, (_, k) => {
      const t = k / 13;
      return { value: lo + (hi - lo) * (Math.exp(t * 5) - 1) / (Math.exp(5) - 1) + 1e-4, color: color(1 - t), width: 1 };
    });
  }, [S]);
  return <ContourLines f={S.f} levels={levels} resolution={compact ? 48 : 72} />;
}

function Runners({ s }: { s: OptState }) {
  const { x, y } = usePlot();
  const path = line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1]));
  return (
    <g pointerEvents="none">
      {s.runners.map((r) => (
        <g key={r.name}>
          <path d={path(r.path) ?? ''} fill="none" stroke={RUNNER_COLORS[r.name]} strokeWidth={2} strokeOpacity={0.9} />
          <motion.circle initial={false} animate={{ cx: x(r.x), cy: y(r.y) }} transition={{ duration: 0.08 }} r={5.5} fill={RUNNER_COLORS[r.name]} stroke="#000" strokeWidth={1.5} />
        </g>
      ))}
    </g>
  );
}

function OptVisual({ state: s, compact }: VisualProps<OptState>) {
  return (
    <Plot>
      <Landscape surface={s.surface} compact={compact} />
      <Runners s={s} />
      {!compact && (
        <Handle
          title="Drag to choose the starting point"
          at={s.start}
          color="#fff"
          onDrag={(px, py) => { s.start = { x: px, y: py }; resetRunners(s); }}
        />
      )}
    </Plot>
  );
}

export const optimizers: AlgorithmDef<OptState> = {
  id: 'optimizers',
  name: 'Gradient descent optimizers',
  group: 'Optimization',
  kind: 'optimization',
  noData: true,
  datasets: [],
  defaultDataset: 'line',
  defaultSpeed: 5,
  thumbnailSteps: 150,
  hint: 'Drag the white dot to restart the race from anywhere. Red is SGD, yellow Momentum, green RMSProp, blue Adam.',
  params: [
    { key: 'surface', label: 'Surface', type: 'select', options: ['narrow valley', 'Rosenbrock', 'two minima'], value: 'narrow valley' },
    { key: 'lr', label: 'Learning rate η', type: 'select', options: [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.16], value: 0.03 },
    { key: 'iters', label: 'Steps', type: 'range', min: 50, max: 1500, step: 50, value: 400 },
  ],
  create: (data, params) => createOpt(data, params as unknown as OptParams),
  Visual: OptVisual,
  equation: () =>
    `{\\color{${COLORS.red}}\\theta \\leftarrow \\theta - \\eta\\,g} \\qquad {\\color{${COLORS.yellow}}v \\leftarrow \\beta v + g,\\ \\theta \\leftarrow \\theta - \\eta v} \\qquad {\\color{${COLORS.blue}}\\theta \\leftarrow \\theta - \\eta\\,\\hat m/(\\sqrt{\\hat v} + \\epsilon)}`,
  metrics: (s) => [
    { label: 'Step', value: s.t },
    ...s.runners.map((r) => ({ label: `f, ${r.name}`, value: fmt(SURFACES[s.surface].f(r.x, r.y), 4) })),
  ],
  docs: {
    description:
      'Almost every model on this site is trained by some form of gradient descent, and this page isolates the optimizer. Four of them start from the same point on the same surface, drawn as contour lines like a topographic map. Plain gradient descent (red) steps straight downhill. Momentum (yellow) accumulates velocity like a ball rolling. RMSProp (green) divides each coordinate’s step by a running average of its recent gradient size. Adam (blue) combines both ideas. On the narrow valley, watch plain gradient descent zigzag across the steep direction while the adaptive methods head along the floor.',
    steps: [
      'All four start at the white point.',
      'At each step each optimizer evaluates the gradient where it stands.',
      'It turns that gradient into a step using its own rule and memory.',
      'Compare the paths and the final values of $f$.',
    ],
    complexity: [['Per step (SGD)', 'O(p)'], ['Per step (Momentum, RMSProp)', 'O(p)'], ['Per step (Adam)', 'O(p)'], ['Extra memory (Adam)', 'O(2p)']],
    complexityNote: 'p is the number of parameters. All four cost the same order per step; the difference is how many steps they need.',
    pseudocode: [
      'for t = 1 to T:',
      '    g ← ∇f(θ)',
      '    SGD:       θ ← θ - η g',
      '    Momentum:  v ← β v + g ;  θ ← θ - η v',
      '    RMSProp:   s ← ρ s + (1-ρ) g² ;  θ ← θ - η g / (√s + ε)',
      '    Adam:      m, v ← moments of g ;  θ ← θ - η m̂ / (√v̂ + ε)',
    ],
    exercises: [
      'On the narrow valley, find the largest learning rate at which SGD still converges. Compare it with 2 divided by the largest curvature (12).',
      'On the two minima surface, which optimizers escape the shallow minimum on the right? Change the starting point and try again.',
      'On Rosenbrock, all of them slow down on the valley floor. Why is the gradient small there even far from the minimum?',
      'Add Nesterov momentum, which evaluates the gradient at θ - ηβv instead of θ, and compare it with plain momentum.',
    ],
    theory: [
      'Near a minimum, a smooth function looks like a quadratic $f(\\theta) \\approx \\frac12(\\theta-\\theta^*)^\\top H(\\theta-\\theta^*)$, with Hessian $H$. In the eigenbasis of $H$ gradient descent decouples: the error along eigenvector $i$ is multiplied by $(1 - \\eta\\lambda_i)$ each step. It converges only if $\\eta < 2/\\lambda_\\text{max}$, and then the slowest direction shrinks by $1 - \\eta\\lambda_\\text{min}$, so the number of steps grows with the condition number $\\kappa = \\lambda_\\text{max}/\\lambda_\\text{min}$. The narrow valley has $\\kappa = 12$: a step size small enough for the steep direction is far too timid for the flat one, and anything larger makes it zigzag.',
      'Momentum (Polyak’s heavy ball) keeps a velocity $v \\leftarrow \\beta v + g$. Oscillating components of the gradient cancel in $v$ while consistent ones add up, and on a quadratic with tuned $\\eta$ and $\\beta$ the number of steps scales with $\\sqrt{\\kappa}$ instead of $\\kappa$. Nesterov’s variant evaluates the gradient at the look-ahead point and achieves the optimal rate for smooth convex functions.',
      'Adaptive methods rescale each coordinate. RMSProp divides by $\\sqrt{s}$, a running root mean square of recent gradients, which approximates a diagonal preconditioner: steep coordinates get smaller steps, flat ones larger. Adam adds momentum on top and corrects both running averages for their initialization at zero,',
      '$$\\hat m_t = \\frac{m_t}{1-\\beta_1^t}, \\qquad \\hat v_t = \\frac{v_t}{1-\\beta_2^t}, \\qquad \\theta \\leftarrow \\theta - \\eta\\,\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}.$$',
      'Since $\\hat m/\\sqrt{\\hat v}$ is roughly bounded by 1 per coordinate, Adam’s steps have size about $\\eta$ regardless of the gradient’s scale, which makes it robust to badly scaled problems but also means it does not slow down by itself near a minimum. In deep learning the gradient is a noisy minibatch estimate; that noise helps escape sharp minima and saddle points, which in high dimensions are far more common than poor local minima.',
    ],
    references: [
      DLBOOK('Chapter 8, Optimization for training deep models (Sections 8.3 and 8.5)'),
      paper('Ruder, An overview of gradient descent optimization algorithms (2016)', 'https://arxiv.org/abs/1609.04747'),
      paper('Polyak, Some methods of speeding up the convergence of iteration methods, USSR Computational Mathematics and Mathematical Physics 4 (1964)'),
      paper('Kingma and Ba, Adam: a method for stochastic optimization, ICLR (2015)', 'https://arxiv.org/abs/1412.6980'),
      paper('Goh, Why momentum really works, Distill (2017)', 'https://distill.pub/2017/momentum/'),
    ],
  },
};
