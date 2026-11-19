import { theory } from './theory';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct, tex, texTerm } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { DecisionLine, Points, Ring } from '../../components/plot/marks';
import { LinearHandles } from '../../components/plot/LinearHandles';
import { createLogistic, logLoss, probability, type LogisticParams, type LogisticState } from './model';

/** ln 3 is where σ(z) = 0.75. The dashed lines mark p = 0.25 and p = 0.75. */
const LN3 = Math.log(3);
const Y = COLORS.yellow;

function LogisticVisual({ state: s, data, version, hover, compact }: VisualProps<LogisticState>) {
  const [w1, w2] = s.w;
  return (
    <Plot heatmap={{ version, color: (x, y) => binaryColor(probability(s, x, y)) }}>
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={LN3} color="rgba(255,255,255,.45)" width={1.2} dash="6 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={-LN3} color="rgba(255,255,255,.45)" width={1.2} dash="6 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} />
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && hover && <Ring at={hover} r={6} color={Y} />}
      {!compact && <LinearHandles s={s} mode="vector" scale={0.06} />}
    </Plot>
  );
}

export const logisticRegression: AlgorithmDef<LogisticState> = {
  id: 'logistic-regression',
  name: 'Logistic regression',
  group: 'Linear, probabilistic and kernel models',
  kind: 'classification',
  datasets: ['blobs', 'linear', 'moons', 'xor'],
  defaultDataset: 'blobs',
  defaultSpeed: 4,
  thumbnailSteps: 400,
  hint: 'Drag the yellow dot to slide the line. The arrow length is the steepness |w|: drag its tip to change it.',
  params: [
    { key: 'lr', label: 'Learning rate η', type: 'select', options: [0.05, 0.1, 0.5, 1, 3], value: 0.5 },
    { key: 'l2', label: 'L2 penalty λ', type: 'select', options: [0, 0.001, 0.01, 0.1], value: 0 },
    { key: 'iters', label: 'Iterations', type: 'range', min: 50, max: 2000, step: 50, value: 400 },
  ],
  create: (data, params) => createLogistic(data, params as unknown as LogisticParams),
  Visual: LogisticVisual,
  history: { label: 'Loss', get: (s) => s.history },
  equation: (s, data, params, hover) => {
    const model = `p(\\mathbf{x}) = \\sigma\\!\\left({\\color{${Y}}${tex(s.w[0])}}\\,x_1 ${texTerm(s.w[1], 'x_2', 2, Y)} ${texTerm(s.b, '', 2, Y)}\\right)`;
    if (hover) return `${model} = ${tex(probability(s, hover.x, hover.y))}`;
    return `${model} \\qquad \\mathcal{L} = ${tex(logLoss(s, data, Number(params.l2)), 4)}`;
  },
  metrics: (s, data) => [
    { label: 'Iteration', value: s.it },
    { label: 'Loss', value: fmt(s.loss, 4) },
    { label: 'Gradient length', value: fmt(s.gradNorm, 5) },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => probability(s, x, y))) },
  ],
  docs: {
    ...theory,
    description:
      'Despite the name, logistic regression is a classifier. It passes a linear combination of the inputs through the sigmoid, $\\sigma(z) = 1/(1+e^{-z})$, which gives a probability between 0 and 1. The weights are fit by gradient descent on the cross-entropy. The boundary is the line where $p = 0.5$, and the dashed lines mark $p = 0.25$ and $p = 0.75$: the closer they are, the more confident the model. The loss is convex, so there is a single minimum and gradient descent always gets there.',
    steps: [
      'Compute $z = \\mathbf{w}\\cdot\\mathbf{x} + b$ for each point.',
      'Turn it into a probability, $p = \\sigma(z)$.',
      'The cross-entropy gradient is simply $(p - y)\\,\\mathbf{x}$.',
      'Move $\\mathbf{w}$ and $b$ against the gradient.',
    ],
    complexity: [['Training', 'O(n\\,d)\\ \\text{per iteration}'], ['Prediction', 'O(d)'], ['Memory', 'O(d)']],
    pseudocode: [
      'w ← random ; b ← 0',
      'repeat T times:',
      '    for each (x, y): p ← σ(w·x + b)',
      '    ∂w ← (1/n) Σ (p - y) · x  + λ·w',
      '    ∂b ← (1/n) Σ (p - y)',
      '    w ← w - η · ∂w ;  b ← b - η · ∂b',
      '    if |∇| < ε: stop',
    ],
    exercises: [
      'On separable data, watch |w| (the arrow) grow forever. Then set λ = 0.01 and compare.',
      'Set η = 3 and watch the loss chart. When does the descent start to oscillate?',
      'Derive on paper why ∂loss/∂z = p - y when the loss is cross-entropy and the output is a sigmoid.',
      'Add x², y² and x·y as inputs. Can logistic regression then solve circles?',
    ],
  },
};
