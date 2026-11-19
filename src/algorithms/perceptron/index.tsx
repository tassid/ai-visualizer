import { theory } from './theory';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct, sigmoid, tex, texTerm } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { DecisionLine, Points, Ring } from '../../components/plot/marks';
import { LinearHandles } from '../../components/plot/LinearHandles';
import { createPerceptron, predict, score, type PerceptronParams, type PerceptronState } from './model';

const Y = COLORS.yellow, B = COLORS.blue;

function PerceptronVisual({ state: s, data, version, compact }: VisualProps<PerceptronState>) {
  const norm = Math.hypot(...s.w) || 1;
  const cur = s.current >= 0 ? data[s.current] : null;
  return (
    <Plot heatmap={{ version, color: (x, y) => binaryColor(sigmoid((16 * score(s, x, y)) / norm)) }}>
      <DecisionLine w1={s.w[0]} w2={s.w[1]} b={s.b} />
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && cur && <Ring at={cur} color={s.updated ? Y : '#fff'} width={s.updated ? 3 : 1.5} />}
      {!compact && <LinearHandles s={s} mode="direction" />}
    </Plot>
  );
}

export const perceptron: AlgorithmDef<PerceptronState> = {
  id: 'perceptron',
  name: 'Perceptron',
  group: 'Neural networks',
  kind: 'classification',
  datasets: ['linear', 'blobs', 'xor', 'circles'],
  defaultDataset: 'linear',
  defaultSpeed: 4,
  thumbnailSteps: 700,
  hint: 'Drag the yellow dot to slide the line and the arrow tip to turn it.',
  params: [
    { key: 'lr', label: 'Learning rate η', type: 'select', options: [0.01, 0.05, 0.1, 0.5, 1], value: 0.1 },
    { key: 'epochs', label: 'Max epochs', type: 'range', min: 5, max: 100, step: 5, value: 40 },
  ],
  create: (data, params) => createPerceptron(data, params as unknown as PerceptronParams),
  Visual: PerceptronVisual,
  history: { label: 'Mistakes per epoch', get: (s) => s.history },
  equation: (s, data) => {
    const f = `f(\\mathbf{x}) = {\\color{${Y}}${tex(s.w[0])}}\\,x_1 ${texTerm(s.w[1], 'x_2', 2, Y)} ${texTerm(s.b, '', 2, Y)}`;
    const p = s.current >= 0 ? data[s.current] : null;
    if (!p) return f;
    const y = p.c ? 1 : -1;
    const v = y * score(s, p.x, p.y);
    return `${f} \\qquad y\\,f({\\color{${B}}\\mathbf{x}}) = ${tex(v)} ${v > 0 ? '> 0' : '\\le 0'}`;
  },
  metrics: (s, data) => [
    { label: 'Epoch', value: s.epoch + 1 },
    { label: 'Point', value: `${s.seen} of ${data.length}` },
    { label: 'Mistakes this epoch', value: s.errors },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => predict(s, x, y))) },
    { label: 'b', value: fmt(s.b, 2) },
  ],
  docs: {
    ...theory,
    description:
      'The perceptron, proposed by Frank Rosenblatt in 1958, is the simplest artificial neuron. It takes a weighted sum of its inputs, $f(\\mathbf{x}) = \\mathbf{w}\\cdot\\mathbf{x} + b$, and picks the class from the sign. Each time it gets a point wrong, it nudges the line toward the right side of that point. The yellow arrow is $\\mathbf{w}$, which is always perpendicular to the line. If the data is linearly separable, the convergence theorem guarantees it finds a separating line. On XOR it never converges, and that limitation, pointed out by Minsky and Papert in 1969, is what led to networks with hidden layers.',
    steps: [
      'Start with small random weights $\\mathbf{w}$ and $b = 0$.',
      'For each point, compute $f = \\mathbf{w}\\cdot\\mathbf{x} + b$ and predict from its sign.',
      'If the prediction is wrong, update $\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta\\, y\\,\\mathbf{x}$ and $b \\leftarrow b + \\eta\\, y$.',
      'Repeat epochs until there are no mistakes or the limit is reached.',
    ],
    complexity: [['Training', 'O(n\\,d)\\ \\text{per epoch}'], ['Prediction', 'O(d)'], ['Memory', 'O(d)']],
    complexityNote: 'n is the number of examples and d the number of features (here d = 2).',
    pseudocode: [
      'w ← small values ; b ← 0',
      'for epoch = 1 to E:',
      '    mistakes ← 0',
      '    for each (x, y) in random order:',
      '        f ← w · x + b',
      '        if y · f ≤ 0:',
      '            w ← w + η · y · x',
      '            b ← b + η · y',
      '            mistakes ← mistakes + 1',
      '    if mistakes = 0: stop',
    ],
    exercises: [
      'Run it on XOR and watch the mistakes chart. Why does it never reach zero?',
      'Drag the line somewhere wrong before pressing play. Does the starting point change how many epochs it needs?',
      'Compare η = 0.01 and η = 1 on separable data. Does the rate matter? Hint: with b = 0 at the start, scaling w does not move the line.',
      'Implement the pocket perceptron, which keeps the best weights seen so far, and try it on the overlapping blobs.',
    ],
  },
};
