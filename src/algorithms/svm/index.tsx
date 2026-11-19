import { AnimatePresence, motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct, sigmoid, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { DecisionLine, Points } from '../../components/plot/marks';
import { LinearHandles } from '../../components/plot/LinearHandles';
import { createSvm, decision, hingeObjective, isSupport, margin, type SvmParams, type SvmState } from './model';

/** Support vectors get a ring that grows in when a point enters the margin and fades when it leaves. */
function SupportRings({ s, data }: { s: SvmState; data: VisualProps<SvmState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <AnimatePresence>
      {data.map((p, i) =>
        isSupport(s, p) ? (
          <motion.circle key={i} cx={x(p.x)} cy={y(p.y)} initial={{ r: 0, opacity: 0 }} animate={{ r: 8.5, opacity: 0.9 }} exit={{ r: 14, opacity: 0 }} transition={{ duration: 0.3 }} fill="none" stroke="#fff" strokeWidth={1.2} pointerEvents="none" />
        ) : null,
      )}
    </AnimatePresence>
  );
}

function SvmVisual({ state: s, data, version, compact }: VisualProps<SvmState>) {
  const [w1, w2] = s.w;
  return (
    <Plot heatmap={{ version, color: (x, y) => binaryColor(sigmoid(3 * decision(s, x, y))) }}>
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={1} color="rgba(255,255,255,.55)" width={1.3} dash="7 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={-1} color="rgba(255,255,255,.55)" width={1.3} dash="7 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} />
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && <SupportRings s={s} data={data} />}
      {!compact && <LinearHandles s={s} mode="vector" scale={0.06} />}
    </Plot>
  );
}

export const svm: AlgorithmDef<SvmState> = {
  id: 'svm',
  name: 'Linear SVM',
  group: 'Linear and probabilistic models',
  kind: 'classification',
  datasets: ['blobs', 'linear', 'moons'],
  defaultDataset: 'blobs',
  defaultSpeed: 5,
  thumbnailSteps: 800,
  hint: 'Drag the arrow tip: a longer w means a narrower margin. Drag the dot to slide the line.',
  params: [
    { key: 'lambda', label: 'Regularization λ', type: 'select', options: [0.001, 0.01, 0.05, 0.1, 0.5], value: 0.01, help: 'Higher λ favors a wide margin over classifying every point.' },
    { key: 'lr', label: 'Learning rate η', type: 'select', options: [0.05, 0.1, 0.5, 1], value: 0.5 },
    { key: 'iters', label: 'Iterations', type: 'range', min: 100, max: 3000, step: 100, value: 800 },
  ],
  create: (data, params) => createSvm(data, params as unknown as SvmParams),
  Visual: SvmVisual,
  history: { label: 'Objective', get: (s) => s.history },
  equation: (s, data, params) =>
    `\\tfrac{\\lambda}{2}\\lVert{\\color{${COLORS.yellow}}\\mathbf{w}}\\rVert^2 + \\tfrac{1}{n}\\sum_i \\max\\!\\big(0,\\ 1 - y_i f(\\mathbf{x}_i)\\big) = ${tex(hingeObjective(s, data, Number(params.lambda)), 4)} \\qquad \\text{margin} = \\frac{2}{\\lVert\\mathbf{w}\\rVert} = ${tex(margin(s), 3)}`,
  metrics: (s, data) => [
    { label: 'Iteration', value: s.it },
    { label: 'Objective', value: fmt(s.loss, 4) },
    { label: 'Margin width', value: fmt(margin(s), 3) },
    { label: 'Support vectors', value: data.filter((p) => isSupport(s, p)).length },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => (decision(s, x, y) >= 0 ? 1 : 0))) },
  ],
  docs: {
    description:
      'A support vector machine looks for the line that separates the classes with the widest possible margin. Only the points on or inside the margin, the support vectors (ringed in white), shape the solution: delete every other point and the line stays put. This is a linear SVM trained by subgradient descent on the hinge loss with an L2 penalty. $\\lambda$ trades a wide margin against classifying every point. For curved boundaries, real SVMs use the kernel trick, which computes inner products in a higher dimensional space without building it.',
    steps: [
      'The boundary is $\\mathbf{w}\\cdot\\mathbf{x} + b = 0$ and the margins are $\\mathbf{w}\\cdot\\mathbf{x} + b = \\pm 1$.',
      'Points with $y\\,f(\\mathbf{x}) \\ge 1$ are outside the margin and cost nothing.',
      'Points inside the margin push the line (hinge loss).',
      'The $\\lambda\\lVert\\mathbf{w}\\rVert^2$ term shrinks $\\mathbf{w}$, which widens the margin $2/\\lVert\\mathbf{w}\\rVert$.',
    ],
    complexity: [['Training (subgradient)', 'O(n\\,d)\\ \\text{per iteration}'], ['Training (dual, kernel)', 'O(n^2)\\ \\text{to}\\ O(n^3)'], ['Prediction (linear)', 'O(d)'], ['Prediction (kernel)', 'O(s\\,d)'], ['Memory', 'O(d)']],
    complexityNote: 's is the number of support vectors.',
    pseudocode: [
      'for t = 1 to T:',
      '    gw ← λ · w ;  gb ← 0',
      '    for each (x, y):',
      '        if y · (w·x + b) < 1:',
      '            gw ← gw - y · x / n',
      '            gb ← gb - y / n',
      '    w ← w - ηₜ · gw ;  b ← b - ηₜ · gb',
    ],
    exercises: [
      'Compare λ = 0.001 and λ = 0.5 on the blobs. What happens to the margin and to the number of support vectors?',
      'Drag a point that is not a support vector far away (but on its own side). Does the line move?',
      'Change the step decay in model.ts to 1/t (the original Pegasos) and compare convergence.',
      'Implement a kernel SVM with the RBF kernel using simplified SMO and try it on the moons.',
    ],
  },
};
