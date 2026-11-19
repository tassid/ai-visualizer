import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { pct, sigmoid, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, SMOOTH } from '../../components/plot/marks';
import { ESL, ISLR, PRML, paper } from '../books';
import { createAda, score, trainingError, type AdaParams, type AdaState, type Stump } from './model';

/** The newest stump: a single cut, sliding to its place. */
function StumpLine({ st }: { st: Stump }) {
  const { x, y } = usePlot();
  const pos = st.feature === 0 ? { x1: x(st.threshold), x2: x(st.threshold), y1: y(-1), y2: y(1) } : { x1: x(-1), x2: x(1), y1: y(st.threshold), y2: y(st.threshold) };
  return <motion.line initial={false} animate={pos} transition={SMOOTH} stroke={COLORS.yellow} strokeWidth={2.5} pointerEvents="none" />;
}

function AdaVisual({ state: s, data, version, compact }: VisualProps<AdaState>) {
  const n = data.length || 1;
  const last = s.stumps[s.stumps.length - 1];
  return (
    <Plot heatmap={s.stumps.length ? { version, resolution: 96, color: (x, y) => binaryColor(sigmoid(2 * score(s, x, y))) } : null}>
      {/* Point area is proportional to its weight: the points AdaBoost is struggling with grow. */}
      <Points data={data} color={(p) => classColor(p.c)} radius={(_, i) => (compact ? 3.5 : Math.max(2, Math.min(16, 4.4 * Math.sqrt(s.weights[i] * n))))} />
      {last && !compact && <StumpLine st={last} />}
    </Plot>
  );
}

export const adaboost: AlgorithmDef<AdaState> = {
  id: 'adaboost',
  name: 'AdaBoost',
  group: 'Neighbors, trees and ensembles',
  kind: 'classification',
  datasets: ['circles', 'moons', 'xor', 'blobs', 'spiral'],
  defaultDataset: 'circles',
  defaultSpeed: 3,
  thumbnailSteps: 60,
  hint: 'Watch the point sizes: each round, the points the last stump got wrong grow heavier.',
  params: [{ key: 'rounds', label: 'Rounds', type: 'range', min: 1, max: 200, step: 1, value: 80 }],
  create: (data, params) => createAda(data, params as unknown as AdaParams),
  Visual: AdaVisual,
  history: { label: 'Training error', get: (s) => s.history },
  equation: (s) => {
    const last = s.stumps[s.stumps.length - 1];
    const head = `F(\\mathbf{x}) = \\sum_{t=1}^{T}\\alpha_t\\,h_t(\\mathbf{x})`;
    if (!last) return head;
    return `${head} \\qquad {\\color{${COLORS.yellow}}\\alpha_{${s.stumps.length}}} = \\tfrac12\\ln\\frac{1-${tex(last.error, 3)}}{${tex(last.error, 3)}} = ${tex(last.alpha)}`;
  },
  metrics: (s, data) => [
    { label: 'Stumps', value: s.stumps.length },
    { label: 'Training error', value: pct(trainingError(s, data)) },
    { label: 'Largest weight', value: data.length ? `${(Math.max(...s.weights) * data.length).toFixed(1)}× average` : 'n/a' },
  ],
  docs: {
    description:
      'AdaBoost builds a strong classifier out of weak ones. Each weak learner here is a decision stump, a single vertical or horizontal cut, barely better than guessing on its own. After each round, the points the stump misclassified get heavier and the rest lighter, so the next stump is forced to deal with them. Point size shows the weight. The final prediction is a vote in which each stump counts with weight $\\alpha_t$, larger for more accurate stumps.',
    steps: [
      'Give every point the same weight $w_i = 1/n$.',
      'Fit the stump $h_t$ with the lowest weighted error $\\varepsilon_t$.',
      'Give it the vote $\\alpha_t = \\frac12\\ln\\frac{1-\\varepsilon_t}{\\varepsilon_t}$.',
      'Reweight: $w_i \\leftarrow w_i\\,e^{-\\alpha_t y_i h_t(\\mathbf{x}_i)}$, then normalize.',
      'Predict with the sign of $F(\\mathbf{x}) = \\sum_t\\alpha_t h_t(\\mathbf{x})$.',
    ],
    complexity: [['Training', 'O(T\\,d\\,n\\log n)'], ['Prediction', 'O(T)'], ['Memory', 'O(T + n)']],
    pseudocode: [
      'wᵢ ← 1/n for every point',
      'for t = 1 to T:',
      '    hₜ ← stump with the lowest weighted error εₜ',
      '    αₜ ← ½ ln((1 - εₜ) / εₜ)',
      '    wᵢ ← wᵢ · exp(-αₜ yᵢ hₜ(xᵢ))',
      '    normalize w to sum to 1',
      'predict(x) = sign( Σₜ αₜ hₜ(x) )',
    ],
    exercises: [
      'Add a few points of the wrong class inside a cluster. How heavy do they get? Why is AdaBoost sensitive to label noise?',
      'Watch the training error chart. Does it keep dropping after it reaches zero? Look up the margin theory of boosting.',
      'Replace stumps with trees of depth 2 (reuse shared/tree.ts with weights). How many rounds do you need now?',
      'Show that the reweighting step gives the last stump a weighted error of exactly 0.5 on the new weights.',
    ],
    theory: [
      'AdaBoost can be derived as greedy minimization of the exponential loss $L = \\sum_i e^{-y_i F(\\mathbf{x}_i)}$ over additive models $F = \\sum_t\\alpha_t h_t$ (Friedman, Hastie and Tibshirani, 2000). Suppose $F_{t-1}$ is fixed and we add $\\alpha h$. With $w_i \\propto e^{-y_i F_{t-1}(\\mathbf{x}_i)}$,',
      '$$L = \\sum_i w_i\\,e^{-\\alpha y_i h(\\mathbf{x}_i)} = e^{-\\alpha}\\!\\!\\sum_{y_i = h(\\mathbf{x}_i)}\\!\\! w_i + e^{\\alpha}\\!\\!\\sum_{y_i \\ne h(\\mathbf{x}_i)}\\!\\! w_i = e^{-\\alpha}(1-\\varepsilon) + e^{\\alpha}\\varepsilon.$$',
      'For any fixed $\\alpha > 0$, the best $h$ minimizes the weighted error $\\varepsilon$, which is the stump search. Setting the derivative in $\\alpha$ to zero gives $\\alpha = \\frac12\\ln\\frac{1-\\varepsilon}{\\varepsilon}$, and the new weights are the old ones times $e^{-\\alpha y_i h(\\mathbf{x}_i)}$. All three lines of the algorithm fall out of one loss.',
      'Freund and Schapire showed the training error is at most $\\prod_t 2\\sqrt{\\varepsilon_t(1-\\varepsilon_t)}$. If every weak learner is better than chance by some $\\gamma$, $\\varepsilon_t \\le \\frac12 - \\gamma$, the bound is at most $e^{-2T\\gamma^2}$: the training error falls exponentially with the number of rounds.',
      'AdaBoost often keeps improving on test data long after the training error hits zero, which puzzled people for years. The margin explanation of Schapire, Freund, Bartlett and Lee is that later rounds keep increasing $y_i F(\\mathbf{x}_i)/\\sum_t\\alpha_t$, the confidence of correct votes, and generalization bounds depend on that margin. The exponential loss also explains its weakness: a mislabeled point with a large negative margin gets an exponentially large weight.',
    ],
    references: [
      ESL('Chapter 10, Boosting and additive trees'),
      ISLR('Section 8.2.3, Boosting'),
      PRML('Section 14.3, Boosting'),
      paper('Freund and Schapire, A decision-theoretic generalization of on-line learning and an application to boosting, Journal of Computer and System Sciences 55 (1997)'),
      paper('Friedman, Hastie and Tibshirani, Additive logistic regression: a statistical view of boosting, Annals of Statistics 28 (2000)'),
      paper('Schapire and Freund, Boosting: Foundations and Algorithms (2012)'),
    ],
  },
};
