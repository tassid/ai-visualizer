import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct, sigmoid, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points } from '../../components/plot/marks';
import { ESL, ISLR, paper } from '../books';
import { createGb, regSegments, scoreAt, type GbParams, type GbState, type RegNode } from './model';

function TreeCuts({ tree }: { tree: RegNode }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {regSegments(tree).map(([x0, y0, x1, y1], k) => (
        <motion.line key={k} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }} x1={x(x0)} y1={y(y0)} x2={x(x1)} y2={y(y1)} stroke={COLORS.yellow} strokeWidth={1.5} />
      ))}
    </g>
  );
}

function GbVisual({ state: s, data, params, version, compact }: VisualProps<GbState>) {
  const lr = Number(params.lr);
  const last = s.trees[s.trees.length - 1];
  return (
    <Plot heatmap={{ version, resolution: 96, color: (x, y) => binaryColor(sigmoid(scoreAt(s, lr, x, y))) }}>
      {/* Point size follows |residual|: the next tree concentrates where these are large. */}
      <Points data={data} color={(p) => classColor(p.c)} radius={(_, i) => (compact || !s.trees.length ? 4 : 2.5 + 9 * Math.abs(s.residuals[i]))} />
      {last && !compact && <TreeCuts key={s.trees.length} tree={last} />}
    </Plot>
  );
}

export const gradientBoosting: AlgorithmDef<GbState> = {
  id: 'gradient-boosting',
  name: 'Gradient boosting',
  group: 'Neighbors, trees and ensembles',
  kind: 'classification',
  datasets: ['moons', 'circles', 'xor', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 4,
  thumbnailSteps: 80,
  hint: 'Point size is the residual each new tree is fit to. Watch it shrink as the rounds add up.',
  params: [
    { key: 'rounds', label: 'Rounds', type: 'range', min: 10, max: 300, step: 10, value: 120 },
    { key: 'depth', label: 'Tree depth', type: 'range', min: 1, max: 4, step: 1, value: 2 },
    { key: 'lr', label: 'Learning rate ν', type: 'select', options: [0.05, 0.1, 0.3, 1], value: 0.3, help: 'Smaller steps need more rounds but usually generalize better.' },
  ],
  create: (data, params) => createGb(data, params as unknown as GbParams),
  Visual: GbVisual,
  history: { label: 'Log loss', get: (s) => s.history },
  equation: (s, _d, params, hover) => {
    const head = `F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\nu\\,{\\color{${COLORS.yellow}}h_m(\\mathbf{x})}, \\qquad r_i = y_i - \\sigma\\big(F_{m-1}(\\mathbf{x}_i)\\big)`;
    return hover ? `${head} \\qquad p = ${tex(sigmoid(scoreAt(s, Number(params.lr), hover.x, hover.y)))}` : head;
  },
  metrics: (s, data, params) => [
    { label: 'Trees', value: s.trees.length },
    { label: 'Log loss', value: fmt(s.loss, 4) },
    { label: 'Training accuracy', value: pct(accuracy(data, (x, y) => sigmoid(scoreAt(s, Number(params.lr), x, y)))) },
  ],
  docs: {
    description:
      'Gradient boosting adds small regression trees one at a time, each fit to what the current model still gets wrong. For classification the model is on the log-odds scale, $p = \\sigma(F(\\mathbf{x}))$, and what each tree learns is the residual $y_i - p_i$: the negative gradient of the log loss. Point size shows that residual. The yellow cuts are the newest tree. Libraries such as XGBoost, LightGBM and CatBoost are refined versions of exactly this loop, and they win many tabular data competitions.',
    steps: [
      'Start with a constant, the log-odds of the base rate.',
      'Compute the residuals $r_i = y_i - \\sigma(F(\\mathbf{x}_i))$.',
      'Fit a small regression tree to the residuals.',
      'Set each leaf to a Newton step $\\sum r_i / \\sum p_i(1-p_i)$.',
      'Add the tree, scaled by the learning rate $\\nu$, and repeat.',
    ],
    complexity: [['Training', 'O(M\\,d\\,n\\log n)'], ['Prediction', 'O(M\\cdot\\text{depth})'], ['Memory', 'O(M\\cdot 2^{\\text{depth}})']],
    complexityNote: 'M is the number of rounds. Unlike a random forest, the trees depend on each other, so they cannot be trained in parallel.',
    pseudocode: [
      'F₀ ← log(p̄ / (1 - p̄))',
      'for m = 1 to M:',
      '    rᵢ ← yᵢ - σ(F(xᵢ))                  (negative gradient)',
      '    h ← regression tree fit to (xᵢ, rᵢ)',
      '    leaf values ← Σ r / Σ p(1 - p)      (Newton step)',
      '    F ← F + ν · h',
      'predict(x) = σ(F(x))',
    ],
    exercises: [
      'Compare ν = 1 with ν = 0.05 at the same number of rounds. Which boundary looks smoother? Which loss is lower?',
      'With depth 1 (stumps), can boosting solve XOR? With depth 2? Why does depth control the interactions the model can capture?',
      'Hold out a third of the points as a test set and plot test loss by round. Where does it start to rise?',
      'Replace the log loss with squared error. Which lines of model.ts change?',
    ],
    theory: [
      'Gradient boosting is gradient descent in function space. We want $F$ minimizing $\\sum_i L(y_i, F(\\mathbf{x}_i))$. At step $m$, the negative gradient of the loss with respect to the model’s outputs is',
      '$$r_{im} = -\\left[\\frac{\\partial L(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)}\\right]_{F = F_{m-1}}.$$',
      'Taking a step $F \\leftarrow F + \\nu\\,\\mathbf{r}$ would only change $F$ at the training points. Instead we fit a regression tree $h_m$ to the pairs $(\\mathbf{x}_i, r_{im})$, which generalizes the step to the whole space, and add $\\nu\\,h_m$. For squared error the negative gradient is the ordinary residual $y_i - F(\\mathbf{x}_i)$, so boosting literally fits trees to residuals. For log loss with $p = \\sigma(F)$ it is $y_i - p_i$.',
      'Each leaf $R_j$ of the tree then gets its own optimal constant, found by a line search: $\\gamma_j = \\arg\\min_\\gamma\\sum_{\\mathbf{x}_i\\in R_j} L(y_i, F_{m-1}(\\mathbf{x}_i) + \\gamma)$. For log loss there is no closed form, and one Newton step gives $\\gamma_j = \\sum r_i/\\sum p_i(1-p_i)$, which is what this implementation uses. XGBoost makes the second order view systematic, choosing splits by the gain $\\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{G^2}{H+\\lambda}$ in sums of gradients $G$ and Hessians $H$.',
      'Shrinkage ($\\nu < 1$) and limited depth are the main regularizers. Friedman found that small $\\nu$ with more rounds almost always generalizes better. Depth limits the order of feature interactions: stumps give an additive model, depth $J$ allows interactions among up to $J$ features. Subsampling the rows each round (stochastic gradient boosting) adds a further variance reduction.',
    ],
    references: [
      ESL('Chapter 10, Boosting and additive trees (Sections 10.9 to 10.12)'),
      ISLR('Section 8.2.3, Boosting'),
      paper('Friedman, Greedy function approximation: a gradient boosting machine, Annals of Statistics 29 (2001)'),
      paper('Friedman, Stochastic gradient boosting, Computational Statistics and Data Analysis 38 (2002)'),
      paper('Chen and Guestrin, XGBoost: a scalable tree boosting system, KDD (2016)', 'https://arxiv.org/abs/1603.02754'),
    ],
  },
};
