import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, pct, sigmoid, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { ContourLines } from '../../components/plot/shapes';
import { ISLR, ESL, PRML, paper } from '../books';
import { createKsvm, decision, supportCount, type KsvmParams, type KsvmState } from './model';

/** Support vectors: a ring whose size grows with α. */
function SupportRings({ s, data }: { s: KsvmState; data: VisualProps<KsvmState>['data'] }) {
  const { x, y } = usePlot();
  let max = 1;
  for (const a of s.alpha) max = Math.max(max, a);
  return (
    <g pointerEvents="none">
      {data.map((p, i) => s.alpha[i] > 0 && <circle key={i} cx={x(p.x)} cy={y(p.y)} r={6 + 6 * Math.sqrt(s.alpha[i] / max)} fill="none" stroke="#fff" strokeOpacity={0.75} strokeWidth={1} />)}
    </g>
  );
}

function KsvmVisual({ state: s, data, version, compact }: VisualProps<KsvmState>) {
  const f = (x: number, y: number) => decision(s, data, x, y);
  const cur = s.current >= 0 ? data[s.current] : null;
  return (
    <Plot heatmap={{ version, resolution: 48, color: (x, y) => binaryColor(sigmoid(2 * f(x, y))) }}>
      {s.t > 0 && (
        <ContourLines
          f={f}
          resolution={compact ? 40 : 56}
          levels={[
            { value: -1, color: 'rgba(255,255,255,.55)', width: 1.2, dash: '6 6' },
            { value: 1, color: 'rgba(255,255,255,.55)', width: 1.2, dash: '6 6' },
            { value: 0, color: COLORS.yellow, width: 2.5 },
          ]}
        />
      )}
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && <SupportRings s={s} data={data} />}
      {!compact && cur && <Ring at={cur} color={s.violated ? COLORS.yellow : '#fff'} width={s.violated ? 3 : 1.5} />}
    </Plot>
  );
}

export const kernelSvm: AlgorithmDef<KsvmState> = {
  id: 'kernel-svm',
  name: 'Kernel SVM',
  group: 'Linear, probabilistic and kernel models',
  kind: 'classification',
  datasets: ['circles', 'moons', 'xor', 'spiral', 'blobs'],
  defaultDataset: 'circles',
  defaultSpeed: 8,
  thumbnailSteps: 1500,
  hint: 'Raise γ to make each support vector’s influence more local, and watch the boundary wrap tighter.',
  params: [
    { key: 'gamma', label: 'Kernel width γ', type: 'select', options: [1, 3, 10, 30, 100], value: 10, help: 'K(a, b) = exp(-γ |a - b|²). Larger γ, more local influence.' },
    { key: 'lambda', label: 'Regularization λ', type: 'select', options: [0.001, 0.01, 0.1], value: 0.01 },
    { key: 'iters', label: 'Steps', type: 'range', min: 200, max: 6000, step: 200, value: 2000 },
  ],
  create: (data, params) => createKsvm(data, params as unknown as KsvmParams),
  Visual: KsvmVisual,
  equation: (s, data, _p, hover) => {
    const head = `f(\\mathbf{x}) = \\frac{1}{\\lambda t}\\sum_j {\\color{${COLORS.yellow}}\\alpha_j}\\,y_j\\,e^{-\\gamma\\lVert\\mathbf{x}-\\mathbf{x}_j\\rVert^2}`;
    return hover ? `${head} = ${tex(decision(s, data, hover.x, hover.y))}` : `${head} \\qquad t = ${s.t}`;
  },
  metrics: (s, data) => [
    { label: 'Step', value: s.t },
    { label: 'Support vectors', value: supportCount(s) },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => (decision(s, data, x, y) >= 0 ? 1 : 0))) },
  ],
  docs: {
    description:
      'A kernel SVM draws curved boundaries by doing linear separation in a feature space it never builds. Each support vector contributes a Gaussian bump $e^{-\\gamma\\lVert\\mathbf{x}-\\mathbf{x}_j\\rVert^2}$, positive or negative depending on its class, and the boundary (yellow) is where the bumps balance. The dashed lines are the margins $f = \\pm 1$. Training here uses kernelized Pegasos: each step draws one point, and if it falls inside the margin its weight $\\alpha$ grows by one. The white rings mark support vectors, larger for larger $\\alpha$.',
    steps: [
      'Start with all $\\alpha_j = 0$.',
      'Draw a random point $i$ and evaluate $y_i f(\\mathbf{x}_i)$ using the current $\\alpha$.',
      'If it is below 1 (inside the margin or misclassified), increment $\\alpha_i$.',
      'Repeat. The decision function is the weighted sum of kernels centered on the support vectors.',
    ],
    complexity: [['Training (Pegasos)', 'O(n\\,s)\\ \\text{per step}'], ['Training (SMO)', 'O(n^2)\\ \\text{to}\\ O(n^3)'], ['Prediction', 'O(s\\,d)'], ['Memory', 'O(n)']],
    complexityNote: 's is the number of support vectors. Prediction cost grows with them, which is why sparse solutions matter.',
    pseudocode: [
      'for t = 1 to T:',
      '    pick i at random',
      '    f ← (1 / (λ t)) Σⱼ αⱼ yⱼ K(xⱼ, xᵢ)',
      '    if yᵢ · f < 1:',
      '        αᵢ ← αᵢ + 1',
      'predict(x) = sign( Σⱼ αⱼ yⱼ K(xⱼ, x) )',
    ],
    exercises: [
      'On the circles, compare γ = 1 and γ = 100. Which one overfits? How many support vectors does each need?',
      'Replace the RBF kernel with a polynomial kernel (1 + a·b)² in model.ts. Which datasets can it solve?',
      'Show that the RBF kernel is an inner product: expand e^(-γ|a-b|²) and find the features.',
      'Implement the SMO algorithm, which solves the dual exactly, and compare the support vectors with Pegasos.',
    ],
    theory: [
      'The dual of the soft margin SVM depends on the data only through inner products:',
      '$$\\max_{\\boldsymbol\\alpha}\\ \\sum_i\\alpha_i - \\tfrac12\\sum_{i,j}\\alpha_i\\alpha_j y_i y_j\\,\\mathbf{x}_i\\cdot\\mathbf{x}_j \\quad\\text{s.t.}\\quad 0 \\le \\alpha_i \\le C,\\ \\ \\sum_i\\alpha_i y_i = 0.$$',
      'Replacing $\\mathbf{x}_i\\cdot\\mathbf{x}_j$ with $\\boldsymbol\\phi(\\mathbf{x}_i)\\cdot\\boldsymbol\\phi(\\mathbf{x}_j)$ for some feature map $\\boldsymbol\\phi$ trains a linear SVM on the features. The kernel trick is that we never need $\\boldsymbol\\phi$, only $K(\\mathbf{a},\\mathbf{b}) = \\boldsymbol\\phi(\\mathbf{a})\\cdot\\boldsymbol\\phi(\\mathbf{b})$. Mercer’s theorem says any symmetric $K$ whose Gram matrices are all positive semidefinite is such an inner product. For the RBF kernel the feature space is infinite dimensional, yet each kernel evaluation costs $O(d)$.',
      'The representer theorem guarantees that the minimizer of a regularized loss in the kernel’s function space has the form $f(\\mathbf{x}) = \\sum_i\\beta_i K(\\mathbf{x}_i,\\mathbf{x})$, a combination of kernels centered at the training points. For the hinge loss most $\\beta_i$ are zero, which is the sparsity of support vectors.',
      'Pegasos applies stochastic subgradient descent to the primal objective with step $\\eta_t = 1/(\\lambda t)$. Unrolling the updates, the weight vector after $t$ steps is $\\mathbf{w}_t = \\frac{1}{\\lambda t}\\sum_j\\alpha_j y_j\\boldsymbol\\phi(\\mathbf{x}_j)$, where $\\alpha_j$ counts the steps in which point $j$ was drawn and violated the margin. Everything can then be written with kernels, which is the version animated here.',
      'The width $\\gamma$ sets the scale of similarity. Small $\\gamma$ makes every point similar to every other and the boundary nearly linear; large $\\gamma$ makes each support vector influence only its immediate neighborhood, and in the limit the SVM memorizes the training set like 1-nearest-neighbor.',
    ],
    references: [
      ISLR('Section 9.3, Support vector machines (kernels)'),
      ESL('Section 12.3, Support vector machines and kernels'),
      PRML('Chapter 6, Kernel methods, and Section 7.1, Maximum margin classifiers'),
      paper('Schölkopf and Smola, Learning with Kernels (2002)'),
      paper('Shalev-Shwartz, Singer, Srebro and Cotter, Pegasos: primal estimated sub-gradient solver for SVM, Mathematical Programming 127 (2011)'),
      paper('Platt, Sequential minimal optimization: a fast algorithm for training support vector machines, Microsoft Research technical report (1998)'),
    ],
  },
};
