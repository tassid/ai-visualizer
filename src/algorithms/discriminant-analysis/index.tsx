import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, pct, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { DecisionLine, Points, Ring } from '../../components/plot/marks';
import { CovEllipse } from '../../components/plot/shapes';
import { ESL, ISLR, PRML, paper } from '../books';
import { createDa, discriminant, ldaLine, posterior, sigmaFor, type DaParams, type DaState } from './model';

function DaVisual({ state: s, data, version, hover, compact }: VisualProps<DaState>) {
  const cur = s.current >= 0 ? data[s.current] : null;
  const line = s.model === 'LDA' ? ldaLine(s) : null;
  return (
    <Plot heatmap={{ version, color: (x, y) => binaryColor(posterior(s, x, y)) }}>
      {line && <DecisionLine w1={line.w[0]} w2={line.w[1]} b={line.b} />}
      <Points data={data} color={(p) => classColor(p.c)} />
      {([0, 1] as const).map((k) => s.moments[k].n >= 2 && <CovEllipse key={k} mean={s.moments[k].mean} cov={sigmaFor(s, k)} color={classColor(k)} />)}
      {!compact && <Ring at={hover ?? cur} color={COLORS.yellow} width={2.5} r={hover ? 6 : 11} />}
    </Plot>
  );
}

export const discriminantAnalysis: AlgorithmDef<DaState> = {
  id: 'discriminant-analysis',
  name: 'Linear and quadratic discriminant analysis',
  group: 'Linear, probabilistic and kernel models',
  kind: 'classification',
  datasets: ['tilted', 'blobs', 'circles', 'moons'],
  defaultDataset: 'tilted',
  defaultSpeed: 5,
  thumbnailSteps: 1000,
  hint: 'Switch between LDA and QDA and watch the ellipses: LDA forces both classes to share one shape.',
  params: [{ key: 'model', label: 'Model', type: 'select', options: ['LDA', 'QDA'], value: 'LDA' }],
  create: (data, params) => createDa(data, params as unknown as DaParams),
  Visual: DaVisual,
  equation: (s, _d, _p, hover) => {
    const head = `\\delta_k(\\mathbf{x}) = \\log\\pi_k - \\tfrac12\\log|\\Sigma_k| - \\tfrac12(\\mathbf{x}-{\\color{${COLORS.yellow}}\\boldsymbol\\mu_k})^{\\!\\top}\\Sigma_k^{-1}(\\mathbf{x}-{\\color{${COLORS.yellow}}\\boldsymbol\\mu_k})`;
    if (!hover || s.moments[0].n < 2 || s.moments[1].n < 2) return head;
    return `${head} \\qquad {\\color{${COLORS.red}}\\delta_A = ${tex(discriminant(s, 0, hover.x, hover.y))}},\\ {\\color{${COLORS.blue}}\\delta_B = ${tex(discriminant(s, 1, hover.x, hover.y))}}`;
  },
  metrics: (s, data) => [
    { label: 'Points seen', value: `${s.seen} of ${data.length}` },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => posterior(s, x, y))) },
  ],
  docs: {
    description:
      'Discriminant analysis models each class as a Gaussian with its own mean and a full covariance matrix, then classifies with Bayes’ theorem. Quadratic discriminant analysis (QDA) gives each class its own covariance, so the boundary is curved. Linear discriminant analysis (LDA) makes both classes share one pooled covariance, and the boundary becomes a straight line. The ellipses show the estimated covariance at 1σ and 2σ. On the tilted classes, compare with naive Bayes, whose ellipses cannot tilt.',
    steps: [
      'Estimate each class’s share $\\pi_k$, mean $\\boldsymbol\\mu_k$ and covariance $\\Sigma_k$.',
      'For LDA, replace the covariances by their pooled average $\\Sigma$.',
      'Score a point with $\\delta_k(\\mathbf{x})$, the log of $\\pi_k\\,\\mathcal{N}(\\mathbf{x};\\boldsymbol\\mu_k,\\Sigma_k)$.',
      'Predict the class with the larger score.',
    ],
    complexity: [['Training', 'O(n\\,d^2 + k\\,d^3)'], ['Prediction', 'O(k\\,d^2)'], ['Memory', 'O(k\\,d^2)']],
    pseudocode: [
      'for each class k: nₖ ← 0 ; μₖ ← 0 ; Cₖ ← 0',
      'for each (x, y):',
      '    nᵧ ← nᵧ + 1 ;  δ ← x - μᵧ',
      '    μᵧ ← μᵧ + δ / nᵧ',
      '    Cᵧ ← Cᵧ + δ (x - μᵧ)ᵀ',
      'LDA: Σ ← (C₀ + C₁) / n            (shared)',
      'QDA: Σₖ ← Cₖ / nₖ                  (per class)',
      'predict(x) = argmaxₖ δₖ(x)',
    ],
    exercises: [
      'On the tilted classes, compare LDA, QDA and naive Bayes. Which ones get the tilt right, and why?',
      'On the circles, LDA fails and QDA works. Explain it with the formula for δₖ.',
      'Show that the quadratic terms cancel when both classes share Σ, leaving a linear boundary.',
      'LDA is also a dimensionality reduction: project onto w = Σ⁻¹(μ₁ - μ₀). Show that w maximizes Fisher’s ratio of between-class to within-class variance.',
    ],
    theory: [
      'Assume $p(\\mathbf{x}\\mid k) = \\mathcal{N}(\\mathbf{x};\\boldsymbol\\mu_k,\\Sigma_k)$. Taking the log of $\\pi_k\\,p(\\mathbf{x}\\mid k)$ and dropping terms common to all classes gives the quadratic discriminant function',
      '$$\\delta_k(\\mathbf{x}) = -\\tfrac12\\log|\\Sigma_k| - \\tfrac12(\\mathbf{x}-\\boldsymbol\\mu_k)^\\top\\Sigma_k^{-1}(\\mathbf{x}-\\boldsymbol\\mu_k) + \\log\\pi_k.$$',
      'The boundary between two classes is $\\delta_0(\\mathbf{x}) = \\delta_1(\\mathbf{x})$, a quadratic equation in $\\mathbf{x}$. If all classes share one covariance $\\Sigma$, the term $\\mathbf{x}^\\top\\Sigma^{-1}\\mathbf{x}$ appears in every $\\delta_k$ and cancels, leaving a function linear in $\\mathbf{x}$:',
      '$$\\delta_k(\\mathbf{x}) = \\mathbf{x}^\\top\\Sigma^{-1}\\boldsymbol\\mu_k - \\tfrac12\\boldsymbol\\mu_k^\\top\\Sigma^{-1}\\boldsymbol\\mu_k + \\log\\pi_k.$$',
      'The pooled estimate is $\\hat\\Sigma = \\frac{1}{n}\\sum_k\\sum_{i: y_i = k}(\\mathbf{x}_i-\\hat{\\boldsymbol\\mu}_k)(\\mathbf{x}_i-\\hat{\\boldsymbol\\mu}_k)^\\top$ (often divided by $n - K$ instead of $n$ to remove bias).',
      'Fisher reached the same line in 1936 without any Gaussian assumption. He asked for the projection $\\mathbf{w}$ that maximizes the separation of the projected class means relative to the spread within each class, $J(\\mathbf{w}) = (\\mathbf{w}^\\top(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0))^2 / \\mathbf{w}^\\top S_W\\mathbf{w}$, and the answer is $\\mathbf{w} \\propto S_W^{-1}(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)$, the direction of the LDA normal vector.',
      'The trade-off between the two is bias against variance. QDA estimates $K\\,d(d+1)/2$ covariance parameters and LDA only $d(d+1)/2$. With little data or many features, LDA’s stronger assumption often wins even when it is wrong; with plenty of data and truly different covariances, QDA does better. Logistic regression gives a linear boundary of the same form as LDA, but fits it directly, without assuming Gaussian classes.',
    ],
    references: [
      ISLR('Section 4.4, Generative models for classification (LDA and QDA)'),
      ESL('Section 4.3, Linear discriminant analysis'),
      PRML('Section 4.1.4, Fisher’s linear discriminant, and Section 4.2, Probabilistic generative models'),
      paper('Fisher, The use of multiple measurements in taxonomic problems, Annals of Eugenics 7 (1936)'),
    ],
  },
};
