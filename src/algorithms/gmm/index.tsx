import { CLUSTER_PALETTE, hexToRgb, rgbCss, COLORS } from '../../core/colors';
import { gauss2 } from '../../core/linalg';
import { fmt, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { Handle, Points } from '../../components/plot/marks';
import { CovEllipse } from '../../components/plot/shapes';
import { ESL, PRML, paper } from '../books';
import { createGmm, type GmmParams, type GmmState } from './model';

const PAL = CLUSTER_PALETTE.map(hexToRgb);

/** A point's color is the responsibility-weighted blend of the component colors. */
function blend(g: Float64Array): string {
  let r = 0, gg = 0, b = 0;
  for (let k = 0; k < g.length; k++) { const c = PAL[k % PAL.length]; r += g[k] * c[0]; gg += g[k] * c[1]; b += g[k] * c[2]; }
  return rgbCss([r, gg, b]);
}

function GmmVisual({ state: s, data, version, compact }: VisualProps<GmmState>) {
  return (
    <Plot
      heatmap={{
        version,
        resolution: 64,
        color: (x, y) => {
          let d = 0;
          for (const c of s.comps) d += c.pi * gauss2(x, y, c.mu, c.cov);
          const a = Math.min(0.35, d / 12);
          return [88, 196, 221, a]; // the mixture density, as a soft blue glow of the plane
        },
      }}
    >
      <Points data={data} color={(_, i) => (s.iter === 0 && s.phase === 'E' ? COLORS.grey : blend(s.gamma[i]))} />
      {s.comps.map((c, k) => <CovEllipse key={k} mean={c.mu} cov={c.cov} color={CLUSTER_PALETTE[k % CLUSTER_PALETTE.length]} />)}
      {!compact &&
        s.comps.map((c, k) => (
          <Handle key={`h${k}`} title={`Drag component ${k + 1}`} at={{ x: c.mu[0], y: c.mu[1] }} visible={false} onDrag={(px, py) => { c.mu = [px, py]; }} />
        ))}
    </Plot>
  );
}

export const gmm: AlgorithmDef<GmmState> = {
  id: 'gmm',
  name: 'Gaussian mixture (EM)',
  group: 'Clustering',
  kind: 'clustering',
  datasets: ['clusters', 'tilted', 'blobs', 'moons'],
  defaultDataset: 'clusters',
  defaultSpeed: 2,
  thumbnailSteps: 120,
  hint: 'Drag a component’s center somewhere wrong and press play: EM pulls it back. Point colors blend by responsibility.',
  params: [
    { key: 'k', label: 'Components', type: 'range', min: 1, max: 8, step: 1, value: 4 },
    { key: 'iters', label: 'Max iterations', type: 'range', min: 10, max: 300, step: 10, value: 150 },
  ],
  create: (data, params) => createGmm(data, params as unknown as GmmParams),
  Visual: GmmVisual,
  history: { label: 'Log-likelihood', get: (s) => s.history },
  equation: (s) =>
    s.phase === 'E'
      ? `\\gamma_{ik} = \\frac{\\pi_k\\,\\mathcal{N}(\\mathbf{x}_i;\\boldsymbol\\mu_k,\\Sigma_k)}{\\sum_j \\pi_j\\,\\mathcal{N}(\\mathbf{x}_i;\\boldsymbol\\mu_j,\\Sigma_j)} \\qquad \\log p(X) = ${tex(s.logLik)}`
      : `{\\color{${COLORS.yellow}}\\boldsymbol\\mu_k} = \\frac{\\sum_i\\gamma_{ik}\\mathbf{x}_i}{\\sum_i\\gamma_{ik}}, \\quad \\Sigma_k = \\frac{\\sum_i\\gamma_{ik}(\\mathbf{x}_i-\\boldsymbol\\mu_k)(\\mathbf{x}_i-\\boldsymbol\\mu_k)^{\\top}}{\\sum_i\\gamma_{ik}} \\qquad \\log p(X) = ${tex(s.logLik)}`,
  metrics: (s) => [
    { label: 'Iteration', value: s.iter },
    { label: 'Step', value: s.phase === 'E' ? 'expectation' : 'maximization' },
    { label: 'Log-likelihood', value: fmt(s.logLik, 2) },
    { label: 'Weights π', value: s.comps.map((c) => c.pi.toFixed(2)).join(', ') },
  ],
  docs: {
    description:
      'A Gaussian mixture says the data came from $k$ Gaussian blobs, each with its own center, shape and share. EM fits it by alternating two steps. The E step asks, for every point, how likely each component is to have produced it; those probabilities, the responsibilities, set each point’s color as a blend of the component colors. The M step moves each component to the weighted mean and covariance of the points it is responsible for. It is a soft version of k-means that also learns the shape of each cluster.',
    steps: [
      'Start with $k$ components at random points, with small round covariances.',
      'E step: compute the responsibilities $\\gamma_{ik}$ with Bayes’ theorem.',
      'M step: update each $\\pi_k$, $\\boldsymbol\\mu_k$ and $\\Sigma_k$ using the points weighted by $\\gamma_{ik}$.',
      'Repeat until the log-likelihood stops improving.',
    ],
    complexity: [['Training', 'O(n\\,k\\,d^2)\\ \\text{per iteration}'], ['Prediction', 'O(k\\,d^2)'], ['Memory', 'O(n\\,k + k\\,d^2)']],
    pseudocode: [
      'initialize πₖ, μₖ, Σₖ',
      'repeat:',
      '    E: γᵢₖ ← πₖ N(xᵢ; μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ; μⱼ, Σⱼ)',
      '    M: Nₖ ← Σᵢ γᵢₖ ;  πₖ ← Nₖ / n',
      '       μₖ ← Σᵢ γᵢₖ xᵢ / Nₖ',
      '       Σₖ ← Σᵢ γᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ / Nₖ',
      'until log-likelihood converges',
    ],
    exercises: [
      'On the tilted classes, compare the result with k-means at k = 2. Why does the mixture fit better?',
      'Set k = 8 on four clusters. What happens to the extra components and their weights π?',
      'Drag two components onto the same cluster. Does EM separate them? Relate this to local maxima.',
      'Prove that k-means is the limit of EM when all covariances are σ²I and σ → 0.',
    ],
    theory: [
      'Introduce a hidden label $z_i \\in \\{1, \\ldots, k\\}$ for the component that generated each point, with $P(z_i = k) = \\pi_k$. The likelihood of the observed data sums over the unknown labels,',
      '$$\\log p(X\\mid\\theta) = \\sum_i \\log\\sum_k \\pi_k\\,\\mathcal{N}(\\mathbf{x}_i;\\boldsymbol\\mu_k,\\Sigma_k),$$',
      'and the log of a sum has no closed form maximizer. If we knew the labels, we would just fit one Gaussian per group. EM exploits that: the E step replaces the unknown labels by their posterior probabilities $\\gamma_{ik} = P(z_i = k\\mid\\mathbf{x}_i,\\theta)$ under the current parameters, and the M step maximizes the expected complete-data log-likelihood $\\sum_i\\sum_k\\gamma_{ik}\\log[\\pi_k\\mathcal{N}(\\mathbf{x}_i;\\boldsymbol\\mu_k,\\Sigma_k)]$, which gives the weighted averages in closed form.',
      'Why it works: for any distribution $q$ over the labels, Jensen’s inequality gives the lower bound $\\log p(X\\mid\\theta) \\ge \\mathcal{L}(q,\\theta) = \\mathbb{E}_q[\\log p(X, Z\\mid\\theta)] + H(q)$, with equality when $q$ is the true posterior. The E step makes the bound tight; the M step raises the bound. So the log-likelihood never goes down, which the chart shows.',
      'Two practical issues. EM finds a local maximum, so it depends on initialization (k-means is a common starting point). And the likelihood is unbounded: a component can shrink onto a single point and its density goes to infinity. The small value added to each covariance here guards against that, and a Bayesian treatment with priors on the parameters removes it properly.',
    ],
    references: [
      PRML('Chapter 9, Mixture models and EM'),
      ESL('Section 8.5, The EM algorithm, and Section 6.8, Mixture models for density estimation and classification'),
      paper('Dempster, Laird and Rubin, Maximum likelihood from incomplete data via the EM algorithm, Journal of the Royal Statistical Society B 39 (1977)'),
    ],
  },
};
