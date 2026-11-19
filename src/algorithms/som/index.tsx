import { COLORS } from '../../core/colors';
import { fmt } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { ESL, paper } from '../books';
import { createSom, quantizationError, type SomParams, type SomState } from './model';

/** The neuron grid as a wire mesh: lines join grid neighbors, wherever they sit in the plane. */
function Mesh({ s }: { s: SomState }) {
  const { x, y } = usePlot();
  const N = s.size;
  const segs: string[] = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const a = s.w[r * N + c];
    if (c + 1 < N) { const b = s.w[r * N + c + 1]; segs.push(`M${x(a.x)},${y(a.y)}L${x(b.x)},${y(b.y)}`); }
    if (r + 1 < N) { const b = s.w[(r + 1) * N + c]; segs.push(`M${x(a.x)},${y(a.y)}L${x(b.x)},${y(b.y)}`); }
  }
  return (
    <g pointerEvents="none">
      <path d={segs.join('')} fill="none" stroke="#ececec" strokeOpacity={0.7} strokeWidth={1.1} />
      {s.w.map((n, j) => <circle key={j} cx={x(n.x)} cy={y(n.y)} r={j === s.bmu ? 5 : 2.6} fill={j === s.bmu ? COLORS.yellow : '#ececec'} />)}
    </g>
  );
}

function SomVisual({ state: s, data, compact }: VisualProps<SomState>) {
  const sample = s.sample >= 0 ? data[s.sample] : null;
  return (
    <Plot>
      <Points data={data} color={() => COLORS.blue} radius={3.6} opacity={() => 0.6} />
      <Mesh s={s} />
      {!compact && sample && <Ring at={sample} r={8} color={COLORS.yellow} />}
    </Plot>
  );
}

export const som: AlgorithmDef<SomState> = {
  id: 'som',
  name: 'Self-organizing map',
  group: 'Dimensionality reduction and maps',
  kind: 'reduction',
  datasets: ['moons', 'circles', 'clusters', 'spiral', 'elongated'],
  defaultDataset: 'moons',
  defaultSpeed: 9,
  thumbnailSteps: 3000,
  hint: 'Watch the yellow neuron: it and its grid neighbors are pulled toward each sample. Early on the whole map moves; later only small patches do.',
  params: [
    { key: 'size', label: 'Grid', type: 'select', options: [4, 6, 8, 10, 14], value: 8 },
    { key: 'steps', label: 'Samples', type: 'range', min: 500, max: 10000, step: 500, value: 4000 },
    { key: 'lr', label: 'Starting learning rate η₀', type: 'select', options: [0.1, 0.3, 0.5, 0.9], value: 0.5 },
  ],
  create: (data, params) => createSom(data, params as unknown as SomParams),
  Visual: SomVisual,
  equation: (s) =>
    `\\mathbf{w}_j \\leftarrow \\mathbf{w}_j + \\eta(t)\\,e^{-d(j,\\,{\\color{${COLORS.yellow}}c})^2/2\\sigma(t)^2}\\,(\\mathbf{x} - \\mathbf{w}_j) \\qquad \\eta = ${s.eta.toFixed(3)},\\ \\sigma = ${s.sigma.toFixed(2)}`,
  metrics: (s, data) => [
    { label: 'Samples seen', value: s.t },
    { label: 'Neurons', value: s.size * s.size },
    { label: 'Quantization error', value: fmt(quantizationError(s, data), 4) },
  ],
  docs: {
    description:
      'A self-organizing map is a grid of neurons whose weights live in the data space. Each sample pulls its best matching neuron toward it, and the neighbors of that neuron on the grid come along, less and less the farther away they are. Over time the grid unfolds across the data like a fishing net thrown over it, while neurons that are neighbors on the grid stay neighbors in the data. That preserved topology is what makes SOMs useful for mapping high-dimensional data onto a 2D grid.',
    steps: [
      'Start with the neurons bunched near the center.',
      'Draw a sample and find its best matching unit, the closest neuron.',
      'Move every neuron toward the sample by $\\eta\\,h\\,(\\mathbf{x}-\\mathbf{w}_j)$, where $h$ falls off with grid distance to the BMU.',
      'Shrink $\\eta$ and the neighborhood width $\\sigma$ over time.',
    ],
    complexity: [['Training', 'O(T\\,m\\,d)'], ['Mapping a point', 'O(m\\,d)'], ['Memory', 'O(m\\,d)']],
    complexityNote: 'm is the number of neurons and T the number of samples drawn.',
    pseudocode: [
      'for t = 1 to T:',
      '    x ← a random sample',
      '    c ← argminⱼ |x - wⱼ|                (best matching unit)',
      '    η ← η(t) ;  σ ← σ(t)               (both decreasing)',
      '    for every neuron j:',
      '        wⱼ ← wⱼ + η · exp(-d(j, c)² / 2σ²) · (x - wⱼ)',
    ],
    exercises: [
      'Use a 4 x 4 grid and a 14 x 14 grid on the spiral. Which one follows the shape, and which one covers it?',
      'Set the learning rate to 0.9 and watch the early steps. Why does the map twist, and does it untwist later?',
      'Make σ shrink to zero. What does the SOM become? (Hint: compare with online k-means.)',
      'Train a SOM on 3D colors (r, g, b) and draw each neuron in its color: a classic demonstration of topology preservation.',
    ],
    theory: [
      'A SOM is a set of prototypes $\\mathbf{w}_j$ arranged on a fixed low-dimensional lattice. Training is competitive: for a sample $\\mathbf{x}$ the winner is $c = \\arg\\min_j\\lVert\\mathbf{x}-\\mathbf{w}_j\\rVert$. Unlike k-means, the update is cooperative: the winner’s lattice neighbors also move,',
      '$$\\mathbf{w}_j \\leftarrow \\mathbf{w}_j + \\eta(t)\\,h_{jc}(t)\\,(\\mathbf{x}-\\mathbf{w}_j), \\qquad h_{jc}(t) = \\exp\\!\\Big(-\\frac{\\lVert\\mathbf{r}_j-\\mathbf{r}_c\\rVert^2}{2\\sigma(t)^2}\\Big),$$',
      'where $\\mathbf{r}_j$ is neuron $j$’s position on the lattice. Training has two phases: an ordering phase with a wide neighborhood, in which the map untangles and takes on the global shape of the data, and a convergence phase with a narrow one, in which prototypes settle into local detail. If the neighborhood shrinks to the winner alone, the update is online k-means.',
      'The map approximately preserves topology: nearby lattice nodes have nearby prototypes. The converse need not hold, and the map can fold. Two numbers are used to judge it: the quantization error, the mean distance from each sample to its winner, and the topographic error, the share of samples whose two closest prototypes are not lattice neighbors.',
      'SOMs have no explicit objective function in their original form, which makes their theory harder than that of PCA or k-means. Hastie, Tibshirani and Friedman describe them as a constrained version of k-means, and related methods with an explicit generative model, such as the generative topographic mapping, were developed to put them on firmer ground.',
    ],
    references: [
      ESL('Section 14.4, Self-organizing maps'),
      paper('Kohonen, Self-organized formation of topologically correct feature maps, Biological Cybernetics 43 (1982)'),
      paper('Kohonen, Self-Organizing Maps, 3rd ed. (2001)'),
      paper('Bishop, Svensén and Williams, GTM: the generative topographic mapping, Neural Computation 10 (1998)'),
    ],
  },
};
