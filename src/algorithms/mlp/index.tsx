import { useMemo } from 'react';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { NetworkDiagram, type NetworkSnapshot } from '../../components/charts/NetworkDiagram';
import { Plot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { createMlp, type MlpParams, type MlpState } from './model';

function MlpVisual({ state: s, data, version, hover, compact }: VisualProps<MlpState>) {
  return (
    <Plot heatmap={{ version, resolution: 64, color: (x, y) => binaryColor(s.net.predict(x, y)) }}>
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && hover && <Ring at={hover} r={6} color={COLORS.yellow} />}
    </Plot>
  );
}

const GRID = 16;

/** The network, with every neuron's activation map. Hover the plot to run a forward pass. */
function MlpNetwork({ state: s, version, hover }: VisualProps<MlpState>) {
  const base = useMemo(() => {
    const net = s.net;
    const maps = net.sizes.map((n) => Array.from({ length: n }, () => new Float64Array(GRID * GRID)));
    // One forward pass per grid point fills the maps of ALL neurons at once.
    for (let gj = 0; gj < GRID; gj++) {
      for (let gi = 0; gi < GRID; gi++) {
        const acts = net.forward(-1 + ((gi + 0.5) / GRID) * 2, 1 - ((gj + 0.5) / GRID) * 2);
        acts.forEach((layer, l) => layer.forEach((v, k) => { maps[l][k][gj * GRID + gi] = v; }));
      }
    }
    const L = net.sizes.length - 1;
    const ranges = net.sizes.map((_, l): [number, number] => (l === 0 ? [-1, 1] : l === L ? [0, 1] : net.act.range));
    return { sizes: net.sizes, weights: net.params.W, maps, ranges, grid: GRID };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, s]);
  const snapshot: NetworkSnapshot = { ...base, probe: hover ? s.net.forward(hover.x, hover.y) : null };
  return <NetworkDiagram net={snapshot} width={560} height={290} version={version} />;
}

export const mlp: AlgorithmDef<MlpState> = {
  id: 'mlp',
  name: 'Multilayer perceptron',
  group: 'Neural networks',
  kind: 'classification',
  datasets: ['circles', 'xor', 'moons', 'spiral', 'blobs', 'linear'],
  defaultDataset: 'circles',
  defaultSpeed: 6,
  thumbnailSteps: 250,
  hint: 'Hover the plane to send that point through the network below.',
  params: [
    { key: 'arch', label: 'Hidden layers', type: 'select', options: ['2', '4', '8', '4-4', '8-8', '16-16', '8-8-8'], value: '8-8', help: 'Neurons per hidden layer.' },
    { key: 'act', label: 'Activation', type: 'select', options: ['tanh', 'relu', 'sigmoid'], value: 'tanh' },
    { key: 'opt', label: 'Optimizer', type: 'select', options: ['Adam', 'Momentum', 'SGD'], value: 'Adam' },
    { key: 'lr', label: 'Learning rate η', type: 'select', options: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], value: 0.03 },
    { key: 'batch', label: 'Batch size', type: 'select', options: ['full', 64, 32, 8], value: 'full' },
    { key: 'epochs', label: 'Epochs', type: 'range', min: 100, max: 4000, step: 100, value: 1500 },
  ],
  create: (data, params) => createMlp(data, params as unknown as MlpParams),
  Visual: MlpVisual,
  aux: { title: 'Network', Component: MlpNetwork },
  history: { label: 'Loss', get: (s) => s.history },
  equation: (s, _data, params, hover) => {
    const g = params.act === 'relu' ? '\\operatorname{ReLU}' : params.act === 'sigmoid' ? '\\sigma' : '\\tanh';
    const layer = `\\mathbf{a}^{(l)} = ${g}\\!\\left({\\color{${COLORS.yellow}}W^{(l)}}\\,\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}\\right)`;
    if (hover) return `${layer} \\qquad \\hat p(${tex(hover.x)},\\ ${tex(hover.y)}) = ${tex(s.net.predict(hover.x, hover.y))}`;
    return `${layer} \\qquad \\mathcal{L} = ${tex(s.loss, 4)}`;
  },
  metrics: (s, data) => [
    { label: 'Epoch', value: s.epoch },
    { label: 'Loss', value: fmt(s.loss, 4) },
    { label: 'Accuracy', value: pct(accuracy(data, (x, y) => s.net.predict(x, y))) },
    { label: 'Shape', value: s.net.sizes.join(' → ') },
    { label: 'Parameters', value: s.net.parameterCount },
  ],
  docs: {
    description:
      'A multilayer perceptron stacks layers of neurons with nonlinear activations. Each hidden layer reshapes the input space, and composing those reshapings lets the network draw curved boundaries, which solves XOR, circles and spirals. Training uses backpropagation: the output error travels backward through the layers by the chain rule, which gives the gradient of every weight, and an optimizer adjusts them. In the panel under the plot, each square shows how one neuron responds across the whole plane. Edges are blue for positive weights and red for negative ones, thicker when larger.',
    steps: [
      'Forward pass: each layer computes $\\mathbf{a}^{(l)} = g(W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)})$.',
      'The output goes through a sigmoid, and the loss is binary cross-entropy.',
      'Backward pass: the output error $\\delta^{(L)} = \\hat p - y$ is multiplied by the transposed weights and the activation derivative, layer by layer.',
      'The batch-averaged gradients update the weights and biases (SGD, Momentum or Adam).',
      'Each animation step is one full epoch over the data.',
    ],
    complexity: [['Training', 'O(n\\,P)\\ \\text{per epoch}'], ['Prediction', 'O(P)'], ['Memory', 'O(P)']],
    complexityNote: 'P is the total number of weights. Backpropagation costs only a constant factor more than the forward pass.',
    pseudocode: [
      'initialize W(l), b(l) at random',
      'for epoch = 1 to E:',
      '    for each batch:',
      '        for each (x, y) in the batch:',
      '            a(0) ← x',
      '            a(l) ← g(W(l)·a(l-1) + b(l))   for l = 1..L',
      '            δ(L) ← a(L) - y',
      '            ∇W(l) += δ(l) · a(l-1)ᵀ',
      '            ∇b(l) += δ(l)',
      "            δ(l-1) ← (W(l)ᵀ·δ(l)) ⊙ g'(a(l-1))",
      '        W, b ← optimizer(W, b, ∇ / batch size)',
      '    if loss < ε: stop',
    ],
    exercises: [
      'With one hidden layer of 2 neurons, can the network solve circles? XOR? Look at the activation maps to see why.',
      'Switch tanh to relu and compare the boundary shapes. ReLU networks draw polygons.',
      'Compare SGD, Momentum and Adam at the same learning rate on the spiral. Which reaches a low loss first?',
      'Add L2 regularization (weight decay) in optimizers.ts and watch its effect on noisy data.',
      'Implement a softmax output layer for more than two classes.',
    ],
  },
};
