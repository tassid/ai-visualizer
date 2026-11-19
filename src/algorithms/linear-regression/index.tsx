import { theory } from './theory';
import { line } from 'd3-shape';
import { motion } from 'framer-motion';
import { useCallback } from 'react';
import { COLORS } from '../../core/colors';
import { fmt, linspace, tex, texTerm } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { LossLandscape } from '../../components/charts/LossLandscape';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Handle, Points, SMOOTH } from '../../components/plot/marks';
import { createLinReg, evaluate, mse, type LinRegParams, type LinRegState } from './model';

const XS = linspace(-1, 1, 160);
const Y = COLORS.yellow;

function Curve({ s }: { s: LinRegState }) {
  const { x, y } = usePlot();
  const d = line<number>().x((v) => x(v)).y((v) => y(evaluate(s, v)))(XS) ?? '';
  return <motion.path initial={false} animate={{ d }} transition={SMOOTH} fill="none" stroke={Y} strokeWidth={2.5} pointerEvents="none" />;
}

/**
 * The squared errors drawn as actual squares, the way 3Blue1Brown shows least
 * squares: the loss is the average area of these squares.
 */
function ResidualSquares({ s, data }: { s: LinRegState; data: VisualProps<LinRegState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {data.map((p, i) => {
        const yHat = evaluate(s, p.x);
        const side = Math.abs(y(yHat) - y(p.y));
        const top = Math.min(y(yHat), y(p.y));
        return <rect key={i} x={x(p.x)} y={top} width={side} height={side} fill={COLORS.red} fillOpacity={0.14} stroke={COLORS.red} strokeOpacity={0.6} strokeWidth={1} />;
      })}
    </g>
  );
}

/** Two handles on the line (degree 1): drag them and the line follows. */
function LineHandles({ s }: { s: LinRegState }) {
  const xa = -0.6, xb = 0.6;
  // Each handle stays at its own x and only moves up and down.
  const set = (which: 'a' | 'b', py: number) => {
    const ya = which === 'a' ? py : evaluate(s, xa);
    const yb = which === 'b' ? py : evaluate(s, xb);
    s.w[0] = (yb - ya) / (xb - xa);
    s.b = ya - s.w[0] * xa;
    s.path.push([s.w[0], s.b]);
  };
  return (
    <>
      <Handle title="Drag up or down to move the line" at={{ x: xa, y: evaluate(s, xa) }} onDrag={(_, py) => set('a', py)} />
      <Handle title="Drag up or down to move the line" at={{ x: xb, y: evaluate(s, xb) }} onDrag={(_, py) => set('b', py)} />
    </>
  );
}

function LinRegVisual({ state: s, data, params, compact }: VisualProps<LinRegState>) {
  return (
    <Plot>
      <ResidualSquares s={s} data={data} />
      <Curve s={s} />
      <Points data={data} color={() => COLORS.blue} radius={4.4} />
      {!compact && Number(params.deg) === 1 && <LineHandles s={s} />}
    </Plot>
  );
}

function LinRegLandscape({ state: s, data, params }: VisualProps<LinRegState>) {
  const loss = useCallback(
    (w: number, b: number) => {
      let sum = 0;
      for (const p of data) sum += (w * p.x + b - p.y) ** 2;
      return sum / Math.max(1, data.length);
    },
    [data],
  );
  if (Number(params.deg) !== 1) return <p className="aux-empty">The landscape needs exactly two parameters. Set the degree to 1.</p>;
  return <LossLandscape loss={loss} dataKey={data} path={s.path} width={560} height={250} />;
}

export const linearRegression: AlgorithmDef<LinRegState> = {
  id: 'linear-regression',
  name: 'Linear regression',
  group: 'Linear, probabilistic and kernel models',
  kind: 'regression',
  datasets: ['line', 'sine'],
  defaultDataset: 'line',
  defaultSpeed: 4,
  thumbnailSteps: 300,
  hint: 'Drag the two yellow dots to fit the line by hand and watch the squares shrink.',
  params: [
    { key: 'deg', label: 'Polynomial degree', type: 'range', min: 1, max: 7, step: 1, value: 1 },
    { key: 'lr', label: 'Learning rate η', type: 'select', options: [0.01, 0.05, 0.1, 0.3, 0.6, 1.2], value: 0.1 },
    { key: 'iters', label: 'Iterations', type: 'range', min: 50, max: 3000, step: 50, value: 300 },
  ],
  create: (data, params) => createLinReg(data, params as unknown as LinRegParams),
  Visual: LinRegVisual,
  aux: { title: 'Loss over w and b', Component: LinRegLandscape },
  history: { label: 'Mean squared error', get: (s) => s.history },
  equation: (s, data) => {
    const terms = s.w.slice(0, 4).map((w, k) => texTerm(w, k === 0 ? 'x' : `x^{${k + 1}}`, 2, Y)).join(' ');
    const more = s.w.length > 4 ? ' + \\cdots' : '';
    return `\\hat y = {\\color{${Y}}${tex(s.b)}} ${terms}${more} \\qquad \\tfrac{1}{n}\\textstyle\\sum (\\hat y_i - y_i)^2 = ${tex(mse(s, data), 4)}`;
  },
  metrics: (s) => [
    { label: 'Iteration', value: s.it },
    { label: 'Mean squared error', value: fmt(s.loss, 5) },
  ],
  docs: {
    ...theory,
    description:
      'Linear regression fits the line (or, with a higher degree, the polynomial) that minimizes the mean squared error between predictions and targets. Each red square has the residual as its side, so its area is the squared error; the loss is the average area. Here the fit runs by gradient descent: each iteration moves the coefficients a little in the direction that shrinks the error fastest. With degree 1, the panel below shows the loss over $w$ and $b$ as contour lines, with the path gradient descent walks to the bottom of the valley.',
    steps: [
      'Start with arbitrary coefficients.',
      'Compute the predictions $\\hat y = b + w_1 x + w_2 x^2 + \\cdots$ and the residuals $\\hat y - y$.',
      'Compute the gradient of the mean squared error for each coefficient.',
      'Step against the gradient, scaled by the learning rate.',
      'Repeat until the error stops falling.',
    ],
    complexity: [['Training (gradient)', 'O(n\\,d)\\ \\text{per iteration}'], ['Training (normal equation)', 'O(n\\,d^2 + d^3)'], ['Prediction', 'O(d)'], ['Memory', 'O(d)']],
    pseudocode: [
      'w ← initial values ; b ← 0',
      'repeat T times:',
      '    for each (x, y):  e ← ŷ(x) - y',
      '    ∂wₖ ← (2/n) · Σ e · xᵏ',
      '    ∂b ← (2/n) · Σ e',
      '    wₖ ← wₖ - η · ∂wₖ',
      '    b ← b - η · ∂b',
      '    if |∇| ≈ 0: stop',
    ],
    exercises: [
      'With degree 1, set η = 1.2 and watch the path on the landscape. Why does it zigzag or blow up?',
      'Fit the line by hand with the handles, then press play. How close were you to the minimum?',
      'On the sine curve, raise the degree from 1 to 7. Where does the fit become good? What happens with few points and a high degree?',
      'Compare the final result with normalEquation() in model.ts, and write a test for it.',
    ],
  },
};
