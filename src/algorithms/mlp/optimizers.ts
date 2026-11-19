/**
 * Optimizers: how a gradient becomes a weight update.
 *
 * SGD:       θ ← θ - η·g
 * Momentum:  v ← β·v + g ;  θ ← θ - η·v        (builds up velocity)
 * Adam:      m ← β1·m + (1-β1)·g                (running mean of the gradient)
 *            v ← β2·v + (1-β2)·g²               (running mean of its square)
 *            θ ← θ - η · m̂ / (sqrt(v̂) + ε)      (m̂, v̂ corrected for starting at zero)
 *
 * Adam gives every weight its own step size, which is why it usually
 * converges without much tuning.
 */
import type { Layers, Network } from './network';

export type OptimizerName = 'Adam' | 'Momentum' | 'SGD';

export interface Optimizer {
  /** `grad` must already be divided by the batch size. */
  step(net: Network, grad: Layers): void;
}

/** Calls `fn` on every parameter vector together with the matching vectors of the other buffers. */
function forEachVector(net: Network, buffers: Layers[], fn: (p: Float64Array, ...rest: Float64Array[]) => void) {
  const { W, B } = net.params;
  for (let l = 0; l < W.length; l++) {
    for (let i = 0; i < W[l].length; i++) fn(W[l][i], ...buffers.map((b) => b.W[l][i]));
    fn(B[l], ...buffers.map((b) => b.B[l]));
  }
}

export function createOptimizer(name: OptimizerName, net: Network, lr: number): Optimizer {
  if (name === 'SGD') {
    return {
      step(n, grad) {
        forEachVector(n, [grad], (p, g) => {
          for (let k = 0; k < p.length; k++) p[k] -= lr * g[k];
        });
      },
    };
  }

  if (name === 'Momentum') {
    const v = net.zerosLike();
    const beta = 0.9;
    return {
      step(n, grad) {
        forEachVector(n, [grad, v], (p, g, vel) => {
          for (let k = 0; k < p.length; k++) {
            vel[k] = beta * vel[k] + g[k];
            p[k] -= lr * vel[k];
          }
        });
      },
    };
  }

  const m = net.zerosLike(), v = net.zerosLike();
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  let t = 0;
  return {
    step(n, grad) {
      t++;
      const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t; // bias correction: m and v start at zero
      forEachVector(n, [grad, m, v], (p, g, mm, vv) => {
        for (let k = 0; k < p.length; k++) {
          mm[k] = b1 * mm[k] + (1 - b1) * g[k];
          vv[k] = b2 * vv[k] + (1 - b2) * g[k] * g[k];
          p[k] -= (lr * (mm[k] / c1)) / (Math.sqrt(vv[k] / c2) + eps);
        }
      });
    },
  };
}
