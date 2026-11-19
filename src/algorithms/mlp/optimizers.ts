/**
 * Otimizadores: como transformar o gradiente em uma atualização dos pesos.
 *
 * SGD:       θ ← θ - η·g
 * Momentum:  v ← β·v + g ;  θ ← θ - η·v        (acumula "inércia")
 * Adam:      m ← β1·m + (1-β1)·g                (média do gradiente)
 *            v ← β2·v + (1-β2)·g²               (média do gradiente ao quadrado)
 *            θ ← θ - η · m̂ / (sqrt(v̂) + ε)      (m̂, v̂ corrigidos pelo viés inicial)
 *
 * Adam dá a cada peso uma taxa de aprendizado própria, por isso costuma
 * convergir rápido sem ajuste fino.
 */
import type { Layers, Network } from './network';

export type OptimizerName = 'Adam' | 'Momentum' | 'SGD';

export interface Optimizer {
  /** `grad` já deve estar dividido pelo tamanho do lote. */
  step(net: Network, grad: Layers): void;
}

/** Aplica `fn` a cada vetor de parâmetros, junto com os vetores correspondentes dos outros buffers. */
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
      const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t; // correção de viés: m e v começam em zero
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
