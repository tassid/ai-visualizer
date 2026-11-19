/**
 * A dense neural network (MLP) written from scratch.
 *
 * Notation used in the comments:
 *   a(0) = input x
 *   z(l) = W(l) · a(l-1) + b(l)        pre-activation of layer l
 *   a(l) = g(z(l))                      activation (tanh, relu or sigmoid)
 *   a(L) = σ(z(L))                      output: probability of class 1
 *
 * The loss is binary cross-entropy. With a sigmoid output, its derivative
 * with respect to z(L) is simply a(L) - y, which keeps backpropagation short.
 */
import { sigmoid } from '../../core/math';
import { randn } from '../../core/random';

export type ActivationName = 'tanh' | 'relu' | 'sigmoid';

export interface Activation {
  f: (z: number) => number;
  /** Derivative written in terms of the ACTIVATION a = f(z), which we already store. */
  df: (a: number) => number;
  /** Output range, used to color the activation maps. */
  range: [number, number];
}

export const ACTIVATIONS: Record<ActivationName, Activation> = {
  tanh: { f: Math.tanh, df: (a) => 1 - a * a, range: [-1, 1] },
  relu: { f: (z) => (z > 0 ? z : 0), df: (a) => (a > 0 ? 1 : 0), range: [0, Infinity] },
  sigmoid: { f: sigmoid, df: (a) => a * (1 - a), range: [0, 1] },
};

/** Weights and biases of all layers. W[l][i][j] connects neuron j of layer l to neuron i of layer l+1. */
export interface Layers {
  W: Float64Array[][];
  B: Float64Array[];
}

export class Network {
  readonly sizes: number[];
  readonly act: Activation;
  params: Layers;

  constructor(sizes: number[], activation: ActivationName) {
    this.sizes = sizes;
    this.act = ACTIVATIONS[activation];
    this.params = { W: [], B: [] };
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      // Xavier (tanh/sigmoid) or He (ReLU) initialization keeps the variance of
      // activations stable from layer to layer, so gradients do not vanish.
      const scale = Math.sqrt((activation === 'relu' ? 2 : 1) / nIn);
      this.params.W.push(Array.from({ length: nOut }, () => Float64Array.from({ length: nIn }, () => randn() * scale)));
      this.params.B.push(new Float64Array(nOut));
    }
  }

  get depth() {
    return this.params.W.length;
  }

  get parameterCount() {
    return this.params.W.reduce((s, m, l) => s + m.length * m[0].length + this.params.B[l].length, 0);
  }

  /** Zeros shaped like the parameters (for gradients and optimizer state). */
  zerosLike(): Layers {
    return {
      W: this.params.W.map((m) => m.map((r) => new Float64Array(r.length))),
      B: this.params.B.map((b) => new Float64Array(b.length)),
    };
  }

  /** Forward pass. Returns the activations of every layer, input included. */
  forward(x: number, y: number): Float64Array[] {
    const acts: Float64Array[] = [Float64Array.of(x, y)];
    let a = acts[0];
    const L = this.depth;
    for (let l = 0; l < L; l++) {
      const W = this.params.W[l], b = this.params.B[l];
      const out = new Float64Array(W.length);
      for (let i = 0; i < W.length; i++) {
        let z = b[i];
        const row = W[i];
        for (let j = 0; j < row.length; j++) z += row[j] * a[j];
        out[i] = l === L - 1 ? sigmoid(z) : this.act.f(z);
      }
      acts.push(out);
      a = out;
    }
    return acts;
  }

  predict(x: number, y: number): number {
    return this.forward(x, y)[this.depth][0];
  }

  /**
   * Backpropagation for one example, ADDING the gradient into `grad`.
   *
   *   δ(L)   = a(L) - y
   *   ∂W(l)  = δ(l) · a(l-1)ᵀ
   *   ∂b(l)  = δ(l)
   *   δ(l-1) = (W(l)ᵀ · δ(l)) ⊙ g'(a(l-1))
   *
   * Returns the example's loss (cross-entropy).
   */
  backward(x: number, y: number, target: number, grad: Layers): number {
    const acts = this.forward(x, y);
    const L = this.depth;
    const out = acts[L][0];
    let delta = Float64Array.of(out - target);

    for (let l = L - 1; l >= 0; l--) {
      const a = acts[l];
      const W = this.params.W[l];
      for (let i = 0; i < delta.length; i++) {
        grad.B[l][i] += delta[i];
        const g = grad.W[l][i];
        for (let j = 0; j < a.length; j++) g[j] += delta[i] * a[j];
      }
      if (l > 0) {
        const prev = new Float64Array(a.length);
        for (let j = 0; j < a.length; j++) {
          let s = 0;
          for (let i = 0; i < delta.length; i++) s += W[i][j] * delta[i];
          prev[j] = s * this.act.df(a[j]);
        }
        delta = prev;
      }
    }
    return -(target * Math.log(out + 1e-9) + (1 - target) * Math.log(1 - out + 1e-9));
  }
}
