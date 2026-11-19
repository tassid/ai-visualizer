/**
 * Naive Bayes gaussiano.
 *
 * Pelo teorema de Bayes:  P(k | x) ∝ P(k) · P(x | k)
 * A hipótese "ingênua" é que os atributos são independentes dada a classe:
 *   P(x | k) = Πⱼ N(xⱼ; μₖⱼ, σ²ₖⱼ)
 *
 * Treinar é só estimar médias e variâncias por classe. Aqui fazemos isso de
 * forma ONLINE, um ponto por vez, com o algoritmo de Welford: ele atualiza
 * média e variância sem guardar os dados e é numericamente estável.
 *
 *   n ← n + 1
 *   δ ← x - μ
 *   μ ← μ + δ / n
 *   M ← M + δ · (x - μ)          (variância = M / n)
 */
import { gaussianPdf } from '../../core/math';
import { range, shuffle } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface NaiveBayesParams {
  smoothing: number;
}

export interface ClassStats {
  n: number;
  mean: [number, number];
  m2: [number, number];
}

export interface NaiveBayesState {
  stats: [ClassStats, ClassStats];
  seen: number;
  current: number;
  smoothing: number;
}

const emptyStats = (): ClassStats => ({ n: 0, mean: [0, 0], m2: [0, 0] });

export const variance = (st: ClassStats, j: number, smoothing: number) => (st.n > 1 ? st.m2[j] / st.n : 0.05) + smoothing;

/** P(classe 1 | x, y). Retorna 0,5 enquanto alguma classe não tiver dados. */
export function posterior(s: NaiveBayesState, x: number, y: number): number {
  const [a, b] = s.stats;
  if (a.n < 1 || b.n < 1) return 0.5;
  const total = a.n + b.n;
  const like = (st: ClassStats) =>
    (st.n / total) * gaussianPdf(x, st.mean[0], variance(st, 0, s.smoothing)) * gaussianPdf(y, st.mean[1], variance(st, 1, s.smoothing));
  const la = like(a), lb = like(b);
  return la + lb === 0 ? 0.5 : lb / (la + lb);
}

/** Uma etapa do algoritmo de Welford. */
export function welford(st: ClassStats, x: number, y: number) {
  st.n++;
  const v = [x, y];
  for (let j = 0; j < 2; j++) {
    const delta = v[j] - st.mean[j];
    st.mean[j] += delta / st.n;
    st.m2[j] += delta * (v[j] - st.mean[j]);
  }
}

export function* train(data: Point[], P: NaiveBayesParams, s: NaiveBayesState): Generator<StepInfo, StepInfo> {
  for (const i of shuffle(range(data.length))) {                        // linha 1
    const p = data[i];
    s.current = i;
    welford(s.stats[p.c], p.x, p.y);                                    // linhas 2 a 4
    s.seen++;
    const st = s.stats[p.c];
    yield { line: 3, message: `Classe ${p.c ? 'B' : 'A'}: n = ${st.n}, μ = (${st.mean[0].toFixed(2)}, ${st.mean[1].toFixed(2)}).` };
  }
  s.current = -1;
  return { line: 6, message: 'Médias e variâncias estimadas. O fundo mostra P(B | x) pelo teorema de Bayes.' };
}

export function createNaiveBayes(data: Point[], P: NaiveBayesParams): Trainer<NaiveBayesState> {
  const state: NaiveBayesState = { stats: [emptyStats(), emptyStats()], seen: 0, current: -1, smoothing: P.smoothing };
  return { state, steps: train(data, P, state) };
}
