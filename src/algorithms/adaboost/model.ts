/**
 * AdaBoost with decision stumps (Freund and Schapire, 1997).
 *
 * Each round fits a stump (a single axis-aligned split) to WEIGHTED data,
 * then raises the weights of the points it got wrong, so the next stump
 * focuses on them. The final classifier is a weighted vote:
 *   F(x) = Σₜ αₜ hₜ(x),   αₜ = ½ ln((1 - εₜ) / εₜ)
 * where εₜ is the stump's weighted error and hₜ(x) ∈ {-1, +1}.
 */
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface AdaParams {
  rounds: number;
}

export interface Stump {
  feature: 0 | 1;
  threshold: number;
  /** +1: predict B when the feature is above the threshold; -1: the opposite. */
  polarity: 1 | -1;
  alpha: number;
  error: number;
}

export interface AdaState {
  weights: Float64Array;
  stumps: Stump[];
  history: number[];
}

export const stumpPredict = (st: Stump, x: number, y: number) => ((st.feature === 0 ? x : y) > st.threshold ? st.polarity : -st.polarity);

export function score(s: AdaState, x: number, y: number): number {
  let F = 0;
  for (const st of s.stumps) F += st.alpha * stumpPredict(st, x, y);
  return F;
}

/**
 * The stump with the lowest weighted error. Sort by each feature and sweep
 * thresholds, keeping running sums of the weights on each side.
 */
export function bestStump(data: Point[], w: Float64Array): Omit<Stump, 'alpha'> {
  let best: Omit<Stump, 'alpha'> = { feature: 0, threshold: 0, polarity: 1, error: Infinity };
  let totalPos = 0;
  data.forEach((p, i) => { if (p.c) totalPos += w[i]; });
  for (const f of [0, 1] as const) {
    const order = data.map((_, i) => i).sort((a, b) => (f === 0 ? data[a].x - data[b].x : data[a].y - data[b].y));
    // threshold below every point: everything is "above"
    let posBelow = 0, negBelow = 0;
    const totalNeg = 1 - totalPos;
    for (let k = 0; k <= order.length; k++) {
      const v0 = k === 0 ? -Infinity : f === 0 ? data[order[k - 1]].x : data[order[k - 1]].y;
      const v1 = k === order.length ? Infinity : f === 0 ? data[order[k]].x : data[order[k]].y;
      // equal values cannot be separated by a threshold between them
      if (!(k > 0 && k < order.length && v0 === v1)) {
        // polarity +1 predicts B above: errors are B points below and A points above
        const errPlus = posBelow + (totalNeg - negBelow);
        const errMinus = 1 - errPlus;
        const t = k === 0 ? -1.01 : k === order.length ? 1.01 : (v0 + v1) / 2;
        if (errPlus < best.error) best = { feature: f, threshold: t, polarity: 1, error: errPlus };
        if (errMinus < best.error) best = { feature: f, threshold: t, polarity: -1, error: errMinus };
      }
      if (k < order.length) {
        const i = order[k];
        if (data[i].c) posBelow += w[i]; else negBelow += w[i];
      }
    }
  }
  return best;
}

export function trainingError(s: AdaState, data: Point[]) {
  let wrong = 0;
  for (const p of data) if ((score(s, p.x, p.y) >= 0 ? 1 : 0) !== p.c) wrong++;
  return data.length ? wrong / data.length : 0;
}

export function* train(data: Point[], P: AdaParams, s: AdaState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'No data. Click the plane to add points.' };
  for (let t = 1; t <= P.rounds; t++) {                                 // line 1
    const st = bestStump(data, s.weights);                              // line 2
    const eps = Math.min(Math.max(st.error, 1e-10), 1 - 1e-10);
    const alpha = 0.5 * Math.log((1 - eps) / eps);                      // line 3
    s.stumps.push({ ...st, alpha });
    let z = 0;
    data.forEach((p, i) => {                                            // line 4
      const y = p.c ? 1 : -1;
      s.weights[i] *= Math.exp(-alpha * y * stumpPredict(s.stumps[s.stumps.length - 1], p.x, p.y));
      z += s.weights[i];
    });
    for (let i = 0; i < n; i++) s.weights[i] /= z;                     // line 5
    const err = trainingError(s, data);
    s.history.push(err);
    yield { line: 4, message: `Round ${t}: stump on ${st.feature === 0 ? 'x' : 'y'} with weighted error ${eps.toFixed(3)}, vote α = ${alpha.toFixed(2)}. Training error ${(err * 100).toFixed(1)}%.` };
    if (eps < 1e-9) return { line: 6, message: 'A stump classifies the weighted data perfectly. Stopping.' };
  }
  return { line: 6, message: `${P.rounds} stumps combined into one vote.` };
}

export function createAda(data: Point[], P: AdaParams): Trainer<AdaState> {
  const state: AdaState = { weights: new Float64Array(data.length).fill(1 / Math.max(1, data.length)), stumps: [], history: [] };
  return { state, steps: train(data, P, state) };
}
