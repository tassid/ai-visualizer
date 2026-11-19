/**
 * k-nearest neighbors (KNN).
 *
 * "Training" only stores the data. All the work happens at prediction:
 *   1. distance from the query to all n examples   → O(n·d)
 *   2. pick the k smallest                         → O(n log n) by sorting
 *   3. vote                                        → O(k)
 *
 * The animation sweeps a grid of cells and classifies each one, showing the
 * neighbors it used. With distance weighting, closer neighbors count more.
 */
import { sqDist } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export type Weighting = 'uniform' | 'distance';

export interface KnnParams {
  k: number;
  res: number;
  weighting: Weighting;
}

export interface Neighborhood {
  x: number;
  y: number;
  /** Indices of the k nearest neighbors. */
  idx: number[];
  /** Distance to the k-th neighbor (radius of the neighborhood). */
  radius: number;
  /** (Weighted) share of votes for class 1. */
  p: number;
}

export interface KnnState {
  res: number;
  /** Probability of class 1 for each cell; -1 = not classified yet. */
  cells: Float32Array;
  done: number;
  current: Neighborhood | null;
  looAccuracy: number;
}

/** The k nearest neighbors of (x, y). `exclude` leaves one point out (leave-one-out). */
export function neighbors(data: Point[], x: number, y: number, k: number, weighting: Weighting = 'uniform', exclude = -1): Neighborhood {
  const d = data.map((p, i) => [i === exclude ? Infinity : sqDist(p.x, p.y, x, y), i] as [number, number]);
  d.sort((a, b) => a[0] - b[0]);
  const top = d.slice(0, Math.min(k, data.length - (exclude >= 0 ? 1 : 0)));
  let votes = 0, total = 0;
  for (const [dist, i] of top) {
    const w = weighting === 'distance' ? 1 / (Math.sqrt(dist) + 1e-6) : 1;
    votes += w * data[i].c;
    total += w;
  }
  return { x, y, idx: top.map((e) => e[1]), radius: top.length ? Math.sqrt(top[top.length - 1][0]) : 0, p: total ? votes / total : 0.5 };
}

/** Leave-one-out accuracy: classify each point using all the others. */
export function leaveOneOut(data: Point[], k: number, weighting: Weighting): number {
  if (data.length < 2) return 0;
  let ok = 0;
  data.forEach((p, i) => {
    const pr = neighbors(data, p.x, p.y, k, weighting, i).p;
    if ((pr > 0.5 ? 1 : pr < 0.5 ? 0 : p.c) === p.c) ok++;
  });
  return ok / data.length;
}

export function* train(data: Point[], P: KnnParams, s: KnnState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'No data. Click the plane to add points.' };
  const R = s.res;
  for (let c = 0; c < R * R; c++) {
    const x = -1 + (((c % R) + 0.5) / R) * 2;
    const y = 1 - ((Math.floor(c / R) + 0.5) / R) * 2;
    const nb = neighbors(data, x, y, P.k, P.weighting);                // lines 1 to 5
    s.cells[c] = nb.p;
    s.current = nb;
    s.done = c + 1;
    const votesB = Math.round(nb.p * nb.idx.length);
    yield { line: 5, message: `Cell ${c + 1}: ${votesB} of ${nb.idx.length} neighbors are class B, so it becomes ${nb.p > 0.5 ? 'B' : nb.p < 0.5 ? 'A' : 'a tie'}.` };
  }
  s.current = null;
  return { line: 5, message: 'Every cell classified.' };
}

export function createKnn(data: Point[], P: KnnParams): Trainer<KnnState> {
  const res = Number(P.res);
  const state: KnnState = { res, cells: new Float32Array(res * res).fill(-1), done: 0, current: null, looAccuracy: leaveOneOut(data, P.k, P.weighting) };
  return { state, steps: train(data, P, state) };
}
