/**
 * k-means (Lloyd's algorithm).
 *
 * Minimizes the inertia: the sum of squared distances from each point to its
 * centroid. It alternates two steps, and NEITHER can increase the inertia:
 *   assignment: each point joins its nearest centroid
 *   update:     each centroid moves to the mean of its points
 * There are finitely many partitions, so the algorithm always stops, but it
 * can stop in a local minimum. That is why initialization matters.
 *
 * k-means++ (Arthur and Vassilvitskii, 2007) picks each new centroid with
 * probability proportional to its squared distance from the nearest centroid
 * already chosen. That spreads them out and guarantees O(log k) of the optimum.
 */
import { randInt, random, uniform } from '../../core/random';
import { sqDist } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface KMeansParams {
  k: number;
  init: 'k-means++' | 'random';
  animateAssign: 'yes' | 'no';
}

export interface KMeansState {
  centroids: { x: number; y: number }[];
  /** Cluster of each point; -1 = not assigned yet. */
  assign: Int16Array;
  phase: 'assignment' | 'update';
  iter: number;
  changed: number;
  inertia: number;
  current: number;
  trails: [number, number][][];
  history: number[];
}

export function nearest(centroids: { x: number; y: number }[], x: number, y: number): [number, number] {
  let best = 0, bestD = Infinity;
  centroids.forEach((c, i) => {
    const d = sqDist(c.x, c.y, x, y);
    if (d < bestD) { bestD = d; best = i; }
  });
  return [best, bestD];
}

export function initCentroids(data: Point[], k: number, method: KMeansParams['init']) {
  const C: { x: number; y: number }[] = [];
  if (!data.length) return C;
  if (method === 'random') {
    for (let i = 0; i < k; i++) C.push({ x: uniform(-0.9, 0.9), y: uniform(-0.9, 0.9) });
    return C;
  }
  const first = data[randInt(data.length)];
  C.push({ x: first.x, y: first.y });
  while (C.length < k) {
    const w = data.map((p) => nearest(C, p.x, p.y)[1]);                  // D(x)²
    let r = random() * w.reduce((a, b) => a + b, 0);
    let i = 0;
    while (i < data.length - 1 && r > w[i]) { r -= w[i]; i++; }         // roulette wheel draw
    C.push({ x: data[i].x, y: data[i].y });
  }
  return C;
}

export const inertia = (data: Point[], s: KMeansState) =>
  data.reduce((sum, p, i) => sum + (s.assign[i] >= 0 ? sqDist(p.x, p.y, s.centroids[s.assign[i]].x, s.centroids[s.assign[i]].y) : 0), 0);

export function* train(data: Point[], P: KMeansParams, s: KMeansState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'No data. Click the plane to add points.' };
  for (let iter = 0; ; iter++) {
    // Assignment step
    s.phase = 'assignment';
    let changed = 0;
    for (let i = 0; i < data.length; i++) {                             // line 2
      const [best] = nearest(s.centroids, data[i].x, data[i].y);        // line 3
      if (s.assign[i] !== best) { s.assign[i] = best; changed++; }
      s.current = i;
      if (P.animateAssign === 'yes') yield { line: 3, message: `Point ${i + 1} joins its nearest centroid, number ${best + 1}.` };
    }
    s.current = -1;
    s.changed = changed;
    s.inertia = inertia(data, s);
    s.history.push(s.inertia);
    if (changed === 0 && iter > 0) return { line: 6, message: `Converged after ${iter} iterations: no point changed cluster.` };
    yield { line: 3, message: `Assignment: ${changed} point(s) changed cluster. Inertia ${s.inertia.toFixed(3)}.` };

    // Update step
    s.phase = 'update';
    s.centroids.forEach((c, k) => {                                     // line 4
      let sx = 0, sy = 0, n = 0;
      data.forEach((p, i) => { if (s.assign[i] === k) { sx += p.x; sy += p.y; n++; } });
      if (n) { c.x = sx / n; c.y = sy / n; }                            // line 5 (an empty cluster stays put)
      s.trails[k].push([c.x, c.y]);
    });
    s.iter = iter + 1;
    s.inertia = inertia(data, s);
    s.history.push(s.inertia);
    yield { line: 5, message: `Update: each centroid moves to the mean of its cluster. Inertia ${s.inertia.toFixed(3)}.` };
  }
}

export function createKMeans(data: Point[], P: KMeansParams): Trainer<KMeansState> {
  const centroids = initCentroids(data, Math.min(P.k, Math.max(1, data.length)), P.init);
  const state: KMeansState = {
    centroids,
    assign: new Int16Array(data.length).fill(-1),
    phase: 'assignment',
    iter: 0,
    changed: 0,
    inertia: NaN,
    current: -1,
    trails: centroids.map((c) => [[c.x, c.y]]),
    history: [],
  };
  return { state, steps: train(data, P, state) };
}
