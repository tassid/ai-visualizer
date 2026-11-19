/**
 * DBSCAN: Density-Based Spatial Clustering of Applications with Noise (Ester et al., 1996).
 *
 * Definitions:
 *   ε-neighborhood of p: the points within distance ε of p (p included)
 *   core:   |neighborhood| ≥ minPts
 *   border: not core, but inside the neighborhood of a core point
 *   noise:  neither core nor border
 *
 * A cluster is the set of points density-reachable from a core point. The
 * expansion is a breadth-first search over a graph in which core points
 * connect to their neighbors.
 */
import { sqDist } from '../../core/math';
import { range, shuffle } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface DbscanParams {
  eps: number;
  minPts: number;
}

export const NOISE = -1;
export const UNSEEN = -2;

export interface DbscanState {
  /** Cluster of each point: UNSEEN, NOISE or 0, 1, 2... */
  label: Int32Array;
  core: Uint8Array;
  visited: Uint8Array;
  queue: number[];
  clusters: number;
  current: number;
}

/** Brute-force ε-neighborhood, O(n). A k-d tree or a spatial grid would bring it near O(log n). */
export function region(data: Point[], i: number, eps: number): number[] {
  return regionAt(data, data[i].x, data[i].y, eps);
}

export function regionAt(data: Point[], x: number, y: number, eps: number): number[] {
  const out: number[] = [], e2 = eps * eps;
  for (let j = 0; j < data.length; j++) if (sqDist(x, y, data[j].x, data[j].y) <= e2) out.push(j);
  return out;
}

export function* train(data: Point[], P: DbscanParams, s: DbscanState): Generator<StepInfo, StepInfo> {
  const inQueue = new Uint8Array(data.length);
  const unlabeled = (j: number) => s.label[j] === UNSEEN || s.label[j] === NOISE;

  for (const p of shuffle(range(data.length))) {                        // line 1
    if (s.visited[p]) continue;
    s.visited[p] = 1;                                                   // line 2
    s.current = p;
    const N = region(data, p, P.eps);                                   // line 3
    if (N.length < P.minPts) {                                          // line 4
      s.label[p] = NOISE;
      yield { line: 4, message: `Only ${N.length} neighbor(s) within ε (fewer than ${P.minPts}): noise, for now.` };
      continue;
    }
    const C = s.clusters++;                                             // line 6
    s.label[p] = C;
    s.core[p] = 1;
    for (const j of N) if (j !== p && unlabeled(j) && !inQueue[j]) { s.queue.push(j); inQueue[j] = 1; }
    yield { line: 6, message: `A core point with ${N.length} neighbors starts cluster ${C + 1}; ${s.queue.length} neighbors join the queue.` };

    while (s.queue.length) {                                            // line 7
      const q = s.queue.shift()!;
      s.current = q;
      let promoted = false;
      if (!s.visited[q]) {                                              // line 8
        s.visited[q] = 1;
        const Nq = region(data, q, P.eps);                              // line 9
        if (Nq.length >= P.minPts) {                                    // line 10
          s.core[q] = 1;
          promoted = true;
          for (const j of Nq) if (unlabeled(j) && !inQueue[j]) { s.queue.push(j); inQueue[j] = 1; }
        }
      }
      const wasNoise = s.label[q] === NOISE;
      if (unlabeled(q)) s.label[q] = C;                                 // line 11
      yield {
        line: promoted ? 10 : 11,
        message: promoted
          ? `This neighbor is also a core point, so cluster ${C + 1} keeps growing (queue: ${s.queue.length}).`
          : `A border point joins cluster ${C + 1}${wasNoise ? ' (it had been marked as noise)' : ''}.`,
      };
    }
  }
  s.current = -1;
  const noise = s.label.reduce((a, l) => a + (l === NOISE ? 1 : 0), 0);
  return { line: 1, message: `Done: ${s.clusters} cluster(s) and ${noise} noise point(s).` };
}

export function createDbscan(data: Point[], P: DbscanParams): Trainer<DbscanState> {
  const n = data.length;
  const state: DbscanState = { label: new Int32Array(n).fill(UNSEEN), core: new Uint8Array(n), visited: new Uint8Array(n), queue: [], clusters: 0, current: -1 };
  return { state, steps: train(data, P, state) };
}
