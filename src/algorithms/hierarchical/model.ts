/**
 * Agglomerative hierarchical clustering.
 *
 * Start with every point in its own cluster and repeatedly merge the two
 * closest clusters until one remains. "Closest" depends on the linkage,
 * and all four classic linkages fit the Lance-Williams update, which gives
 * the distance from a new merged cluster (a ∪ b) to any other cluster i
 * from the old distances alone:
 *   single:   min(d_ai, d_bi)
 *   complete: max(d_ai, d_bi)
 *   average:  (n_a d_ai + n_b d_bi) / (n_a + n_b)
 *   Ward:     ((n_a + n_i) d_ai + (n_b + n_i) d_bi - n_i d_ab) / (n_a + n_b + n_i)   on squared distances
 *
 * The whole merge sequence is the dendrogram. Cutting it at a height gives a
 * flat clustering.
 */
import { sqDist } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export type Linkage = 'single' | 'complete' | 'average' | 'ward';

export interface HcParams {
  linkage: Linkage;
  k: number;
}

export interface Merge {
  a: number;
  b: number;
  /** Id of the new cluster: n, n + 1, ... (ids below n are the points). */
  id: number;
  height: number;
  size: number;
}

export interface HcState {
  merges: Merge[];
  /** Cluster id of each point after the merges done so far. */
  owner: Int32Array;
  /** Members of each live cluster id. */
  members: Map<number, number[]>;
  /** The two clusters joined in the last step, for highlighting. */
  last: Merge | null;
  /** Final labels when the dendrogram is cut into k clusters. */
  cut: Int32Array | null;
}

export function* train(data: Point[], P: HcParams, s: HcState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (n < 2) return { line: 0, message: 'Add at least two points.' };
  const ward = P.linkage === 'ward';
  // distance matrix between live clusters (squared for Ward, plain otherwise)
  const D = new Map<number, Map<number, number>>();
  const size = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    D.set(i, new Map());
    size.set(i, 1);
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const d2 = sqDist(data[i].x, data[i].y, data[j].x, data[j].y);
    const d = ward ? d2 : Math.sqrt(d2);
    D.get(i)!.set(j, d);
    D.get(j)!.set(i, d);
  }

  for (let step = 0; step < n - 1; step++) {                            // line 1
    let best = Infinity, ba = -1, bb = -1;
    for (const [a, row] of D) for (const [b, d] of row) if (a < b && d < best) { best = d; ba = a; bb = b; } // line 2
    const id = n + step, na = size.get(ba)!, nb = size.get(bb)!;
    const row = new Map<number, number>();
    for (const [i, dai] of D.get(ba)!) {                                // lines 3 and 4: Lance-Williams
      if (i === bb) continue;
      const dbi = D.get(bb)!.get(i)!, ni = size.get(i)!;
      let d: number;
      if (P.linkage === 'single') d = Math.min(dai, dbi);
      else if (P.linkage === 'complete') d = Math.max(dai, dbi);
      else if (P.linkage === 'average') d = (na * dai + nb * dbi) / (na + nb);
      else d = ((na + ni) * dai + (nb + ni) * dbi - ni * best) / (na + nb + ni);
      row.set(i, d);
      D.get(i)!.delete(ba);
      D.get(i)!.delete(bb);
      D.get(i)!.set(id, d);
    }
    D.delete(ba);
    D.delete(bb);
    D.set(id, row);
    size.delete(ba);
    size.delete(bb);
    size.set(id, na + nb);

    const merge: Merge = { a: ba, b: bb, id, height: ward ? Math.sqrt(best) : best, size: na + nb };
    s.merges.push(merge);
    s.last = merge;
    const joined = [...s.members.get(ba)!, ...s.members.get(bb)!];
    s.members.delete(ba);
    s.members.delete(bb);
    s.members.set(id, joined);
    for (const i of joined) s.owner[i] = id;
    yield { line: 3, message: `Merge ${step + 1}: two clusters at distance ${merge.height.toFixed(3)} become one of ${na + nb} points. ${n - step - 1} cluster(s) left.` };
  }
  s.last = null;
  s.cut = cutTree(n, s.merges, P.k);
  return { line: 6, message: `The dendrogram is complete. Cut into ${P.k} clusters, shown in color.` };
}

/** Labels after undoing the last k - 1 merges. */
export function cutTree(n: number, merges: Merge[], k: number): Int32Array {
  const parent = new Int32Array(n + merges.length).map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  merges.slice(0, Math.max(0, merges.length - (k - 1))).forEach((m) => { parent[find(m.a)] = m.id; parent[find(m.b)] = m.id; });
  const roots = new Map<number, number>();
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!roots.has(r)) roots.set(r, roots.size);
    out[i] = roots.get(r)!;
  }
  return out;
}

export function createHc(data: Point[], P: HcParams): Trainer<HcState> {
  const n = data.length;
  const state: HcState = { merges: [], owner: new Int32Array(n).map((_, i) => i), members: new Map(data.map((_, i) => [i, [i]])), last: null, cut: null };
  return { state, steps: train(data, P, state) };
}
