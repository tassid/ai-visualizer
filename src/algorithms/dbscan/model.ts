/**
 * DBSCAN: Density-Based Spatial Clustering of Applications with Noise (Ester et al., 1996).
 *
 * Definições:
 *   vizinhança-ε de p: pontos a distância ≤ ε de p (inclui o próprio p)
 *   núcleo:  |vizinhança| ≥ minPts
 *   borda:   não é núcleo, mas está na vizinhança de um núcleo
 *   ruído:   nem núcleo nem borda
 *
 * Um grupo é o conjunto de pontos alcançáveis por densidade a partir de um
 * núcleo. A expansão funciona como uma busca em largura (BFS) por um grafo
 * em que núcleos se conectam aos seus vizinhos.
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
  /** Grupo de cada ponto: UNSEEN, NOISE ou 0, 1, 2... */
  label: Int32Array;
  core: Uint8Array;
  visited: Uint8Array;
  queue: number[];
  clusters: number;
  current: number;
}

/** Vizinhança-ε por força bruta, O(n). Com uma KD tree ou grade espacial, cairia para ~O(log n). */
export function region(data: Point[], i: number, eps: number): number[] {
  const p = data[i], out: number[] = [], e2 = eps * eps;
  for (let j = 0; j < data.length; j++) if (sqDist(p.x, p.y, data[j].x, data[j].y) <= e2) out.push(j);
  return out;
}

export function* train(data: Point[], P: DbscanParams, s: DbscanState): Generator<StepInfo, StepInfo> {
  const inQueue = new Uint8Array(data.length);
  const unlabeled = (j: number) => s.label[j] === UNSEEN || s.label[j] === NOISE;

  for (const p of shuffle(range(data.length))) {                        // linha 1
    if (s.visited[p]) continue;
    s.visited[p] = 1;                                                   // linha 2
    s.current = p;
    const N = region(data, p, P.eps);                                   // linha 3
    if (N.length < P.minPts) {                                          // linha 4
      s.label[p] = NOISE;
      yield { line: 4, message: `Ponto com só ${N.length} vizinho(s) no raio ε (< ${P.minPts}): marcado como ruído, por enquanto.` };
      continue;
    }
    const C = s.clusters++;                                             // linha 6
    s.label[p] = C;
    s.core[p] = 1;
    for (const j of N) if (j !== p && unlabeled(j) && !inQueue[j]) { s.queue.push(j); inQueue[j] = 1; }
    yield { line: 6, message: `Núcleo com ${N.length} vizinhos: começa o grupo ${C + 1} e ${s.queue.length} vizinhos entram na fila.` };

    while (s.queue.length) {                                            // linha 7
      const q = s.queue.shift()!;
      s.current = q;
      let promoted = false;
      if (!s.visited[q]) {                                              // linha 8
        s.visited[q] = 1;
        const Nq = region(data, q, P.eps);                              // linha 9
        if (Nq.length >= P.minPts) {                                    // linha 10
          s.core[q] = 1;
          promoted = true;
          for (const j of Nq) if (unlabeled(j) && !inQueue[j]) { s.queue.push(j); inQueue[j] = 1; }
        }
      }
      const wasNoise = s.label[q] === NOISE;
      if (unlabeled(q)) s.label[q] = C;                                 // linha 11
      yield {
        line: promoted ? 10 : 11,
        message: promoted
          ? `Vizinho também é núcleo: o grupo ${C + 1} se expande (fila: ${s.queue.length}).`
          : `Vizinho de borda entra no grupo ${C + 1}${wasNoise ? ' (era ruído e foi resgatado)' : ''}.`,
      };
    }
  }
  s.current = -1;
  const noise = s.label.reduce((a, l) => a + (l === NOISE ? 1 : 0), 0);
  return { line: 1, message: `Fim: ${s.clusters} grupo(s) e ${noise} ponto(s) de ruído.` };
}

export function createDbscan(data: Point[], P: DbscanParams): Trainer<DbscanState> {
  const n = data.length;
  const state: DbscanState = { label: new Int32Array(n).fill(UNSEEN), core: new Uint8Array(n), visited: new Uint8Array(n), queue: [], clusters: 0, current: -1 };
  return { state, steps: train(data, P, state) };
}
