/**
 * K-Means (algoritmo de Lloyd).
 *
 * Minimiza a inércia: soma das distâncias quadradas de cada ponto ao seu centroide.
 * Alterna dois passos, e NENHUM deles aumenta a inércia:
 *   atribuição: cada ponto vai para o centroide mais próximo
 *   atualização: cada centroide vai para a média dos seus pontos
 * Como há um número finito de partições, o algoritmo sempre para; mas pode
 * parar num mínimo local, por isso a inicialização importa.
 *
 * k-means++ (Arthur e Vassilvitskii, 2007): escolhe cada novo centroide com
 * probabilidade proporcional à distância² até o centroide mais próximo já
 * escolhido. Isso espalha os centroides e dá garantia de O(log k) do ótimo.
 */
import { randInt, random, uniform } from '../../core/random';
import { sqDist } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface KMeansParams {
  k: number;
  init: 'k-means++' | 'aleatória';
  animateAssign: 'sim' | 'não';
}

export interface KMeansState {
  centroids: { x: number; y: number }[];
  /** Grupo de cada ponto; -1 = ainda sem grupo. */
  assign: Int16Array;
  phase: 'atribuição' | 'atualização';
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
  if (method === 'aleatória') {
    for (let i = 0; i < k; i++) C.push({ x: uniform(-0.9, 0.9), y: uniform(-0.9, 0.9) });
    return C;
  }
  const first = data[randInt(data.length)];
  C.push({ x: first.x, y: first.y });
  while (C.length < k) {
    const w = data.map((p) => nearest(C, p.x, p.y)[1]);                  // D(x)²
    let r = random() * w.reduce((a, b) => a + b, 0);
    let i = 0;
    while (i < data.length - 1 && r > w[i]) { r -= w[i]; i++; }         // sorteio por roleta
    C.push({ x: data[i].x, y: data[i].y });
  }
  return C;
}

export const inertia = (data: Point[], s: KMeansState) =>
  data.reduce((sum, p, i) => sum + (s.assign[i] >= 0 ? sqDist(p.x, p.y, s.centroids[s.assign[i]].x, s.centroids[s.assign[i]].y) : 0), 0);

export function* train(data: Point[], P: KMeansParams, s: KMeansState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'Sem dados.' };
  for (let iter = 0; ; iter++) {
    // Passo de atribuição
    s.phase = 'atribuição';
    let changed = 0;
    for (let i = 0; i < data.length; i++) {                             // linha 2
      const [best] = nearest(s.centroids, data[i].x, data[i].y);        // linha 3
      if (s.assign[i] !== best) { s.assign[i] = best; changed++; }
      s.current = i;
      if (P.animateAssign === 'sim') yield { line: 3, message: `Ponto ${i + 1} vai para o centroide mais próximo (${best + 1}).` };
    }
    s.current = -1;
    s.changed = changed;
    s.inertia = inertia(data, s);
    s.history.push(s.inertia);
    if (changed === 0 && iter > 0) return { line: 6, message: `Convergiu após ${iter} iterações: nenhum ponto mudou de grupo.` };
    yield { line: 3, message: `Atribuição: ${changed} ponto(s) mudaram de grupo. Inércia = ${s.inertia.toFixed(3)}.` };

    // Passo de atualização
    s.phase = 'atualização';
    s.centroids.forEach((c, k) => {                                     // linha 4
      let sx = 0, sy = 0, n = 0;
      data.forEach((p, i) => { if (s.assign[i] === k) { sx += p.x; sy += p.y; n++; } });
      if (n) { c.x = sx / n; c.y = sy / n; }                            // linha 5 (grupo vazio fica parado)
      s.trails[k].push([c.x, c.y]);
    });
    s.iter = iter + 1;
    s.inertia = inertia(data, s);
    s.history.push(s.inertia);
    yield { line: 5, message: `Atualização: centroides vão para a média dos seus grupos. Inércia = ${s.inertia.toFixed(3)}.` };
  }
}

export function createKMeans(data: Point[], P: KMeansParams): Trainer<KMeansState> {
  const centroids = initCentroids(data, Math.min(P.k, Math.max(1, data.length)), P.init);
  const state: KMeansState = {
    centroids,
    assign: new Int16Array(data.length).fill(-1),
    phase: 'atribuição',
    iter: 0,
    changed: 0,
    inertia: NaN,
    current: -1,
    trails: centroids.map((c) => [[c.x, c.y]]),
    history: [],
  };
  return { state, steps: train(data, P, state) };
}
