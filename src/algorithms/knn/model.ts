/**
 * K vizinhos mais próximos (KNN).
 *
 * "Treinar" é só guardar os dados. Toda a computação acontece na predição:
 *   1. distância da consulta até todos os n exemplos   → O(n·d)
 *   2. selecionar os k menores                         → O(n log n) ordenando
 *   3. votação                                         → O(k)
 *
 * A animação varre uma grade de células e classifica cada uma, mostrando
 * os vizinhos usados. Com pesos por distância, vizinhos mais próximos
 * valem mais no voto.
 */
import { sqDist } from '../../core/math';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface KnnParams {
  k: number;
  res: number;
  weighting: 'uniforme' | 'distância';
}

export interface Neighborhood {
  x: number;
  y: number;
  /** Índices dos k vizinhos mais próximos. */
  idx: number[];
  /** Distância até o k-ésimo vizinho (raio da vizinhança). */
  radius: number;
  /** Fração (ponderada) de votos na classe 1. */
  p: number;
}

export interface KnnState {
  res: number;
  /** Probabilidade da classe 1 em cada célula; -1 = ainda não classificada. */
  cells: Float32Array;
  done: number;
  current: Neighborhood | null;
  looAccuracy: number;
}

/** Os k vizinhos mais próximos de (x, y). `exclude` permite deixar um ponto de fora (leave-one-out). */
export function neighbors(data: Point[], x: number, y: number, k: number, weighting: KnnParams['weighting'] = 'uniforme', exclude = -1): Neighborhood {
  const d = data.map((p, i) => [i === exclude ? Infinity : sqDist(p.x, p.y, x, y), i] as [number, number]);
  d.sort((a, b) => a[0] - b[0]);
  const top = d.slice(0, Math.min(k, data.length - (exclude >= 0 ? 1 : 0)));
  let votes = 0, total = 0;
  for (const [dist, i] of top) {
    const w = weighting === 'distância' ? 1 / (Math.sqrt(dist) + 1e-6) : 1;
    votes += w * data[i].c;
    total += w;
  }
  return { x, y, idx: top.map((e) => e[1]), radius: top.length ? Math.sqrt(top[top.length - 1][0]) : 0, p: total ? votes / total : 0.5 };
}

/** Acurácia leave-one-out: classifica cada ponto usando todos os outros. */
export function leaveOneOut(data: Point[], k: number, weighting: KnnParams['weighting']): number {
  if (data.length < 2) return 0;
  let ok = 0;
  data.forEach((p, i) => {
    const pr = neighbors(data, p.x, p.y, k, weighting, i).p;
    if ((pr > 0.5 ? 1 : pr < 0.5 ? 0 : p.c) === p.c) ok++;
  });
  return ok / data.length;
}

export function* train(data: Point[], P: KnnParams, s: KnnState): Generator<StepInfo, StepInfo> {
  if (!data.length) return { line: 0, message: 'Sem dados.' };
  const R = s.res;
  for (let c = 0; c < R * R; c++) {
    const x = -1 + ((c % R) + 0.5) / R * 2;
    const y = 1 - (Math.floor(c / R) + 0.5) / R * 2;
    const nb = neighbors(data, x, y, P.k, P.weighting);                // linhas 1 a 5
    s.cells[c] = nb.p;
    s.current = nb;
    s.done = c + 1;
    const votesB = Math.round(nb.p * nb.idx.length);
    yield { line: 5, message: `Célula ${c + 1}: ${votesB} de ${nb.idx.length} vizinhos são da classe B, então ela fica ${nb.p > 0.5 ? 'B' : nb.p < 0.5 ? 'A' : 'empatada'}.` };
  }
  s.current = null;
  return { line: 5, message: 'Plano inteiro classificado. Passe o mouse para consultar qualquer ponto.' };
}

export function createKnn(data: Point[], P: KnnParams): Trainer<KnnState> {
  const res = Number(P.res);
  const state: KnnState = { res, cells: new Float32Array(res * res).fill(-1), done: 0, current: null, looAccuracy: leaveOneOut(data, P.k, P.weighting) };
  return { state, steps: train(data, P, state) };
}
