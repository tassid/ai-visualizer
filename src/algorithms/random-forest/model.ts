/**
 * Floresta aleatória (Breiman, 2001).
 *
 * Duas fontes de aleatoriedade deixam as árvores diferentes entre si:
 *   1. bagging: cada árvore treina numa amostra bootstrap (n sorteios com reposição);
 *   2. subespaço aleatório: cada corte só pode usar um subconjunto dos atributos.
 *
 * A média de modelos com erros pouco correlacionados tem variância menor que
 * a de cada modelo. É por isso que a fronteira da floresta fica mais suave
 * do que a de uma árvore só.
 *
 * Cerca de 36,8% dos pontos ficam fora de cada amostra bootstrap
 * (a probabilidade de nunca ser sorteado é (1 - 1/n)ⁿ ≈ 1/e). Esses pontos
 * "out of bag" servem como um conjunto de validação gratuito.
 */
import { randInt, random } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';
import { applySplit, bestSplit, makeNode, predictTree, type TreeNode } from '../shared/tree';

export interface ForestParams {
  trees: number;
  depth: number;
  maxFeatures: 'todos' | '1 aleatório';
}

export interface ForestState {
  trees: TreeNode[];
  /** inBag[t][i] = 1 se o ponto i entrou na amostra da árvore t. */
  inBag: Uint8Array[];
  oobAccuracy: number;
  history: number[];
}

/** Cresce uma árvore completa (sem animação) numa amostra. */
export function growTree(data: Point[], idx: number[], P: ForestParams): TreeNode {
  const root = makeNode(data, idx, 0, [-1, 1, -1, 1]);
  const stack = [root];
  while (stack.length) {
    const node = stack.pop()!;
    if (node.depth >= P.depth || node.n < 2 || node.prob === 0 || node.prob === 1) { node.leaf = true; continue; }
    let split;
    if (P.maxFeatures === 'todos') split = bestSplit(data, node.idx, [0, 1]);
    else {
      const f: 0 | 1 = random() < 0.5 ? 0 : 1;
      split = bestSplit(data, node.idx, [f]) ?? bestSplit(data, node.idx, [f === 0 ? 1 : 0]);
    }
    if (!split) { node.leaf = true; continue; }
    stack.push(...applySplit(data, node, split));
  }
  return root;
}

export const forestPredict = (s: ForestState, x: number, y: number) => {
  if (!s.trees.length) return 0.5;
  let sum = 0;
  for (const t of s.trees) sum += predictTree(t, x, y);
  return sum / s.trees.length;
};

/** Acurácia out of bag: cada ponto é avaliado só pelas árvores que não o viram. */
export function outOfBag(data: Point[], s: ForestState): number {
  let ok = 0, counted = 0;
  data.forEach((p, i) => {
    let sum = 0, c = 0;
    s.trees.forEach((t, k) => { if (!s.inBag[k][i]) { sum += predictTree(t, p.x, p.y); c++; } });
    if (c) { counted++; if ((sum / c > 0.5 ? 1 : 0) === p.c) ok++; }
  });
  return counted ? ok / counted : NaN;
}

export function* train(data: Point[], P: ForestParams, s: ForestState): Generator<StepInfo, StepInfo> {
  const n = data.length;
  if (!n) return { line: 0, message: 'Sem dados.' };
  for (let t = 1; t <= P.trees; t++) {                                  // linha 1
    const idx: number[] = [];
    const bag = new Uint8Array(n);
    for (let k = 0; k < n; k++) { const i = randInt(n); idx.push(i); bag[i] = 1; } // linha 2
    s.trees.push(growTree(data, idx, P));                               // linha 3
    s.inBag.push(bag);                                                  // linha 4
    s.oobAccuracy = outOfBag(data, s);
    s.history.push(1 - s.oobAccuracy);
    const unique = bag.reduce((a, b) => a + b, 0);
    yield { line: 3, message: `Árvore ${t}: amostra com ${unique} pontos distintos (${((1 - unique / n) * 100).toFixed(0)}% ficaram de fora). OOB = ${(s.oobAccuracy * 100).toFixed(1)}%.` };
  }
  return { line: 6, message: `Floresta com ${P.trees} árvores pronta. A previsão é a média de todas.` };
}

export function createForest(data: Point[], P: ForestParams): Trainer<ForestState> {
  const state: ForestState = { trees: [], inBag: [], oobAccuracy: NaN, history: [] };
  return { state, steps: train(data, P, state) };
}
