/**
 * Árvore de decisão (CART com Gini), construída em largura (BFS).
 *
 * A versão recursiva clássica constrói um ramo inteiro antes do outro. Aqui
 * usamos uma fila para que a animação mostre a árvore crescendo nível por
 * nível, o que é mais fácil de acompanhar.
 */
import type { Point, StepInfo, Trainer } from '../../core/types';
import { applySplit, bestSplit, makeNode, type TreeNode } from '../shared/tree';

export interface TreeParams {
  depth: number;
  minSplit: number;
}

export interface TreeState {
  root: TreeNode;
  queue: TreeNode[];
  current: TreeNode | null;
  leaves: number;
}

export function* train(data: Point[], P: TreeParams, s: TreeState): Generator<StepInfo, StepInfo> {
  while (s.queue.length) {
    const node = s.queue.shift()!;                                      // linha 1
    s.current = node;
    const pure = node.prob === 0 || node.prob === 1;
    if (pure || node.depth >= P.depth || node.n < P.minSplit) {         // linha 2
      node.leaf = true;
      s.leaves++;
      const why = pure ? 'é puro' : node.depth >= P.depth ? 'atingiu a profundidade máxima' : 'tem poucas amostras';
      yield { line: 3, message: `Nó com ${node.n} pontos ${why}: vira folha (P(B) = ${node.prob.toFixed(2)}).` };
      continue;
    }
    const split = bestSplit(data, node.idx, [0, 1]);                    // linhas 4 a 6
    if (!split) {
      node.leaf = true;
      s.leaves++;
      yield { line: 3, message: 'Nenhum corte reduz a impureza: vira folha.' };
      continue;
    }
    const [l, r] = applySplit(data, node, split);                       // linha 7
    s.queue.push(l, r);                                                 // linha 8
    yield {
      line: 7,
      message: `Corte ${split.feature === 0 ? 'x' : 'y'} < ${split.threshold.toFixed(2)}: Gini cai ${split.gain.toFixed(3)} (${l.n} à esquerda, ${r.n} à direita).`,
    };
  }
  s.current = null;
  return { line: 0, message: `Árvore completa com ${s.leaves} folhas.` };
}

export function createTree(data: Point[], P: TreeParams): Trainer<TreeState> {
  const root = makeNode(data, data.map((_, i) => i), 0, [-1, 1, -1, 1]);
  const state: TreeState = { root, queue: [root], current: null, leaves: 0 };
  return { state, steps: train(data, P, state) };
}
