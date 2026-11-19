/**
 * Estruturas e funções de árvore de decisão, compartilhadas pela árvore
 * simples e pela floresta aleatória.
 *
 * Critério de impureza: Gini.  G(p) = 1 - p² - (1-p)² = 2p(1-p)
 *   G = 0   → nó puro (só uma classe)
 *   G = 0,5 → metade de cada classe (máxima incerteza)
 *
 * Um corte é bom quando a média ponderada do Gini dos filhos é bem menor
 * que o Gini do pai. A diferença é o "ganho de impureza".
 */
import type { DiagramNode } from '../../components/charts/TreeDiagram';
import type { Point } from '../../core/types';

/** Retângulo [x0, x1, y0, y1] que o nó ocupa no plano. */
export type Box = [number, number, number, number];

export interface TreeNode {
  id: number;
  idx: number[];
  depth: number;
  box: Box;
  /** Fração da classe 1 entre os exemplos do nó. */
  prob: number;
  n: number;
  leaf: boolean;
  /** Atributo do corte: 0 = x, 1 = y. */
  feature?: 0 | 1;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface Split {
  feature: 0 | 1;
  threshold: number;
  /** Gini ponderado dos filhos. */
  gini: number;
  gain: number;
}

export const gini = (p: number) => 2 * p * (1 - p);

let nextId = 0;

export function makeNode(data: Point[], idx: number[], depth: number, box: Box): TreeNode {
  let pos = 0;
  for (const i of idx) pos += data[i].c;
  return { id: nextId++, idx, depth, box, prob: idx.length ? pos / idx.length : 0.5, n: idx.length, leaf: false };
}

const value = (p: Point, f: 0 | 1) => (f === 0 ? p.x : p.y);

/**
 * Melhor corte entre os atributos permitidos.
 *
 * Truque para ficar O(n log n) por atributo em vez de O(n²): ordenamos os
 * exemplos pelo atributo e varremos da esquerda para a direita, mantendo a
 * contagem acumulada da classe 1. Assim cada limiar candidato custa O(1).
 */
export function bestSplit(data: Point[], idx: number[], features: (0 | 1)[]): Split | null {
  const n = idx.length;
  if (n < 2) return null;
  let total = 0;
  for (const i of idx) total += data[i].c;
  const parent = gini(total / n);
  let best: Split | null = null;

  for (const f of features) {
    const sorted = idx.slice().sort((a, b) => value(data[a], f) - value(data[b], f));
    let left = 0;
    for (let k = 1; k < n; k++) {
      left += data[sorted[k - 1]].c;
      const v0 = value(data[sorted[k - 1]], f), v1 = value(data[sorted[k]], f);
      if (v1 === v0) continue; // não dá para separar valores iguais
      const g = (k * gini(left / k) + (n - k) * gini((total - left) / (n - k))) / n;
      const gain = parent - g;
      if (gain > 1e-9 && (!best || g < best.gini)) best = { feature: f, threshold: (v0 + v1) / 2, gini: g, gain };
    }
  }
  return best;
}

/** Divide o nó em dois filhos, recortando também o retângulo. */
export function applySplit(data: Point[], node: TreeNode, split: Split): [TreeNode, TreeNode] {
  node.feature = split.feature;
  node.threshold = split.threshold;
  const L: number[] = [], R: number[] = [];
  for (const i of node.idx) (value(data[i], split.feature) < split.threshold ? L : R).push(i);
  const [x0, x1, y0, y1] = node.box, t = split.threshold;
  const boxL: Box = split.feature === 0 ? [x0, t, y0, y1] : [x0, x1, y0, t];
  const boxR: Box = split.feature === 0 ? [t, x1, y0, y1] : [x0, x1, t, y1];
  node.left = makeNode(data, L, node.depth + 1, boxL);
  node.right = makeNode(data, R, node.depth + 1, boxR);
  return [node.left, node.right];
}

/** Desce a árvore até uma folha (ou um nó ainda não dividido). */
export function predictTree(node: TreeNode, x: number, y: number): number {
  let n = node;
  while (n.left && n.right) n = (n.feature === 0 ? x : y) < n.threshold! ? n.left : n.right;
  return n.prob;
}

export function countNodes(n: TreeNode | undefined): number {
  return n ? 1 + countNodes(n.left) + countNodes(n.right) : 0;
}

export function treeDepth(n: TreeNode): number {
  return n.left && n.right ? Math.max(treeDepth(n.left), treeDepth(n.right)) : n.depth;
}

/** Lista os cortes como segmentos no plano: [x0, y0, x1, y1]. */
export function splitSegments(n: TreeNode, out: [number, number, number, number][] = []) {
  if (!n.left || !n.right) return out;
  const [x0, x1, y0, y1] = n.box, t = n.threshold!;
  out.push(n.feature === 0 ? [t, y0, t, y1] : [x0, t, x1, t]);
  splitSegments(n.left, out);
  splitSegments(n.right, out);
  return out;
}

/** Converte para o formato do componente TreeDiagram. */
export function toDiagram(n: TreeNode): DiagramNode {
  return {
    id: n.id,
    prob: n.prob,
    pending: !n.leaf && !n.left,
    label: n.left ? `${n.feature === 0 ? 'x' : 'y'} < ${n.threshold!.toFixed(2)}` : undefined,
    children: n.left && n.right ? [toDiagram(n.left), toDiagram(n.right)] : undefined,
  };
}
