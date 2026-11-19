import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { TreeDiagram } from '../../components/charts/TreeDiagram';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, SPRING } from '../../components/plot/marks';
import { countNodes, predictTree, splitSegments, toDiagram, treeDepth, type TreeNode } from '../shared/tree';
import { createTree, type TreeParams, type TreeState } from './model';

/** Cortes desenhados com animação de "crescimento" a partir do centro. */
export function SplitLines({ root, color = 'rgba(255,255,255,.75)', width = 1.6 }: { root: TreeNode; color?: string; width?: number }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {splitSegments(root).map(([x0, y0, x1, y1]) => {
        const key = `${x0.toFixed(4)}-${y0.toFixed(4)}-${x1.toFixed(4)}-${y1.toFixed(4)}`;
        const cx = x((x0 + x1) / 2), cy = y((y0 + y1) / 2);
        return (
          <motion.line key={key} initial={{ x1: cx, y1: cy, x2: cx, y2: cy }} animate={{ x1: x(x0), y1: y(y0), x2: x(x1), y2: y(y1) }} transition={SPRING} stroke={color} strokeWidth={width} />
        );
      })}
    </g>
  );
}

function BoxOutline({ node, color, dash, width = 1 }: { node: TreeNode; color: string; dash?: string; width?: number }) {
  const { x, y } = usePlot();
  const [x0, x1, y0, y1] = node.box;
  return (
    <motion.rect
      initial={false}
      animate={{ x: x(x0) + 1, y: y(y1) + 1, width: Math.max(0, x(x1) - x(x0) - 2), height: Math.max(0, y(y0) - y(y1) - 2) }}
      transition={{ duration: 0.15 }}
      fill="none" stroke={color} strokeWidth={width} strokeDasharray={dash} pointerEvents="none"
    />
  );
}

function TreeVisual({ state: s, data, version, compact }: VisualProps<TreeState>) {
  return (
    <Plot interactive={false} heatmap={{ version, resolution: 112, color: (x, y) => binaryColor(predictTree(s.root, x, y)) }}>
      <SplitLines root={s.root} />
      {!compact && s.queue.map((q) => <BoxOutline key={q.id} node={q} color="rgba(255,255,255,.3)" dash="4 5" />)}
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && s.current && <BoxOutline node={s.current} color={COLORS.accent} width={2.5} />}
    </Plot>
  );
}

function TreeAux({ state: s }: VisualProps<TreeState>) {
  return <TreeDiagram root={toDiagram(s.root)} currentId={s.current?.id} width={560} height={250} />;
}

export const decisionTree: AlgorithmDef<TreeState> = {
  id: 'arvore-decisao',
  name: 'Árvore de decisão',
  group: 'Instâncias e árvores',
  tag: 'Partição recursiva',
  kind: 'classification',
  summary: 'Cortes sucessivos que reduzem a impureza de Gini.',
  datasets: ['moons', 'xor', 'circles', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 3,
  thumbnailSteps: 500,
  params: [
    { key: 'depth', label: 'Profundidade máxima', type: 'range', min: 1, max: 12, step: 1, value: 6 },
    { key: 'minSplit', label: 'Mínimo de amostras para dividir', type: 'range', min: 2, max: 30, step: 1, value: 4 },
  ],
  create: (data, params) => createTree(data, params as unknown as TreeParams),
  Visual: TreeVisual,
  aux: { title: 'Estrutura da árvore (cor = proporção da classe B no nó)', Component: TreeAux },
  metrics: (s, data) => [
    { label: 'Nós', value: countNodes(s.root) },
    { label: 'Folhas', value: s.leaves },
    { label: 'Nós pendentes', value: s.queue.length },
    { label: 'Profundidade', value: treeDepth(s.root) },
    { label: 'Acurácia (treino)', value: pct(accuracy(data, (x, y) => predictTree(s.root, x, y))) },
  ],
  docs: {
    description:
      'A árvore de decisão divide o plano recursivamente com cortes paralelos aos eixos. Em cada nó, testa todos os cortes possíveis em cada atributo e escolhe o que mais reduz a impureza de Gini. A animação constrói a árvore nível por nível: cada passo decide um nó (o retângulo amarelo), dividindo ou transformando em folha. Os retângulos tracejados ainda aguardam decisão. Árvores profundas memorizam os dados (repare na acurácia de treino chegando a 100%), árvores rasas generalizam mais. O diagrama abaixo mostra a estrutura, com o teste de cada nó.',
    steps: [
      'Comece com todos os dados na raiz.',
      'Para cada atributo, ordene os valores e teste cada ponto médio como limiar.',
      'Escolha o corte com menor Gini ponderado, onde G = 2p(1 - p).',
      'Crie dois filhos e repita até o nó ficar puro, pequeno demais ou atingir a profundidade máxima.',
    ],
    complexity: [['Treino', 'O(d · n log n) por nível'], ['Predição', 'O(h)'], ['Espaço', 'O(número de nós)']],
    complexityNote: 'h é a profundidade da árvore. No pior caso, h chega a n.',
    pseudocode: [
      'fila ← [raiz com todos os dados]',
      'enquanto fila não vazia: nó ← fila.remover()',
      '    se nó é puro, profundo ou pequeno demais:',
      '        nó vira folha ; continuar',
      '    para cada atributo f e limiar t:',
      '        g ← gini ponderado do corte (f < t)',
      '    (f*, t*) ← corte de menor g',
      '    dividir nó em esquerda (f* < t*) e direita',
      '    fila.adicionar(esquerda, direita)',
    ],
    exercises: [
      'Com profundidade 12 e mínimo 2, a acurácia de treino chega a 100%. Adicione alguns pontos "errados" com o clique e veja a árvore criar regiões só para eles.',
      'Troque o Gini pela entropia (-p log p - (1-p) log(1-p)) em shared/tree.ts. As árvores mudam muito?',
      'Implemente poda por custo-complexidade (cost-complexity pruning).',
      'Por que árvores têm dificuldade com fronteiras diagonais, como no conjunto linearmente separável?',
    ],
  },
};
