import { binaryColor, classColor } from '../../core/colors';
import { accuracy, pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { Points } from '../../components/plot/marks';
import { countNodes } from '../shared/tree';
import { SplitLines } from '../decision-tree';
import { createForest, forestPredict, type ForestParams, type ForestState } from './model';

function ForestVisual({ state: s, data, version, compact }: VisualProps<ForestState>) {
  const last = s.trees[s.trees.length - 1];
  const bag = s.inBag[s.inBag.length - 1];
  return (
    <Plot interactive={false} heatmap={s.trees.length ? { version, resolution: 96, color: (x, y) => binaryColor(forestPredict(s, x, y)) } : null}>
      {last && !compact && <SplitLines key={s.trees.length} root={last} color="rgba(255,209,102,.55)" width={1.2} />}
      <Points data={data} color={(p) => classColor(p.c)} radius={(_, i) => (!bag || bag[i] ? 4.6 : 2.6)} />
    </Plot>
  );
}

export const randomForest: AlgorithmDef<ForestState> = {
  id: 'random-forest',
  name: 'Floresta aleatória',
  group: 'Instâncias e árvores',
  tag: 'Ensemble (bagging)',
  kind: 'classification',
  summary: 'Média de muitas árvores treinadas em amostras diferentes.',
  datasets: ['moons', 'circles', 'xor', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 3,
  thumbnailSteps: 20,
  params: [
    { key: 'trees', label: 'Número de árvores', type: 'range', min: 1, max: 80, step: 1, value: 40 },
    { key: 'depth', label: 'Profundidade máxima', type: 'range', min: 1, max: 12, step: 1, value: 7 },
    { key: 'maxFeatures', label: 'Atributos por corte', type: 'select', options: ['1 aleatório', 'todos'], value: '1 aleatório', help: 'Com "todos", só o bootstrap diferencia as árvores.' },
  ],
  create: (data, params) => createForest(data, params as unknown as ForestParams),
  Visual: ForestVisual,
  history: { label: 'Erro out of bag', get: (s) => s.history },
  metrics: (s, data) => [
    { label: 'Árvores', value: s.trees.length },
    { label: 'Acurácia (treino)', value: pct(accuracy(data, (x, y) => forestPredict(s, x, y))) },
    { label: 'Acurácia out of bag', value: pct(s.oobAccuracy) },
    { label: 'Nós na última árvore', value: s.trees.length ? countNodes(s.trees[s.trees.length - 1]) : 0 },
  ],
  docs: {
    description:
      'A floresta aleatória combina muitas árvores de decisão. Cada árvore é treinada em uma amostra bootstrap (sorteio com reposição) e, a cada corte, só pode usar um atributo sorteado. As árvores erram de formas diferentes, e a média das previsões reduz a variância, gerando fronteiras mais suaves e estáveis que as de uma árvore só. As linhas amarelas mostram os cortes da árvore mais recente, e os pontos menores ficaram fora da amostra dela (out of bag). O erro out of bag, no gráfico ao lado, estima o erro de teste sem separar dados.',
    steps: [
      'Sorteie n exemplos com reposição (bootstrap).',
      'Cresça uma árvore nessa amostra, sorteando quais atributos cada corte pode usar.',
      'Repita para T árvores.',
      'A previsão é a média (ou votação) de todas as árvores.',
    ],
    complexity: [['Treino', 'O(T · d · n log n · h)'], ['Predição', 'O(T · h)'], ['Espaço', 'O(T · nós)']],
    complexityNote: 'T é o número de árvores e h a profundidade. As árvores são independentes, então o treino paraleliza perfeitamente.',
    pseudocode: [
      'floresta ← []',
      'para t = 1 até T:',
      '    amostra ← sortear n exemplos com reposição',
      '    árvore ← construir(amostra, atributos aleatórios por corte)',
      '    floresta.adicionar(árvore)',
      '',
      'prever(x) = média de árvore(x) para cada árvore na floresta',
    ],
    exercises: [
      'Compare uma floresta de 1 árvore com uma de 80: o que acontece com a acurácia out of bag?',
      'Troque "1 aleatório" por "todos". As árvores ficam mais parecidas entre si? Como isso afeta o erro?',
      'Implemente a importância de atributos por diminuição média de impureza.',
      'Implemente o gradient boosting com árvores rasas (profundidade 1 a 3) e compare com a floresta.',
    ],
  },
};
