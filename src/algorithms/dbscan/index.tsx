import { CLUSTER_PALETTE, COLORS } from '../../core/colors';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { motion, AnimatePresence } from 'framer-motion';
import { createDbscan, NOISE, UNSEEN, type DbscanParams, type DbscanState } from './model';

/** Pontos na fila de expansão ganham um anel que surge e some com animação. */
function QueueRings({ s, data }: { s: DbscanState; data: VisualProps<DbscanState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <AnimatePresence>
      {s.queue.map((j) => (
        <motion.circle key={j} cx={x(data[j].x)} cy={y(data[j].y)} initial={{ r: 2, opacity: 0 }} animate={{ r: 7.5, opacity: 0.8 }} exit={{ r: 12, opacity: 0 }} fill="none" stroke={COLORS.accent} strokeWidth={1.2} />
      ))}
    </AnimatePresence>
  );
}

function DbscanVisual({ state: s, data, params, compact }: VisualProps<DbscanState>) {
  const cur = s.current >= 0 ? data[s.current] : null;
  const color = (i: number) => (s.label[i] === UNSEEN ? COLORS.neutral : s.label[i] === NOISE ? COLORS.noise : CLUSTER_PALETTE[s.label[i] % CLUSTER_PALETTE.length]);
  return (
    <Plot interactive={false}>
      {!compact && <QueueRings s={s} data={data} />}
      <Points data={data} color={(_, i) => color(i)} radius={(_, i) => (s.core[i] ? 5.4 : s.label[i] === NOISE ? 3.4 : 3.9)} />
      {!compact && cur && (
        <>
          <Ring at={cur} worldRadius={Number(params.eps)} color="rgba(255,255,255,.8)" width={1.4} dash="5 5" />
          <Ring at={cur} r={8} color={COLORS.accent} width={2.5} />
        </>
      )}
    </Plot>
  );
}

export const dbscan: AlgorithmDef<DbscanState> = {
  id: 'dbscan',
  name: 'DBSCAN',
  group: 'Não supervisionado',
  tag: 'Agrupamento por densidade',
  kind: 'clustering',
  summary: 'Grupos crescem a partir de regiões densas; o resto é ruído.',
  datasets: ['moons', 'circles', 'clusters', 'spiral'],
  defaultDataset: 'moons',
  defaultSpeed: 6,
  thumbnailSteps: 2000,
  params: [
    { key: 'eps', label: 'Raio ε', type: 'range', min: 0.03, max: 0.3, step: 0.01, value: 0.15 },
    { key: 'minPts', label: 'minPts', type: 'range', min: 2, max: 12, step: 1, value: 4, help: 'Vizinhos mínimos (contando o próprio ponto) para ser núcleo.' },
  ],
  create: (data, params) => createDbscan(data, params as unknown as DbscanParams),
  Visual: DbscanVisual,
  metrics: (s, data) => {
    let noise = 0, core = 0, visited = 0;
    s.label.forEach((l, i) => { if (l === NOISE) noise++; if (s.core[i]) core++; if (s.visited[i]) visited++; });
    return [
      { label: 'Grupos', value: s.clusters },
      { label: 'Pontos de núcleo', value: core },
      { label: 'Ruído', value: noise },
      { label: 'Fila de expansão', value: s.queue.length },
      { label: 'Visitados', value: `${visited} / ${data.length}` },
    ];
  },
  docs: {
    description:
      'O DBSCAN agrupa pontos por densidade. Um ponto é de núcleo (desenhado maior) se tiver pelo menos minPts vizinhos dentro do raio ε. Os grupos crescem conectando núcleos vizinhos, como uma busca em largura: a fila de expansão aparece com anéis amarelos. Pontos que não pertencem a nenhuma região densa ficam como ruído, em cinza escuro. Ao contrário do K-Means, não é preciso informar o número de grupos e ele encontra formas arbitrárias, como as luas. O círculo tracejado é o raio ε do ponto em análise.',
    steps: [
      'Pegue um ponto ainda não visitado e conte seus vizinhos no raio ε.',
      'Se forem menos que minPts, marque como ruído (ele ainda pode virar borda depois).',
      'Se forem pelo menos minPts, abra um novo grupo e coloque os vizinhos na fila.',
      'Expanda a fila: vizinhos que também são núcleo trazem os vizinhos deles.',
      'Repita até visitar todos os pontos.',
    ],
    complexity: [['Treino (força bruta)', 'O(n²)'], ['Treino (índice espacial)', 'O(n log n)'], ['Predição', 'não se aplica'], ['Espaço', 'O(n)']],
    pseudocode: [
      'C ← 0',
      'para cada ponto p não visitado:',
      '    marcar p como visitado',
      '    N ← vizinhos(p, ε)',
      '    se |N| < minPts: p ← ruído ; continuar',
      '',
      '    C ← C + 1 ;  p ← grupo C ;  fila ← N',
      '    para cada q na fila:',
      '        se q não visitado: marcar q',
      '            N\' ← vizinhos(q, ε)',
      '            se |N\'| ≥ minPts: fila ← fila ∪ N\'',
      '        se q sem grupo: q ← grupo C',
    ],
    exercises: [
      'Nas luas, encontre a faixa de ε em que surgem exatamente 2 grupos. O que acontece abaixo e acima dela?',
      'Implemente uma grade espacial (células de lado ε) em region() e meça o ganho de tempo.',
      'Desenhe o gráfico k-distância (distância ao k-ésimo vizinho, ordenada) para escolher ε.',
      'Pesquise o HDBSCAN e explique como ele elimina a necessidade de escolher ε.',
    ],
  },
};
