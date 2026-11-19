import { motion } from 'framer-motion';
import { line } from 'd3-shape';
import { CLUSTER_PALETTE, COLORS, hexToRgb } from '../../core/colors';
import { fmt } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring } from '../../components/plot/marks';
import { createKMeans, nearest, type KMeansParams, type KMeansState } from './model';

const PAL_RGB = CLUSTER_PALETTE.map(hexToRgb);

/**
 * Centroides como losangos que deslizam com mola até a nova média.
 * O rastro pontilhado mostra todas as posições anteriores.
 */
function Centroids({ s }: { s: KMeansState }) {
  const { x, y } = usePlot();
  const path = line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1]));
  return (
    <g pointerEvents="none">
      {s.trails.map((t, k) => (
        <path key={`t${k}`} d={path(t) ?? ''} fill="none" stroke="rgba(255,255,255,.55)" strokeWidth={1.5} strokeDasharray="3 4" />
      ))}
      {s.centroids.map((c, k) => (
        <motion.g key={k} initial={false} animate={{ x: x(c.x), y: y(c.y) }} transition={{ type: 'spring', stiffness: 120, damping: 16 }}>
          <rect x={-8} y={-8} width={16} height={16} transform="rotate(45)" fill={CLUSTER_PALETTE[k % CLUSTER_PALETTE.length]} stroke="#fff" strokeWidth={2.5} />
        </motion.g>
      ))}
    </g>
  );
}

function KMeansVisual({ state: s, data, version, compact }: VisualProps<KMeansState>) {
  const started = s.iter > 0 || s.assign.some((a) => a >= 0);
  return (
    <Plot
      interactive={false}
      heatmap={started ? { version, resolution: 90, color: (x, y) => [...PAL_RGB[nearest(s.centroids, x, y)[0] % PAL_RGB.length], 0.16] } : null}
    >
      <Points data={data} color={(_, i) => (s.assign[i] >= 0 ? CLUSTER_PALETTE[s.assign[i] % CLUSTER_PALETTE.length] : COLORS.neutral)} />
      {!compact && s.current >= 0 && <Ring at={data[s.current]} r={9} color="#fff" />}
      <Centroids s={s} />
    </Plot>
  );
}

export const kmeans: AlgorithmDef<KMeansState> = {
  id: 'kmeans',
  name: 'K-Means',
  group: 'Não supervisionado',
  tag: 'Agrupamento',
  kind: 'clustering',
  summary: 'Alterna atribuição e média até os centroides pararem.',
  datasets: ['clusters', 'blobs', 'moons', 'circles'],
  defaultDataset: 'clusters',
  defaultSpeed: 1,
  thumbnailSteps: 200,
  params: [
    { key: 'k', label: 'k (grupos)', type: 'range', min: 2, max: 8, step: 1, value: 4 },
    { key: 'init', label: 'Inicialização', type: 'select', options: ['k-means++', 'aleatória'], value: 'k-means++' },
    { key: 'animateAssign', label: 'Animar atribuição ponto a ponto', type: 'select', options: ['não', 'sim'], value: 'não' },
  ],
  create: (data, params) => createKMeans(data, params as unknown as KMeansParams),
  Visual: KMeansVisual,
  history: { label: 'Inércia', get: (s) => s.history },
  metrics: (s) => [
    { label: 'Iteração', value: s.iter },
    { label: 'Fase atual', value: s.phase },
    { label: 'Pontos que mudaram', value: s.changed },
    { label: 'Inércia', value: fmt(s.inertia, 3) },
  ],
  docs: {
    description:
      'O K-Means divide os dados em k grupos alternando dois passos: atribuir cada ponto ao centroide mais próximo e mover cada centroide para a média dos seus pontos. O fundo colorido são as regiões de Voronoi (a área mais próxima de cada centroide). Nenhum dos dois passos aumenta a inércia, por isso o algoritmo sempre converge, embora possa parar num mínimo local. A inicialização k-means++ espalha os centroides iniciais e costuma dar resultados melhores. Teste as luas e os círculos para ver onde ele falha: o K-Means assume grupos arredondados e de tamanho parecido.',
    steps: [
      'Escolha k centroides iniciais (aleatórios ou k-means++).',
      'Atribua cada ponto ao centroide mais próximo (regiões de Voronoi).',
      'Mova cada centroide para a média dos pontos atribuídos a ele.',
      'Repita até nenhum ponto mudar de grupo.',
    ],
    complexity: [['Treino', 'O(n·k·d) por iteração'], ['Predição', 'O(k·d)'], ['Espaço', 'O(n + k·d)']],
    pseudocode: [
      'escolher k centroides c₁ ... cₖ',
      'repita:',
      '    para cada ponto x:',
      '        grupo(x) ← argminⱼ |x - cⱼ|²',
      '    para cada j:',
      '        cⱼ ← média dos x com grupo(x) = j',
      'até nenhum grupo mudar',
    ],
    exercises: [
      'Rode várias vezes com inicialização aleatória e k = 4. Quantas vezes ele cai num mínimo local ruim?',
      'Implemente o método do cotovelo: rode para k = 1..8 e plote a inércia final.',
      'Calcule o coeficiente de silhueta para escolher k automaticamente.',
      'Implemente o K-Medoids, que usa pontos reais como centros e é mais robusto a outliers.',
    ],
  },
};
