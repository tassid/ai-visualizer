import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Label, Points, Ring } from '../../components/plot/marks';
import { createKnn, neighbors, type KnnParams, type KnnState, type Neighborhood } from './model';

/** Células já classificadas, desenhadas como retângulos SVG. */
function Cells({ s }: { s: KnnState }) {
  const { size } = usePlot();
  const R = s.res, cs = size / R;
  const rects = [];
  for (let c = 0; c < s.done; c++) {
    const [r, g, b, a] = binaryColor(s.cells[c]);
    rects.push(<rect key={c} x={(c % R) * cs} y={Math.floor(c / R) * cs} width={cs + 0.6} height={cs + 0.6} fill={`rgba(${r | 0},${g | 0},${b | 0},${a + 0.08})`} />);
  }
  return <g pointerEvents="none">{rects}</g>;
}

/** Linhas até os vizinhos, círculo do raio e o marcador da consulta. */
function Query({ q, data, showLabel }: { q: Neighborhood; data: VisualProps<KnnState>['data']; showLabel: boolean }) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {q.idx.map((i) => (
        <motion.line key={i} initial={false} animate={{ x1: x(q.x), y1: y(q.y), x2: x(data[i].x), y2: y(data[i].y) }} transition={{ duration: 0.08 }} stroke="rgba(255,255,255,.55)" strokeWidth={1} />
      ))}
      <Ring at={q} worldRadius={q.radius} color="rgba(255,209,102,.8)" width={1.4} dash="5 5" />
      <motion.rect initial={false} animate={{ x: x(q.x) - 6, y: y(q.y) - 6 }} transition={{ duration: 0.08 }} width={12} height={12} fill={q.p > 0.5 ? COLORS.classB : q.p < 0.5 ? COLORS.classA : '#fff'} stroke="#fff" strokeWidth={2} />
      {showLabel && <Label at={q}>{Math.round(q.p * q.idx.length)} de {q.idx.length} votos na classe B</Label>}
    </g>
  );
}

function KnnVisual({ state: s, data, params, hover, done, compact }: VisualProps<KnnState>) {
  const hovering = hover && data.length ? neighbors(data, hover.x, hover.y, Number(params.k), params.weighting as KnnParams['weighting']) : null;
  const q = compact ? null : hovering ?? s.current;
  const near = new Set(q?.idx ?? []);
  return (
    <Plot interactive={false}>
      <Cells s={s} />
      <Points data={data} color={(p) => classColor(p.c)} radius={(_, i) => (near.has(i) ? 6.5 : 4.4)} />
      {q && <Query q={q} data={data} showLabel={!!hovering || done} />}
    </Plot>
  );
}

export const knn: AlgorithmDef<KnnState> = {
  id: 'knn',
  name: 'K vizinhos mais próximos (KNN)',
  group: 'Instâncias e árvores',
  tag: 'Classificação por vizinhança',
  kind: 'classification',
  summary: 'Vota entre os k exemplos mais próximos.',
  datasets: ['moons', 'circles', 'xor', 'spiral', 'blobs'],
  defaultDataset: 'moons',
  defaultSpeed: 8,
  thumbnailSteps: 2000,
  hint: 'Quando a varredura terminar (ou com o treino pausado), passe o mouse pelo gráfico para ver os vizinhos de qualquer ponto.',
  params: [
    { key: 'k', label: 'k (vizinhos)', type: 'range', min: 1, max: 31, step: 2, value: 7, help: 'k ímpar evita empates em duas classes.' },
    { key: 'weighting', label: 'Peso do voto', type: 'select', options: ['uniforme', 'distância'], value: 'uniforme' },
    { key: 'res', label: 'Resolução da varredura', type: 'select', options: [20, 30, 40], value: 30 },
  ],
  create: (data, params) => createKnn(data, params as unknown as KnnParams),
  Visual: KnnVisual,
  metrics: (s, data, params) => [
    { label: 'Células classificadas', value: `${s.done} / ${s.res * s.res}` },
    { label: 'k', value: params.k },
    { label: 'Acurácia (leave-one-out)', value: pct(s.looAccuracy) },
    { label: 'Pontos guardados', value: data.length },
  ],
  docs: {
    description:
      'O KNN não tem fase de treino: ele simplesmente guarda os dados. Para classificar um ponto novo, mede a distância até todos os exemplos, pega os k mais próximos e faz uma votação. A animação varre o plano célula por célula, mostrando os vizinhos usados em cada decisão e o raio que os contém. Valores pequenos de k geram fronteiras irregulares que decoram o ruído (sobreajuste); valores grandes suavizam demais (subajuste). A acurácia leave-one-out classifica cada ponto usando todos os outros, uma forma honesta de medir o erro.',
    steps: [
      'Guarde todos os exemplos de treino.',
      'Para um ponto de consulta, calcule a distância até cada exemplo.',
      'Ordene e selecione os k mais próximos.',
      'A classe prevista é a mais votada entre esses vizinhos (opcionalmente ponderada por 1/distância).',
    ],
    complexity: [['Treino', 'O(1)'], ['Predição (força bruta)', 'O(n·d)'], ['Predição (KD tree)', '≈ O(log n) em dimensão baixa'], ['Espaço', 'O(n·d)']],
    complexityNote: 'Em dimensões altas as distâncias ficam parecidas entre si (maldição da dimensionalidade) e estruturas como KD tree perdem eficiência.',
    pseudocode: [
      'função classificar(q):',
      '    D ← []',
      '    para cada (x, y) nos dados:',
      '        D.adicionar( (distância(q, x), y) )',
      '    ordenar D pela distância ; V ← primeiros k',
      '    retornar classe mais votada em V',
    ],
    exercises: [
      'Varie k de 1 a 31 nas luas e anote a acurácia leave-one-out. Qual k é o melhor?',
      'Troque a distância euclidiana pela de Manhattan em model.ts. A fronteira muda de forma?',
      'Implemente uma KD tree e compare o tempo de predição com a força bruta para 10.000 pontos.',
      'Use o KNN para regressão: a previsão vira a média dos y dos vizinhos.',
    ],
  },
};
