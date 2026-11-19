import { motion } from 'framer-motion';
import { binaryColor, classColor, COLORS } from '../../core/colors';
import { accuracy, fmt, pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points, Ring, SPRING } from '../../components/plot/marks';
import { createNaiveBayes, posterior, variance, type ClassStats, type NaiveBayesParams, type NaiveBayesState } from './model';

/**
 * Elipses de 1σ e 2σ de cada classe. Como os atributos são independentes
 * (hipótese ingênua), as elipses ficam sempre alinhadas aos eixos.
 * O Framer Motion anima centro e raios conforme as estatísticas mudam.
 */
function ClassEllipses({ st, c, smoothing }: { st: ClassStats; c: number; smoothing: number }) {
  const { x, y, size } = usePlot();
  if (st.n < 2) return null;
  const sx = Math.sqrt(variance(st, 0, smoothing)), sy = Math.sqrt(variance(st, 1, smoothing));
  const k = size / 2;
  return (
    <g pointerEvents="none">
      {[1, 2].map((m) => (
        <motion.ellipse
          key={m}
          initial={false}
          animate={{ cx: x(st.mean[0]), cy: y(st.mean[1]), rx: m * sx * k, ry: m * sy * k }}
          transition={SPRING}
          fill="none"
          stroke={classColor(c)}
          strokeWidth={m === 1 ? 2.2 : 1.2}
          strokeDasharray={m === 2 ? '5 5' : undefined}
        />
      ))}
      <motion.circle initial={false} animate={{ cx: x(st.mean[0]), cy: y(st.mean[1]) }} transition={SPRING} r={6} fill={classColor(c)} stroke="#fff" strokeWidth={2} />
    </g>
  );
}

function NaiveBayesVisual({ state: s, data, version, compact }: VisualProps<NaiveBayesState>) {
  const cur = s.current >= 0 ? data[s.current] : null;
  return (
    <Plot interactive={false} heatmap={{ version, color: (x, y) => binaryColor(posterior(s, x, y)) }}>
      <Points data={data} color={(p) => classColor(p.c)} opacity={(_, i) => (s.current === -1 || i === s.current ? 1 : 0.85)} />
      <ClassEllipses st={s.stats[0]} c={0} smoothing={s.smoothing} />
      <ClassEllipses st={s.stats[1]} c={1} smoothing={s.smoothing} />
      {!compact && <Ring at={cur} color={COLORS.accent} width={2.5} />}
    </Plot>
  );
}

export const naiveBayes: AlgorithmDef<NaiveBayesState> = {
  id: 'naive-bayes',
  name: 'Naive Bayes gaussiano',
  group: 'Modelos lineares e probabilísticos',
  tag: 'Classificação bayesiana',
  kind: 'classification',
  summary: 'Uma gaussiana por classe e o teorema de Bayes.',
  datasets: ['blobs', 'circles', 'linear', 'moons'],
  defaultDataset: 'blobs',
  defaultSpeed: 4,
  thumbnailSteps: 1000,
  params: [{ key: 'smoothing', label: 'Suavização da variância', type: 'select', options: [0, 0.001, 0.01, 0.05], value: 0.001, help: 'Soma um valor fixo às variâncias, evitando divisões por quase zero.' }],
  create: (data, params) => createNaiveBayes(data, params as unknown as NaiveBayesParams),
  Visual: NaiveBayesVisual,
  metrics: (s, data) => [
    { label: 'Pontos vistos', value: `${s.seen} / ${data.length}` },
    { label: 'P(A) / P(B)', value: `${fmt(s.stats[0].n / Math.max(1, s.seen), 2)} / ${fmt(s.stats[1].n / Math.max(1, s.seen), 2)}` },
    { label: 'μ A', value: `(${fmt(s.stats[0].mean[0], 2)}, ${fmt(s.stats[0].mean[1], 2)})` },
    { label: 'μ B', value: `(${fmt(s.stats[1].mean[0], 2)}, ${fmt(s.stats[1].mean[1], 2)})` },
    { label: 'Acurácia', value: pct(accuracy(data, (x, y) => posterior(s, x, y))) },
  ],
  docs: {
    description:
      'O Naive Bayes aplica o teorema de Bayes supondo que os atributos são independentes dentro de cada classe. Na versão gaussiana, cada classe é descrita por uma média e uma variância por atributo, desenhadas como elipses de 1σ e 2σ alinhadas aos eixos. A animação estima essas estatísticas um ponto por vez com o algoritmo de Welford, e o fundo mostra a probabilidade a posteriori da classe B. Como as variâncias das classes podem diferir, a fronteira é quadrática: repare que nos círculos ela vira uma curva fechada.',
    steps: [
      'Para cada classe, estime a proporção P(k), a média μₖ e a variância σ²ₖ de cada atributo.',
      'Para um ponto novo, calcule P(k) · Πⱼ N(xⱼ; μₖⱼ, σ²ₖⱼ) para cada classe.',
      'Normalize os valores para que somem 1: isso é P(k | x).',
      'Preveja a classe com maior probabilidade a posteriori.',
    ],
    complexity: [['Treino', 'O(n·d)'], ['Predição', 'O(k·d)'], ['Espaço', 'O(k·d)']],
    complexityNote: 'k é o número de classes. Um dos modelos mais baratos que existem, e treina em uma única passada.',
    pseudocode: [
      'para cada classe k: nₖ ← 0 ; μₖ ← 0 ; Mₖ ← 0',
      'para cada (x, y):',
      '    nᵧ ← nᵧ + 1 ;  δ ← x - μᵧ',
      '    μᵧ ← μᵧ + δ / nᵧ',
      '    Mᵧ ← Mᵧ + δ · (x - μᵧ)',
      'σ²ₖ ← Mₖ / nₖ ;  P(k) ← nₖ / n',
      'prever(x) = argmaxₖ  P(k) · Πⱼ N(xⱼ; μₖⱼ, σ²ₖⱼ)',
    ],
    exercises: [
      'Nos círculos, por que o Naive Bayes funciona razoavelmente apesar de ser um modelo tão simples?',
      'Crie um conjunto em que os dois atributos são fortemente correlacionados e veja a hipótese ingênua falhar.',
      'Troque as elipses alinhadas por elipses com covariância completa (isso vira a Análise Discriminante Quadrática).',
      'Calcule tudo em escala logarítmica (log-verossimilhança) para evitar underflow com muitos atributos.',
    ],
  },
};
