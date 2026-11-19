import { binaryColor, classColor } from '../../core/colors';
import { accuracy, fmt, pct, sigmoid } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { DecisionLine, Points } from '../../components/plot/marks';
import { motion, AnimatePresence } from 'framer-motion';
import { createSvm, decision, isSupport, margin, type SvmParams, type SvmState } from './model';

function SupportRings({ s, data }: { s: SvmState; data: VisualProps<SvmState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <AnimatePresence>
      {data.map((p, i) =>
        isSupport(s, p) ? (
          <motion.circle key={i} cx={x(p.x)} cy={y(p.y)} initial={{ r: 0, opacity: 0 }} animate={{ r: 8.5, opacity: 0.9 }} exit={{ r: 14, opacity: 0 }} fill="none" stroke="#fff" strokeWidth={1.3} />
        ) : null,
      )}
    </AnimatePresence>
  );
}

function SvmVisual({ state: s, data, version, compact }: VisualProps<SvmState>) {
  const [w1, w2] = s.w;
  return (
    <Plot interactive={false} heatmap={{ version, color: (x, y) => binaryColor(sigmoid(3 * decision(s, x, y))) }}>
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={1} color="rgba(255,255,255,.6)" width={1.5} dash="7 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={-1} color="rgba(255,255,255,.6)" width={1.5} dash="7 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} />
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && <SupportRings s={s} data={data} />}
    </Plot>
  );
}

export const svm: AlgorithmDef<SvmState> = {
  id: 'svm',
  name: 'SVM linear',
  group: 'Modelos lineares e probabilísticos',
  tag: 'Margem máxima',
  kind: 'classification',
  summary: 'A reta com a maior margem possível entre as classes.',
  datasets: ['blobs', 'linear', 'moons'],
  defaultDataset: 'blobs',
  defaultSpeed: 5,
  thumbnailSteps: 800,
  params: [
    { key: 'lambda', label: 'Regularização (λ)', type: 'select', options: [0.001, 0.01, 0.05, 0.1, 0.5], value: 0.01, help: 'λ alto prioriza margem larga; λ baixo prioriza acertar todos os pontos.' },
    { key: 'lr', label: 'Taxa de aprendizado (η)', type: 'select', options: [0.05, 0.1, 0.5, 1], value: 0.5 },
    { key: 'iters', label: 'Iterações', type: 'range', min: 100, max: 3000, step: 100, value: 800 },
  ],
  create: (data, params) => createSvm(data, params as unknown as SvmParams),
  Visual: SvmVisual,
  history: { label: 'Perda hinge + regularização', get: (s) => s.history },
  metrics: (s, data) => [
    { label: 'Iteração', value: s.it },
    { label: 'Perda', value: fmt(s.loss, 4) },
    { label: 'Largura da margem', value: fmt(margin(s), 3) },
    { label: 'Vetores de suporte', value: data.filter((p) => isSupport(s, p)).length },
    { label: 'Acurácia', value: pct(accuracy(data, (x, y) => (decision(s, x, y) >= 0 ? 1 : 0))) },
  ],
  docs: {
    description:
      'A máquina de vetores de suporte (SVM) procura a reta que separa as classes com a maior margem possível. Apenas os pontos sobre a margem ou dentro dela, os vetores de suporte (circulados em branco), influenciam a solução: você pode apagar todos os outros e a reta não muda. Esta é a SVM linear treinada por subgradiente da perda hinge com regularização L2. O λ controla o equilíbrio entre margem larga e acertar todos os pontos. Para fronteiras curvas, SVMs reais usam o truque do kernel, que calcula produtos internos num espaço de dimensão maior sem construí-lo.',
    steps: [
      'A fronteira é w·x + b = 0 e as margens são w·x + b = ±1.',
      'Pontos com y·(w·x + b) ≥ 1 estão fora da margem e não geram perda.',
      'Pontos dentro da margem empurram a reta (perda hinge).',
      'O termo λ·|w|² encolhe w, o que alarga a margem (largura = 2 / |w|).',
    ],
    complexity: [['Treino (subgradiente)', 'O(n·d) por iteração'], ['Treino (dual, kernel)', 'entre O(n²) e O(n³)'], ['Predição (linear)', 'O(d)'], ['Predição (kernel)', 'O(s·d)'], ['Espaço', 'O(d)']],
    complexityNote: 's é o número de vetores de suporte.',
    pseudocode: [
      'para t = 1 até T:',
      '    gw ← λ · w ;  gb ← 0',
      '    para cada (x, y):',
      '        se y · (w·x + b) < 1:',
      '            gw ← gw - y · x / n',
      '            gb ← gb - y / n',
      '    w ← w - ηₜ · gw ;  b ← b - ηₜ · gb',
    ],
    exercises: [
      'Compare λ = 0,001 e λ = 0,5 nas duas nuvens: o que acontece com a margem e com o número de vetores de suporte?',
      'Mude o decaimento da taxa em model.ts para 1/t (Pegasos original) e compare a convergência.',
      'Implemente a SVM com kernel RBF usando o algoritmo SMO simplificado e teste nas luas.',
      'Compare a reta da SVM com a da regressão logística nos mesmos dados. Por que diferem?',
    ],
  },
};
