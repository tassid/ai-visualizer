import { binaryColor, classColor } from '../../core/colors';
import { accuracy, fmt, pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { DecisionLine, Points } from '../../components/plot/marks';
import { createLogistic, probability, type LogisticParams, type LogisticState } from './model';

/** ln(3): onde σ(z) = 0,75. As linhas tracejadas marcam p = 0,25 e p = 0,75. */
const LN3 = Math.log(3);

function LogisticVisual({ state: s, data, version }: VisualProps<LogisticState>) {
  const [w1, w2] = s.w;
  return (
    <Plot interactive={false} heatmap={{ version, color: (x, y) => binaryColor(probability(s, x, y)) }}>
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={LN3} color="rgba(255,255,255,.5)" width={1.4} dash="6 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} offset={-LN3} color="rgba(255,255,255,.5)" width={1.4} dash="6 6" />
      <DecisionLine w1={w1} w2={w2} b={s.b} />
      <Points data={data} color={(p) => classColor(p.c)} />
    </Plot>
  );
}

export const logisticRegression: AlgorithmDef<LogisticState> = {
  id: 'regressao-logistica',
  name: 'Regressão logística',
  group: 'Modelos lineares e probabilísticos',
  tag: 'Classificação probabilística',
  kind: 'classification',
  summary: 'Sigmoide sobre uma reta, treinada por entropia cruzada.',
  datasets: ['blobs', 'linear', 'moons', 'xor'],
  defaultDataset: 'blobs',
  defaultSpeed: 4,
  thumbnailSteps: 400,
  params: [
    { key: 'lr', label: 'Taxa de aprendizado (η)', type: 'select', options: [0.05, 0.1, 0.5, 1, 3], value: 0.5 },
    { key: 'l2', label: 'Regularização L2 (λ)', type: 'select', options: [0, 0.001, 0.01, 0.1], value: 0, help: 'Penaliza pesos grandes; deixa a transição de probabilidade mais larga.' },
    { key: 'iters', label: 'Iterações', type: 'range', min: 50, max: 2000, step: 50, value: 400 },
  ],
  create: (data, params) => createLogistic(data, params as unknown as LogisticParams),
  Visual: LogisticVisual,
  history: { label: 'Perda (entropia cruzada)', get: (s) => s.history },
  metrics: (s, data) => [
    { label: 'Iteração', value: s.it },
    { label: 'Perda', value: fmt(s.loss, 4) },
    { label: '|∇|', value: fmt(s.gradNorm, 5) },
    { label: 'Acurácia', value: pct(accuracy(data, (x, y) => probability(s, x, y))) },
    { label: 'w', value: `(${fmt(s.w[0], 2)}, ${fmt(s.w[1], 2)})` },
    { label: 'b', value: fmt(s.b, 2) },
  ],
  docs: {
    description:
      'Apesar do nome, a regressão logística é um classificador. Ela aplica a função sigmoide a uma combinação linear das entradas, produzindo uma probabilidade entre 0 e 1. Os pesos são ajustados minimizando a entropia cruzada por descida de gradiente. A fronteira de decisão é a reta onde p = 0,5, e as linhas tracejadas marcam p = 0,25 e p = 0,75: quanto mais próximas da fronteira, mais confiante o modelo está. Como a perda é convexa, existe um único mínimo e o gradiente sempre chega lá.',
    steps: [
      'Calcule z = w·x + b para cada ponto.',
      'Converta em probabilidade com p = 1 / (1 + e^(-z)).',
      'O gradiente da entropia cruzada é simplesmente (p - y)·x.',
      'Atualize w e b na direção oposta ao gradiente.',
    ],
    complexity: [['Treino', 'O(n·d) por iteração'], ['Predição', 'O(d)'], ['Espaço', 'O(d)']],
    pseudocode: [
      'w ← aleatório ; b ← 0',
      'repita T vezes:',
      '    para cada (x, y): p ← σ(w·x + b)',
      '    ∂w ← (1/n) Σ (p - y) · x  + λ·w',
      '    ∂b ← (1/n) Σ (p - y)',
      '    w ← w - η · ∂w ;  b ← b - η · ∂b',
      '    se |∇| < ε: pare',
    ],
    exercises: [
      'Nos dados linearmente separáveis, observe |w| crescer sem parar. Depois ligue λ = 0,01 e compare.',
      'Use η = 3 e veja o que acontece com a perda. Em que ponto a descida começa a oscilar?',
      'Derive no papel por que ∂perda/∂z = p - y quando a perda é a entropia cruzada e a saída é sigmoide.',
      'Adicione atributos x², y² e x·y como entradas: a regressão logística passa a resolver os círculos?',
    ],
  },
};
