import { accuracy, fmt, pct } from '../../core/math';
import type { AlgorithmDef } from '../../core/types';
import { createPerceptron, predict, type PerceptronParams, type PerceptronState } from './model';
import { PerceptronVisual } from './Visual';

export const perceptron: AlgorithmDef<PerceptronState> = {
  id: 'perceptron',
  name: 'Perceptron',
  group: 'Redes neurais',
  tag: 'Classificação linear',
  kind: 'classification',
  summary: 'O neurônio artificial original: corrige a reta a cada erro.',
  datasets: ['linear', 'blobs', 'xor', 'circles'],
  defaultDataset: 'linear',
  defaultSpeed: 4,
  thumbnailSteps: 700,
  params: [
    { key: 'lr', label: 'Taxa de aprendizado (η)', type: 'select', options: [0.01, 0.05, 0.1, 0.5, 1], value: 0.1, help: 'Tamanho do empurrão dado na reta a cada erro.' },
    { key: 'epochs', label: 'Máximo de épocas', type: 'range', min: 5, max: 100, step: 5, value: 40 },
  ],
  create: (data, params) => createPerceptron(data, params as unknown as PerceptronParams),
  Visual: PerceptronVisual,
  history: { label: 'Erros por época', get: (s) => s.history },
  metrics: (s, data) => [
    { label: 'Época', value: s.epoch + 1 },
    { label: 'Amostra', value: `${s.seen} / ${data.length}` },
    { label: 'Erros nesta época', value: s.errors },
    { label: 'Acurácia', value: pct(accuracy(data, (x, y) => predict(s, x, y))) },
    { label: 'w', value: `(${fmt(s.w[0], 2)}, ${fmt(s.w[1], 2)})` },
    { label: 'b', value: fmt(s.b, 2) },
  ],
  docs: {
    description:
      'O perceptron, proposto por Frank Rosenblatt em 1958, é o neurônio artificial mais simples. Ele calcula uma soma ponderada das entradas e decide a classe pelo sinal do resultado. Sempre que erra um exemplo, empurra a reta de decisão na direção certa. A seta amarela é o vetor de pesos w, que é sempre perpendicular à reta. Se os dados forem linearmente separáveis, o teorema de convergência garante que ele encontra uma reta que separa tudo. No XOR ele nunca converge, e foi exatamente essa limitação, apontada por Minsky e Papert em 1969, que motivou as redes de múltiplas camadas.',
    steps: [
      'Inicialize os pesos w e o viés b com valores pequenos.',
      'Para cada exemplo, calcule f = w·x + b e preveja a classe pelo sinal de f.',
      'Se a previsão estiver errada, atualize w ← w + η·y·x e b ← b + η·y.',
      'Repita as épocas até não haver mais erros ou atingir o limite.',
    ],
    complexity: [['Treino', 'O(n·d) por época'], ['Predição', 'O(d)'], ['Espaço', 'O(d)']],
    complexityNote: 'n é o número de exemplos e d o número de atributos (aqui, d = 2).',
    pseudocode: [
      'w ← valores pequenos ; b ← 0',
      'para época = 1 até E:',
      '    erros ← 0',
      '    para cada (x, y) em ordem aleatória:',
      '        f ← w · x + b',
      '        se y · f ≤ 0:',
      '            w ← w + η · y · x',
      '            b ← b + η · y',
      '            erros ← erros + 1',
      '    se erros = 0: pare (convergiu)',
    ],
    exercises: [
      'Rode no XOR e observe o gráfico de erros por época. Por que ele nunca chega a zero?',
      'Compare η = 0,01 e η = 1 nos dados separáveis. A taxa muda o número de épocas até convergir? (Dica: com b = 0 inicial, escalar w não muda a reta.)',
      'Implemente o perceptron com margem: atualize também quando 0 < y·f < γ.',
      'Implemente o "pocket perceptron", que guarda os melhores pesos já vistos, e teste nas duas nuvens com sobreposição.',
    ],
  },
};
