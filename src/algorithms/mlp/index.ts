import { accuracy, fmt, pct } from '../../core/math';
import type { AlgorithmDef } from '../../core/types';
import { createMlp, type MlpParams, type MlpState } from './model';
import { MlpNetwork, MlpVisual } from './Visual';

export const mlp: AlgorithmDef<MlpState> = {
  id: 'mlp',
  name: 'Perceptron multicamadas (MLP)',
  group: 'Redes neurais',
  tag: 'Rede neural profunda',
  kind: 'classification',
  summary: 'Camadas ocultas e backpropagation para fronteiras curvas.',
  datasets: ['circles', 'xor', 'moons', 'spiral', 'blobs', 'linear'],
  defaultDataset: 'circles',
  defaultSpeed: 6,
  thumbnailSteps: 250,
  params: [
    { key: 'arch', label: 'Camadas ocultas', type: 'select', options: ['2', '4', '8', '4-4', '8-8', '16-16', '8-8-8'], value: '8-8', help: 'Neurônios por camada oculta, separados por hífen.' },
    { key: 'act', label: 'Ativação', type: 'select', options: ['tanh', 'relu', 'sigmoid'], value: 'tanh' },
    { key: 'opt', label: 'Otimizador', type: 'select', options: ['Adam', 'Momentum', 'SGD'], value: 'Adam' },
    { key: 'lr', label: 'Taxa de aprendizado (η)', type: 'select', options: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], value: 0.03 },
    { key: 'batch', label: 'Tamanho do lote', type: 'select', options: ['completo', 64, 32, 8], value: 'completo', help: 'Lotes menores fazem mais atualizações por época, com gradiente mais ruidoso.' },
    { key: 'epochs', label: 'Épocas', type: 'range', min: 100, max: 4000, step: 100, value: 1500 },
  ],
  create: (data, params) => createMlp(data, params as unknown as MlpParams),
  Visual: MlpVisual,
  aux: { title: 'Rede: cada quadrado mostra a ativação do neurônio em todo o plano', Component: MlpNetwork },
  history: { label: 'Perda (entropia cruzada)', get: (s) => s.history },
  metrics: (s, data) => [
    { label: 'Época', value: s.epoch },
    { label: 'Perda', value: fmt(s.loss, 4) },
    { label: 'Acurácia', value: pct(accuracy(data, (x, y) => s.net.predict(x, y))) },
    { label: 'Arquitetura', value: s.net.sizes.join(' → ') },
    { label: 'Parâmetros', value: s.net.parameterCount },
  ],
  docs: {
    description:
      'O perceptron multicamadas empilha camadas de neurônios com funções de ativação não lineares. Cada camada oculta transforma o espaço de entrada, e a composição dessas transformações permite desenhar fronteiras curvas, resolvendo XOR, círculos e espirais. O treino usa retropropagação (backpropagation): o erro da saída volta pelas camadas pela regra da cadeia, gerando o gradiente de cada peso, que é ajustado pelo otimizador. No painel abaixo do gráfico, cada quadrado mostra como um neurônio responde em todo o plano, e as conexões ficam verdes para pesos positivos e vermelhas para negativos, mais grossas quanto maior o peso.',
    steps: [
      'Propagação direta: cada camada calcula a = g(W·a anterior + b).',
      'A saída passa por uma sigmoide e a perda é a entropia cruzada binária.',
      'Retropropagação: o erro δ da saída é multiplicado pelos pesos transpostos e pela derivada da ativação, camada a camada.',
      'Os gradientes médios do lote ajustam pesos e vieses (SGD, Momentum ou Adam).',
      'Cada passo da animação é uma época completa sobre todos os dados.',
    ],
    complexity: [['Treino', 'O(n·P) por época'], ['Predição', 'O(P)'], ['Espaço', 'O(P)']],
    complexityNote: 'P é o número total de pesos. A retropropagação custa só um fator constante a mais que a propagação direta.',
    pseudocode: [
      'inicialize W(l), b(l) aleatórios',
      'para época = 1 até E:',
      '    para cada lote:',
      '        para cada (x, y) no lote:',
      '            a(0) ← x',
      '            a(l) ← g(W(l)·a(l-1) + b(l))   para l = 1..L',
      '            δ(L) ← a(L) - y',
      '            ∇W(l) += δ(l) · a(l-1)ᵀ',
      '            ∇b(l) += δ(l)',
      '            δ(l-1) ← (W(l)ᵀ·δ(l)) ⊙ g\'(a(l-1))',
      '        W, b ← otimizador(W, b, ∇ / tamanho do lote)',
      '    se perda < ε: pare',
    ],
    exercises: [
      'Com 1 camada de 2 neurônios, a rede resolve os círculos? E o XOR? Observe os mapas de ativação para entender por quê.',
      'Troque tanh por relu e compare o formato das fronteiras (ReLU gera regiões poligonais).',
      'Compare SGD, Momentum e Adam com a mesma taxa no conjunto espiral. Qual chega primeiro a perda baixa?',
      'Adicione regularização L2 (weight decay) em optimizers.ts e veja o efeito em dados com ruído.',
      'Implemente uma camada de saída softmax para mais de duas classes.',
    ],
  },
};
