import { useCallback } from 'react';
import { COLORS } from '../../core/colors';
import { fmt } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { LossLandscape } from '../../components/charts/LossLandscape';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Points } from '../../components/plot/marks';
import { motion } from 'framer-motion';
import { line } from 'd3-shape';
import { linspace } from '../../core/math';
import { createLinReg, evaluate, type LinRegParams, type LinRegState } from './model';

const XS = linspace(-1, 1, 160);

function Curve({ s }: { s: LinRegState }) {
  const { x, y } = usePlot();
  const d = line<number>().x((v) => x(v)).y((v) => y(evaluate(s, v)))(XS) ?? '';
  return <motion.path initial={false} animate={{ d }} transition={{ duration: 0.1 }} fill="none" stroke={COLORS.accent} strokeWidth={2.8} />;
}

function Residuals({ s, data }: { s: LinRegState; data: VisualProps<LinRegState>['data'] }) {
  const { x, y } = usePlot();
  return (
    <g>
      {data.map((p, i) => (
        <line key={i} x1={x(p.x)} x2={x(p.x)} y1={y(p.y)} y2={y(evaluate(s, p.x))} stroke={COLORS.classA} strokeOpacity={0.55} strokeWidth={1.2} />
      ))}
    </g>
  );
}

function LinRegVisual({ state: s, data }: VisualProps<LinRegState>) {
  return (
    <Plot interactive={false}>
      <Residuals s={s} data={data} />
      <Curve s={s} />
      <Points data={data} color={() => COLORS.classB} radius={4.4} />
    </Plot>
  );
}

function LinRegLandscape({ state: s, data, params }: VisualProps<LinRegState>) {
  const loss = useCallback(
    (w: number, b: number) => {
      let sum = 0;
      for (const p of data) sum += (w * p.x + b - p.y) ** 2;
      return sum / Math.max(1, data.length);
    },
    [data],
  );
  if (Number(params.deg) !== 1) {
    return <div className="aux-empty">A paisagem só é desenhada com grau 1, quando há exatamente dois parâmetros (w e b).</div>;
  }
  return <LossLandscape loss={loss} dataKey={data} path={s.path} width={560} height={250} />;
}

export const linearRegression: AlgorithmDef<LinRegState> = {
  id: 'regressao-linear',
  name: 'Regressão linear',
  group: 'Modelos lineares e probabilísticos',
  tag: 'Regressão',
  kind: 'regression',
  summary: 'Descida de gradiente sobre o erro quadrático médio.',
  datasets: ['line', 'sine'],
  defaultDataset: 'line',
  defaultSpeed: 4,
  thumbnailSteps: 300,
  params: [
    { key: 'deg', label: 'Grau do polinômio', type: 'range', min: 1, max: 7, step: 1, value: 1, help: 'Grau 1 = reta. Graus maiores criam curvas.' },
    { key: 'lr', label: 'Taxa de aprendizado (η)', type: 'select', options: [0.01, 0.05, 0.1, 0.3, 0.6, 1.2], value: 0.1 },
    { key: 'iters', label: 'Iterações', type: 'range', min: 50, max: 3000, step: 50, value: 300 },
  ],
  create: (data, params) => createLinReg(data, params as unknown as LinRegParams),
  Visual: LinRegVisual,
  aux: { title: 'Paisagem da perda em função de w e b, com o caminho do gradiente', Component: LinRegLandscape },
  history: { label: 'Erro quadrático médio', get: (s) => s.history },
  metrics: (s) => [
    { label: 'Iteração', value: s.it },
    { label: 'MSE', value: fmt(s.loss, 5) },
    { label: 'b', value: fmt(s.b, 3) },
    ...s.w.slice(0, 4).map((w, k) => ({ label: `w${k + 1}`, value: fmt(w, 3) })),
  ],
  docs: {
    description:
      'A regressão linear ajusta a reta (ou o polinômio, ao aumentar o grau) que minimiza o erro quadrático médio entre previsões e valores reais. Os segmentos vermelhos são os resíduos: a perda é a média dos seus quadrados. Aqui o ajuste é feito por descida de gradiente: a cada iteração os coeficientes andam um pouco na direção que mais reduz o erro. Com grau 1, o painel inferior mostra a paisagem da perda em função de w e b, com curvas de nível, e o caminho que o gradiente percorre até o fundo do vale.',
    steps: [
      'Comece com coeficientes arbitrários.',
      'Calcule as previsões ŷ = b + w₁x + w₂x² + ... e os resíduos ŷ - y.',
      'Calcule o gradiente do erro quadrático médio em relação a cada coeficiente.',
      'Dê um passo contra o gradiente, proporcional à taxa de aprendizado.',
      'Repita até o erro parar de cair.',
    ],
    complexity: [['Treino (gradiente)', 'O(n·d) por iteração'], ['Treino (equação normal)', 'O(n·d² + d³)'], ['Predição', 'O(d)'], ['Espaço', 'O(d)']],
    pseudocode: [
      'w ← valores iniciais ; b ← 0',
      'repita T vezes:',
      '    para cada (x, y):  e ← ŷ(x) - y',
      '    ∂wₖ ← (2/n) · Σ e · xᵏ',
      '    ∂b ← (2/n) · Σ e',
      '    wₖ ← wₖ - η · ∂wₖ',
      '    b ← b - η · ∂b',
      '    se |∇| ≈ 0: pare',
    ],
    exercises: [
      'Com grau 1, use η = 1,2 e observe o caminho na paisagem. Por que ele ziguezagueia ou diverge?',
      'Na curva seno, aumente o grau de 1 até 7. Em que grau o ajuste fica bom? O que acontece com poucos pontos e grau alto?',
      'Compare o resultado final com normalEquation() em model.ts. Escreva um teste para isso.',
      'Implemente descida de gradiente estocástica (um ponto por vez) e compare o caminho na paisagem.',
    ],
  },
};
