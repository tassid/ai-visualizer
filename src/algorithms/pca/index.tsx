import { scaleLinear } from 'd3-scale';
import { line } from 'd3-shape';
import { motion } from 'framer-motion';
import { COLORS } from '../../core/colors';
import { fmt, pct } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { Arrow, Points, SPRING } from '../../components/plot/marks';
import { createPca, type PcaParams, type PcaState } from './model';

/** Reta giratória pelo centro dos dados, com as projeções de cada ponto. */
function Projections({ s, data, show }: { s: PcaState; data: VisualProps<PcaState>['data']; show: boolean }) {
  const { x, y } = usePlot();
  const co = Math.cos(s.angle), si = Math.sin(s.angle);
  return (
    <g pointerEvents="none">
      <motion.line initial={false} animate={{ x1: x(s.mx - co * 2), y1: y(s.my - si * 2), x2: x(s.mx + co * 2), y2: y(s.my + si * 2) }} transition={{ duration: 0.08 }} stroke={COLORS.accent} strokeWidth={2} />
      {show &&
        data.map((p, i) => {
          const t = (p.x - s.mx) * co + (p.y - s.my) * si; // coordenada ao longo da reta
          const qx = s.mx + t * co, qy = s.my + t * si;
          return (
            <g key={i}>
              <line x1={x(p.x)} y1={y(p.y)} x2={x(qx)} y2={y(qy)} stroke="rgba(255,209,102,.18)" />
              <circle cx={x(qx)} cy={y(qy)} r={2.4} fill={COLORS.accent} />
            </g>
          );
        })}
    </g>
  );
}

function Components({ s }: { s: PcaState }) {
  const s1 = 2 * Math.sqrt(s.l1), s2 = 2 * Math.sqrt(s.l2);
  const c = Math.cos(s.pc1), n = Math.sin(s.pc1);
  return (
    <motion.g initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={SPRING}>
      <Arrow from={[s.mx, s.my]} to={[s.mx + c * s1, s.my + n * s1]} color={COLORS.accent} width={3} />
      <Arrow from={[s.mx, s.my]} to={[s.mx - n * s2, s.my + c * s2]} color={COLORS.violet} width={3} />
    </motion.g>
  );
}

function PcaVisual({ state: s, data, params, compact }: VisualProps<PcaState>) {
  return (
    <Plot interactive={false}>
      {!s.finished && <Projections s={s} data={data} show={params.projections === 'sim' && !compact} />}
      <Points data={data} color={() => COLORS.blue} radius={4} />
      {s.finished && <Components s={s} />}
      <CenterDot s={s} />
    </Plot>
  );
}

function CenterDot({ s }: { s: PcaState }) {
  const { x, y } = usePlot();
  return <circle cx={x(s.mx)} cy={y(s.my)} r={5} fill="#fff" />;
}

/** Variância projetada em função do ângulo, com D3. O máximo coincide com o PC1. */
function VarianceCurve({ state: s }: VisualProps<PcaState>) {
  const W = 560, H = 200, pad = 32;
  const sx = scaleLinear().domain([0, Math.PI]).range([pad, W - pad]);
  const sy = scaleLinear().domain([0, s.l1 * 1.1 || 1]).range([H - pad, pad / 2]);
  const d = line<[number, number]>().x((p) => sx(p[0])).y((p) => sy(p[1]))(s.curve) ?? '';
  const last = s.curve[s.curve.length - 1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="aux-svg">
      <line x1={pad} x2={W - pad} y1={H - pad} y2={H - pad} stroke="rgba(255,255,255,.15)" />
      <line x1={sx(s.pc1)} x2={sx(s.pc1)} y1={pad / 2} y2={H - pad} stroke="rgba(255,209,102,.6)" strokeDasharray="4 4" />
      <text x={sx(s.pc1) + 6} y={pad} className="node-label" fill={COLORS.accent}>PC1 = {((s.pc1 * 180) / Math.PI).toFixed(1)}° (λ₁ = {s.l1.toFixed(3)})</text>
      <path d={d} fill="none" stroke={COLORS.blue} strokeWidth={2.4} />
      {last && <motion.circle initial={false} animate={{ cx: sx(last[0]), cy: sy(last[1]) }} transition={{ duration: 0.08 }} r={4.5} fill="#fff" />}
      {[0, 45, 90, 135, 180].map((deg) => (
        <text key={deg} x={sx((deg * Math.PI) / 180)} y={H - 10} textAnchor="middle" className="node-label muted">{deg}°</text>
      ))}
    </svg>
  );
}

export const pca: AlgorithmDef<PcaState> = {
  id: 'pca',
  name: 'Análise de componentes principais (PCA)',
  group: 'Não supervisionado',
  tag: 'Redução de dimensionalidade',
  kind: 'reduction',
  summary: 'Gira uma reta procurando a direção de maior variância.',
  datasets: ['elongated', 'clusters', 'blobs'],
  defaultDataset: 'elongated',
  defaultSpeed: 5,
  thumbnailSteps: 60,
  params: [
    { key: 'stepDeg', label: 'Passo do ângulo (graus)', type: 'select', options: [0.5, 1, 1.5, 3, 5], value: 1.5 },
    { key: 'projections', label: 'Mostrar projeções', type: 'select', options: ['sim', 'não'], value: 'sim' },
  ],
  create: (data, params) => createPca(data, params as unknown as PcaParams),
  Visual: PcaVisual,
  aux: { title: 'Variância projetada em função do ângulo', Component: VarianceCurve },
  metrics: (s) => {
    const v = s.curve.length ? s.curve[s.curve.length - 1][1] : 0;
    return [
      { label: 'Ângulo atual', value: `${((s.angle * 180) / Math.PI).toFixed(1)}°` },
      { label: 'Variância projetada', value: fmt(v, 4) },
      { label: 'Melhor encontrada', value: s.best.v >= 0 ? `${fmt(s.best.v, 4)} em ${((s.best.angle * 180) / Math.PI).toFixed(0)}°` : '...' },
      { label: 'λ₁ (PC1)', value: fmt(s.l1, 4) },
      { label: 'λ₂ (PC2)', value: fmt(s.l2, 4) },
      { label: 'Variância explicada PC1', value: pct(s.l1 / (s.l1 + s.l2 || 1)) },
    ];
  },
  docs: {
    description:
      'A análise de componentes principais (PCA) encontra as direções em que os dados variam mais. A animação gira uma reta pelo centro dos dados e projeta cada ponto nela: o espalhamento das projeções mede a variância naquela direção, e o gráfico abaixo registra esse valor para cada ângulo. A direção de variância máxima é o primeiro componente principal (PC1, em amarelo) e o PC2 (roxo) é perpendicular a ele, com comprimentos proporcionais ao desvio padrão em cada direção. Na prática o PCA resolve isso de uma vez com os autovetores da matriz de covariância. Ele é usado para reduzir dimensões, comprimir dados e visualizar conjuntos com muitos atributos.',
    steps: [
      'Centralize os dados subtraindo a média.',
      'Calcule a matriz de covariância Σ.',
      'Os autovetores de Σ são as direções principais; os autovalores, a variância em cada uma.',
      'Projete os dados nos k primeiros componentes para reduzir a dimensão.',
    ],
    complexity: [['Treino (autovalores)', 'O(n·d² + d³)'], ['Treino (SVD truncado)', 'O(n·d·k)'], ['Projeção', 'O(d·k) por ponto'], ['Espaço', 'O(d²)']],
    pseudocode: [
      'μ ← média dos dados ;  X ← dados - μ',
      'Σ ← (1/n) · Xᵀ X',
      'para θ de 0° a 180°:',
      '    u ← (cos θ, sin θ)',
      '    variância(θ) ← uᵀ Σ u',
      'PC1 ← autovetor de maior autovalor de Σ',
    ],
    exercises: [
      'Por que basta girar de 0° a 180°, e não até 360°?',
      'Mostre que uᵀ Σ u é igual à variância das projeções (u · xᵢ).',
      'Implemente a iteração de potência (power iteration) para achar o PC1 sem fórmula fechada.',
      'Aplique PCA a um conjunto real de alta dimensão (por exemplo, dígitos 8x8) e visualize os 2 primeiros componentes.',
    ],
  },
};
