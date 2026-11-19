/**
 * Gráfico de linha com D3 (escalas + gerador de linha/área) renderizado em SVG pelo React.
 *
 * Padrão usado no projeto: o D3 calcula (escalas, caminhos `d`), o React
 * desenha. Assim não há manipulação direta do DOM pelo D3.
 */
import { extent } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import { area, line, curveMonotoneX } from 'd3-shape';
import { motion } from 'framer-motion';

interface Props {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  emptyText?: string;
}

const MAX_POINTS = 400;

export function LineChart({ values, width = 320, height = 130, color = 'var(--accent)', emptyText = 'O gráfico aparece quando o treino começar.' }: Props) {
  const pad = { l: 8, r: 8, t: 22, b: 14 };
  if (!values.length) {
    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        <text x={width / 2} y={height / 2} textAnchor="middle" className="chart-empty">{emptyText}</text>
      </svg>
    );
  }
  // Reduz para no máximo MAX_POINTS pontos, mantendo sempre o último.
  const stride = Math.max(1, Math.floor(values.length / MAX_POINTS));
  const pts = values.map((v, i) => [i, v] as [number, number]).filter(([i]) => i % stride === 0 || i === values.length - 1);

  let [lo, hi] = extent(pts, (d) => d[1]) as [number, number];
  if (hi - lo < 1e-12) { hi += 1; lo -= 1e-9; }
  const sx = scaleLinear().domain([0, Math.max(1, values.length - 1)]).range([pad.l, width - pad.r]);
  const sy = scaleLinear().domain([lo, hi]).range([height - pad.b, pad.t]);

  const path = line<[number, number]>().x((d) => sx(d[0])).y((d) => sy(d[1])).curve(curveMonotoneX)(pts)!;
  const fill = area<[number, number]>().x((d) => sx(d[0])).y0(height - pad.b).y1((d) => sy(d[1])).curve(curveMonotoneX)(pts)!;
  const last = values[values.length - 1];
  const f = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(4));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart">
      <path d={fill} fill={color} opacity={0.1} />
      <path d={path} fill="none" stroke={color} strokeWidth={2} />
      <motion.circle initial={false} animate={{ cx: sx(values.length - 1), cy: sy(last) }} transition={{ duration: 0.1 }} r={3.5} fill="#fff" />
      <text x={pad.l} y={13} className="chart-label">máx {f(hi)}</text>
      <text x={width - pad.r} y={13} textAnchor="end" className="chart-value">atual: {f(last)}</text>
    </svg>
  );
}
