/**
 * Line chart with D3 (scales and line generator), rendered by React.
 *
 * Pattern used across the project: D3 computes (scales, path strings),
 * React draws. D3 never touches the DOM directly.
 *
 * The only printed number is the latest value, placed at the end of the line.
 */
import { extent } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import { line, curveMonotoneX } from 'd3-shape';
import { motion } from 'framer-motion';
import { COLORS } from '../../core/colors';

interface Props {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}

const MAX_POINTS = 400;

export function LineChart({ values, width = 320, height = 110, color = COLORS.yellow }: Props) {
  const pad = { l: 4, r: 64, t: 10, b: 10 };
  if (!values.length) return <svg viewBox={`0 0 ${width} ${height}`} className="chart" aria-hidden />;

  const stride = Math.max(1, Math.floor(values.length / MAX_POINTS));
  const pts = values.map((v, i) => [i, v] as [number, number]).filter(([i]) => i % stride === 0 || i === values.length - 1);
  let [lo, hi] = extent(pts, (d) => d[1]) as [number, number];
  if (hi - lo < 1e-12) { hi += 1; lo -= 1e-9; }
  const sx = scaleLinear().domain([0, Math.max(1, values.length - 1)]).range([pad.l, width - pad.r]);
  const sy = scaleLinear().domain([lo, hi]).range([height - pad.b, pad.t]);
  const path = line<[number, number]>().x((d) => sx(d[0])).y((d) => sy(d[1])).curve(curveMonotoneX)(pts)!;
  const last = values[values.length - 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart">
      <line x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} className="chart-base" />
      <path d={path} fill="none" stroke={color} strokeWidth={2} />
      <motion.g initial={false} animate={{ x: sx(values.length - 1), y: sy(last) }} transition={{ duration: 0.1 }}>
        <circle r={3.5} fill={color} />
        <text x={8} y={4} className="chart-value">{Math.abs(last) >= 100 ? last.toFixed(0) : last.toFixed(4)}</text>
      </motion.g>
    </svg>
  );
}
