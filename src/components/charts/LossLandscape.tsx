/**
 * Loss landscape with contour lines (d3-contour).
 *
 * We evaluate the loss on a grid of (w, b), and d3-contour turns that grid
 * into polygons of equal height, like a topographic map. On top we draw the
 * path gradient descent has walked.
 */
import { contours } from 'd3-contour';
import { scaleLinear, scaleSequential } from 'd3-scale';
import { interpolateViridis } from 'd3-scale-chromatic';
import { line } from 'd3-shape';
import { motion } from 'framer-motion';
import { useMemo } from 'react';
import { COLORS } from '../../core/colors';
import { geoIdentity, geoPath } from './geo';

interface Props {
  loss: (w: number, b: number) => number;
  /** Changes when the data changes, to recompute the grid. */
  dataKey: unknown;
  path: [number, number][];
  wDomain?: [number, number];
  bDomain?: [number, number];
  width: number;
  height: number;
}

const GRID_W = 80;

export function LossLandscape({ loss, dataKey, path, wDomain = [-2, 2], bDomain = [-1, 1], width, height }: Props) {
  const gridH = Math.round((GRID_W * height) / width);

  const { polys, color } = useMemo(() => {
    const values = new Array<number>(GRID_W * gridH);
    for (let j = 0; j < gridH; j++) {
      for (let i = 0; i < GRID_W; i++) {
        const w = wDomain[0] + ((i + 0.5) / GRID_W) * (wDomain[1] - wDomain[0]);
        const b = bDomain[1] - ((j + 0.5) / gridH) * (bDomain[1] - bDomain[0]);
        values[j * GRID_W + i] = Math.log(loss(w, b) + 1e-6); // log scale keeps the rings readable
      }
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const thresholds = Array.from({ length: 16 }, (_, k) => lo + ((hi - lo) * k) / 16);
    return {
      polys: contours().size([GRID_W, gridH]).thresholds(thresholds)(values),
      color: scaleSequential(interpolateViridis).domain([hi + (hi - lo) * 0.4, lo]),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataKey, width, height]);

  const proj = geoIdentity(width / GRID_W, height / gridH);
  const sw = scaleLinear().domain(wDomain).range([0, width]);
  const sb = scaleLinear().domain(bDomain).range([height, 0]);
  const trail = line<[number, number]>().x((d) => sw(d[0])).y((d) => sb(d[1]))(path) ?? '';
  const cur = path[path.length - 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg">
      {polys.map((p, k) => (
        <path key={k} d={geoPath(p, proj)} fill={color(p.value)} fillOpacity={0.55} stroke={color(p.value)} strokeWidth={1} />
      ))}
      <path d={trail} fill="none" stroke={COLORS.yellow} strokeWidth={2} />
      {cur && <motion.circle initial={false} animate={{ cx: sw(cur[0]), cy: sb(cur[1]) }} transition={{ duration: 0.12 }} r={6} fill={COLORS.yellow} stroke="#000" />}
      <text x={width - 10} y={height - 10} textAnchor="end" className="node-label math">w</text>
      <text x={10} y={20} className="node-label math">b</text>
    </svg>
  );
}
