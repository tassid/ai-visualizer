/**
 * Paisagem da perda com curvas de nível (d3-contour).
 *
 * Avaliamos a função de perda numa grade de (w, b), e o d3-contour transforma
 * essa grade em polígonos de mesma altura, como num mapa topográfico.
 * Por cima, desenhamos o caminho percorrido pela descida de gradiente.
 */
import { contours } from 'd3-contour';
import { geoPath, geoIdentity } from './geo';
import { scaleLinear, scaleSequential } from 'd3-scale';
import { interpolateViridis } from 'd3-scale-chromatic';
import { line } from 'd3-shape';
import { motion } from 'framer-motion';
import { useMemo } from 'react';

interface Props {
  loss: (w: number, b: number) => number;
  /** Muda quando os dados mudam, para recalcular a grade. */
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
        values[j * GRID_W + i] = Math.log(loss(w, b) + 1e-6); // log deixa as curvas mais legíveis
      }
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const thresholds = Array.from({ length: 16 }, (_, k) => lo + ((hi - lo) * k) / 16);
    const polys = contours().size([GRID_W, gridH]).thresholds(thresholds)(values);
    const color = scaleSequential(interpolateViridis).domain([hi, lo]);
    return { polys, color };
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
        <path key={k} d={geoPath(p, proj)} fill={color(p.value)} stroke="rgba(0,0,0,.25)" strokeWidth={0.6} />
      ))}
      <path d={trail} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
      {cur && (
        <motion.circle initial={false} animate={{ cx: sw(cur[0]), cy: sb(cur[1]) }} transition={{ duration: 0.12 }} r={6} fill="var(--accent)" stroke="#000" />
      )}
      <text x={width - 10} y={height - 10} textAnchor="end" className="node-label">w (inclinação) →</text>
      <text x={10} y={18} className="node-label">↑ b (intercepto)</text>
    </svg>
  );
}
