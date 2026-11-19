/**
 * The main drawing area: a canvas (decision background) under an SVG
 * (points, lines and handles animated with Framer Motion).
 *
 * Why both? The background has thousands of pixels that change every frame,
 * which canvas draws fast. Elements with meaning (lines, centroids,
 * neighbors) live in the SVG, where Framer Motion can tween them between
 * steps and the pointer can grab them.
 *
 * D3 scales (d3-scale) map world coordinates [-1, 1] to SVG pixels [0, 560].
 * The y axis is flipped because SVG y grows downward.
 *
 * The grid copies Manim's NumberPlane: faint blue lines, white axes with ticks.
 */
import { scaleLinear, type ScaleLinear } from 'd3-scale';
import { motion } from 'framer-motion';
import { createContext, useContext, useMemo, useRef, type ReactNode } from 'react';
import type { RGBA } from '../../core/colors';
import { useEdit } from '../../engine/editing';
import { Heatmap } from './Heatmap';

export const PLOT_SIZE = 560;

export interface PlotScales {
  x: ScaleLinear<number, number>;
  y: ScaleLinear<number, number>;
  size: number;
}

const makeScales = (size: number): PlotScales => ({
  x: scaleLinear().domain([-1, 1]).range([0, size]),
  y: scaleLinear().domain([-1, 1]).range([size, 0]),
  size,
});

const PlotContext = createContext<PlotScales>(makeScales(PLOT_SIZE));
export const usePlot = () => useContext(PlotContext);

export interface HeatmapSpec {
  color: (x: number, y: number) => RGBA;
  version: number;
  resolution?: number;
}

/** Converts a pointer event into world coordinates. */
export function toWorld(svg: SVGSVGElement, e: { clientX: number; clientY: number }) {
  const r = svg.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * 2 - 1, y: 1 - ((e.clientY - r.top) / r.height) * 2 };
}

export function Plot({ heatmap, children }: { heatmap?: HeatmapSpec | null; children?: ReactNode }) {
  const scales = useMemo(() => makeScales(PLOT_SIZE), []);
  const edit = useEdit();
  const ref = useRef<SVGSVGElement>(null);
  return (
    <PlotContext.Provider value={scales}>
      <div className={'plot' + (edit ? ' editable' : '')}>
        {heatmap && <Heatmap {...heatmap} />}
        <svg ref={ref} viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`} onPointerLeave={() => edit?.onHover(null)}>
          <NumberPlane animate={!!edit} />
          {edit && <EditLayer svg={ref} />}
          {children}
        </svg>
      </div>
    </PlotContext.Provider>
  );
}

/** Manim-style grid. On the algorithm page the lines draw themselves in once. */
function NumberPlane({ animate }: { animate: boolean }) {
  const { x, y, size } = usePlot();
  const minor = [-0.75, -0.25, 0.25, 0.75];
  const major = [-1, -0.5, 0.5, 1];
  const draw = (i: number) =>
    animate ? { initial: { pathLength: 0 }, animate: { pathLength: 1 }, transition: { duration: 0.6, delay: i * 0.03, ease: [0.22, 1, 0.36, 1] as const } } : {};
  return (
    <g pointerEvents="none">
      {minor.map((t) => (
        <g key={`m${t}`} className="plane-minor">
          <line x1={x(t)} x2={x(t)} y1={0} y2={size} />
          <line y1={y(t)} y2={y(t)} x1={0} x2={size} />
        </g>
      ))}
      {major.map((t, i) => (
        <g key={`M${t}`} className="plane-major">
          <motion.line {...draw(i)} x1={x(t)} x2={x(t)} y1={size} y2={0} />
          <motion.line {...draw(i)} y1={y(t)} y2={y(t)} x1={0} x2={size} />
        </g>
      ))}
      <motion.line className="plane-axis" {...draw(0)} x1={0} x2={size} y1={y(0)} y2={y(0)} />
      <motion.line className="plane-axis" {...draw(0)} x1={x(0)} x2={x(0)} y1={size} y2={0} />
      {major.filter((t) => Math.abs(t) < 1).map((t) => (
        <g key={`t${t}`} className="plane-tick">
          <line x1={x(t)} x2={x(t)} y1={y(0) - 5} y2={y(0) + 5} />
          <line y1={y(t)} y2={y(t)} x1={x(0) - 5} x2={x(0) + 5} />
          <text x={x(t)} y={y(0) + 20} textAnchor="middle">{t}</text>
          <text x={x(0) - 10} y={y(t) + 4} textAnchor="end">{t}</text>
        </g>
      ))}
    </g>
  );
}

/**
 * Invisible layer that edits the data:
 *   click on empty space   → add a point (of the selected class)
 *   drag a point           → move it
 *   shift-click a point    → delete it
 * It sits under the marks; marks ignore the pointer, handles do not.
 */
function EditLayer({ svg }: { svg: React.RefObject<SVGSVGElement | null> }) {
  const edit = useEdit()!;
  const { size } = usePlot();
  const dragging = useRef<number | null>(null);
  const data = useRef(edit.data);
  data.current = edit.data;

  const nearest = (p: { x: number; y: number }) => {
    let best = -1, bestD = (14 / size) * 2; // 14 px, in world units
    data.current.forEach((q, i) => {
      const d = Math.hypot(q.x - p.x, q.y - p.y);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  };

  return (
    <rect
      x={0} y={0} width={size} height={size} fill="transparent"
      onPointerDown={(e) => {
        const p = toWorld(svg.current!, e);
        const i = nearest(p);
        if (e.shiftKey || e.button === 2) {
          if (i >= 0) edit.setData(data.current.filter((_, k) => k !== i));
          return;
        }
        (e.target as Element).setPointerCapture(e.pointerId);
        if (i >= 0) dragging.current = i;
        else {
          dragging.current = data.current.length;
          edit.setData([...data.current, { x: p.x, y: p.y, c: edit.brush ?? 0 }]);
        }
      }}
      onPointerMove={(e) => {
        const p = toWorld(svg.current!, e);
        edit.onHover(p);
        const i = dragging.current;
        if (i === null || !data.current[i]) return;
        const next = data.current.slice();
        next[i] = { ...next[i], x: Math.max(-0.99, Math.min(0.99, p.x)), y: Math.max(-0.99, Math.min(0.99, p.y)) };
        edit.setData(next);
      }}
      onPointerUp={() => { dragging.current = null; }}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}
