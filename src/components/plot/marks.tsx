/**
 * Reusable SVG marks: points, lines, rings, arrows and drag handles.
 *
 * Moving marks use `motion.*` components with `initial={false}` and
 * `animate={{...}}`. When a prop changes, Framer Motion tweens from the old
 * value to the new one instead of jumping, with Manim's "smooth" easing.
 */
import { motion, type Transition } from 'framer-motion';
import { useRef } from 'react';
import { COLORS } from '../../core/colors';
import { useEdit } from '../../engine/editing';
import type { Point } from '../../core/types';
import { toWorld, usePlot } from './Plot';

/** Manim's default rate function is a smooth ease in and out. */
export const SMOOTH: Transition = { duration: 0.25, ease: [0.45, 0, 0.55, 1] };
export const FAST: Transition = { duration: 0.1, ease: 'linear' };

/* ---------- Points ---------- */

interface PointsProps {
  data: Point[];
  color: (p: Point, i: number) => string;
  radius?: number | ((p: Point, i: number) => number);
  opacity?: (p: Point, i: number) => number;
}

/**
 * Data points. Plain <circle> with a CSS transition on color and radius:
 * with hundreds of points this is lighter than one motion component each.
 */
export function Points({ data, color, radius = 4.6, opacity }: PointsProps) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {data.map((p, i) => (
        <circle
          key={i}
          className="pt"
          cx={x(p.x)}
          cy={y(p.y)}
          r={typeof radius === 'function' ? radius(p, i) : radius}
          fill={color(p, i)}
          opacity={opacity ? opacity(p, i) : 1}
        />
      ))}
    </g>
  );
}

/* ---------- Line w1·x + w2·y + b = offset ---------- */

/** Two points of the line, far enough apart to cross the whole plot. */
export function lineEndpoints(w1: number, w2: number, b: number, offset = 0): [number, number, number, number] | null {
  const far = 3;
  if (Math.abs(w2) > Math.abs(w1)) {
    if (Math.abs(w2) < 1e-12) return null;
    return [-far, (offset - b + w1 * far) / w2, far, (offset - b - w1 * far) / w2];
  }
  if (Math.abs(w1) < 1e-12) return null;
  return [(offset - b + w2 * far) / w1, -far, (offset - b - w2 * far) / w1, far];
}

interface LineProps {
  w1: number;
  w2: number;
  b: number;
  offset?: number;
  color?: string;
  width?: number;
  dash?: string;
}

/** A linear decision boundary that glides to its new place after each update. */
export function DecisionLine({ w1, w2, b, offset = 0, color = COLORS.yellow, width = 2.5, dash }: LineProps) {
  const { x, y } = usePlot();
  const e = lineEndpoints(w1, w2, b, offset);
  if (!e) return null;
  return (
    <motion.line
      initial={false}
      animate={{ x1: x(e[0]), y1: y(e[1]), x2: x(e[2]), y2: y(e[3]) }}
      transition={SMOOTH}
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dash}
      strokeLinecap="round"
      pointerEvents="none"
    />
  );
}

/* ---------- Ring ---------- */

interface RingProps {
  at: { x: number; y: number } | null | undefined;
  r?: number;
  color?: string;
  width?: number;
  dash?: string;
  /** Radius in world units (e.g. DBSCAN's ε). Overrides `r`. */
  worldRadius?: number;
}

export function Ring({ at, r = 11, color = '#fff', width = 2, dash, worldRadius }: RingProps) {
  const { x, y, size } = usePlot();
  if (!at) return null;
  const radius = worldRadius !== undefined ? (worldRadius / 2) * size : r;
  return (
    <motion.circle
      initial={false}
      animate={{ cx: x(at.x), cy: y(at.y), r: radius }}
      transition={FAST}
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dash}
      pointerEvents="none"
    />
  );
}

/* ---------- Arrow (Manim style: thin shaft, triangular tip) ---------- */

export function Arrow({ from, to, color = COLORS.yellow, width = 2.5 }: { from: [number, number]; to: [number, number]; color?: string; width?: number }) {
  const { x, y } = usePlot();
  const [ax, ay, bx, by] = [x(from[0]), y(from[1]), x(to[0]), y(to[1])];
  const ang = Math.atan2(by - ay, bx - ax);
  const head = 12;
  const sx = bx - head * 0.8 * Math.cos(ang), sy = by - head * 0.8 * Math.sin(ang);
  const p1 = [bx - head * Math.cos(ang - 0.38), by - head * Math.sin(ang - 0.38)];
  const p2 = [bx - head * Math.cos(ang + 0.38), by - head * Math.sin(ang + 0.38)];
  return (
    <g pointerEvents="none">
      <motion.line initial={false} animate={{ x1: ax, y1: ay, x2: sx, y2: sy }} transition={SMOOTH} stroke={color} strokeWidth={width} />
      <motion.polygon initial={false} animate={{ points: `${bx},${by} ${p1[0]},${p1[1]} ${p2[0]},${p2[1]}` }} transition={SMOOTH} fill={color} />
    </g>
  );
}

/* ---------- Plain segment ---------- */

export function Segment({ a, b, color, width = 1, dash, opacity = 1 }: { a: [number, number]; b: [number, number]; color: string; width?: number; dash?: string; opacity?: number }) {
  const { x, y } = usePlot();
  return <line x1={x(a[0])} y1={y(a[1])} x2={x(b[0])} y2={y(b[1])} stroke={color} strokeWidth={width} strokeDasharray={dash} opacity={opacity} pointerEvents="none" />;
}

/* ---------- Label next to a point ---------- */

export function Label({ at, children, color = '#fff', dx = 14, dy = 4 }: { at: { x: number; y: number }; children: React.ReactNode; color?: string; dx?: number; dy?: number }) {
  const { x, y, size } = usePlot();
  const px = Math.min(size - 200, x(at.x) + dx);
  return (
    <text x={px} y={Math.max(16, y(at.y) + dy)} fill={color} pointerEvents="none" className="plot-label">
      {children}
    </text>
  );
}

/* ---------- Drag handle ---------- */

interface HandleProps {
  at: { x: number; y: number };
  onDrag(x: number, y: number): void;
  color?: string;
  /** Draw the handle (false = invisible hit area over a mark that already shows it). */
  visible?: boolean;
  title: string;
}

/**
 * A draggable control point. It writes straight into the model state through
 * `onDrag`, then asks the page to redraw. Hidden in thumbnails.
 */
export function Handle({ at, onDrag, color = COLORS.yellow, visible = true, title }: HandleProps) {
  const { x, y } = usePlot();
  const edit = useEdit();
  const active = useRef(false);
  if (!edit) return null;
  return (
    <g className="handle">
      {visible && <circle cx={x(at.x)} cy={y(at.y)} r={7} fill={color} stroke="#000" strokeWidth={1.5} pointerEvents="none" />}
      <circle
        cx={x(at.x)} cy={y(at.y)} r={14} fill="transparent"
        aria-label={title}
        onPointerDown={(e) => {
          e.stopPropagation();
          (e.target as Element).setPointerCapture(e.pointerId);
          active.current = true;
        }}
        onPointerMove={(e) => {
          if (!active.current) return;
          const svg = (e.target as SVGElement).ownerSVGElement!;
          const p = toWorld(svg, e);
          onDrag(Math.max(-1, Math.min(1, p.x)), Math.max(-1, Math.min(1, p.y)));
          edit.touch();
        }}
        onPointerUp={() => { active.current = false; }}
      >
        <title>{title}</title>
      </circle>
    </g>
  );
}
