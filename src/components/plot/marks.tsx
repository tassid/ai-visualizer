/**
 * Elementos SVG reutilizáveis: pontos, retas, anéis e setas.
 *
 * Os que se movem usam componentes `motion.*` do Framer Motion com
 * `initial={false}` e `animate={{...}}`. Quando a prop muda, o Framer anima
 * do valor antigo para o novo com uma mola (spring), em vez de pular.
 */
import { motion, type Transition } from 'framer-motion';
import { usePlot } from './Plot';
import type { Point } from '../../core/types';

export const SPRING: Transition = { type: 'spring', stiffness: 260, damping: 28, mass: 0.6 };
export const FAST: Transition = { duration: 0.12, ease: 'linear' };

/* ---------- Pontos ---------- */

interface PointsProps {
  data: Point[];
  color: (p: Point, i: number) => string;
  radius?: number | ((p: Point, i: number) => number);
  opacity?: (p: Point, i: number) => number;
}

/**
 * Pontos de dados. Usamos <circle> comum com transição CSS de cor e raio
 * (classe .pt no CSS): com centenas de pontos isso é mais leve do que um
 * componente motion por ponto.
 */
export function Points({ data, color, radius = 4.6, opacity }: PointsProps) {
  const { x, y } = usePlot();
  return (
    <g pointerEvents="none">
      {data.map((p, i) => {
        const r = typeof radius === 'function' ? radius(p, i) : radius;
        return (
          <circle
            key={i}
            className="pt"
            cx={x(p.x)}
            cy={y(p.y)}
            r={r}
            fill={color(p, i)}
            opacity={opacity ? opacity(p, i) : 1}
          />
        );
      })}
    </g>
  );
}

/* ---------- Reta w1·x + w2·y + b = offset ---------- */

/** Calcula dois pontos da reta, longe o bastante para cruzar todo o gráfico. */
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
  transition?: Transition;
}

/** Fronteira de decisão linear que desliza suavemente a cada atualização dos pesos. */
export function DecisionLine({ w1, w2, b, offset = 0, color = 'var(--accent)', width = 2.6, dash, transition = SPRING }: LineProps) {
  const { x, y } = usePlot();
  const e = lineEndpoints(w1, w2, b, offset);
  if (!e) return null;
  return (
    <motion.line
      initial={false}
      animate={{ x1: x(e[0]), y1: y(e[1]), x2: x(e[2]), y2: y(e[3]) }}
      transition={transition}
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dash}
      strokeLinecap="round"
      pointerEvents="none"
    />
  );
}

/* ---------- Anel de destaque ---------- */

interface RingProps {
  at: { x: number; y: number } | null | undefined;
  r?: number;
  color?: string;
  width?: number;
  dash?: string;
  /** Raio em unidades do mundo (ex.: raio ε do DBSCAN). Sobrescreve `r`. */
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

/* ---------- Seta ---------- */

export function Arrow({ from, to, color = 'var(--accent)', width = 2.5 }: { from: [number, number]; to: [number, number]; color?: string; width?: number }) {
  const { x, y } = usePlot();
  const [ax, ay, bx, by] = [x(from[0]), y(from[1]), x(to[0]), y(to[1])];
  const ang = Math.atan2(by - ay, bx - ax);
  const head = 11;
  const p1 = [bx - head * Math.cos(ang - 0.4), by - head * Math.sin(ang - 0.4)];
  const p2 = [bx - head * Math.cos(ang + 0.4), by - head * Math.sin(ang + 0.4)];
  return (
    <g pointerEvents="none">
      <motion.line initial={false} animate={{ x1: ax, y1: ay, x2: bx, y2: by }} transition={SPRING} stroke={color} strokeWidth={width} />
      <motion.polygon initial={false} animate={{ points: `${bx},${by} ${p1[0]},${p1[1]} ${p2[0]},${p2[1]}` }} transition={SPRING} fill={color} />
    </g>
  );
}

/* ---------- Segmento simples (sem animação) ---------- */

export function Segment({ a, b, color, width = 1, dash, opacity = 1 }: { a: [number, number]; b: [number, number]; color: string; width?: number; dash?: string; opacity?: number }) {
  const { x, y } = usePlot();
  return <line x1={x(a[0])} y1={y(a[1])} x2={x(b[0])} y2={y(b[1])} stroke={color} strokeWidth={width} strokeDasharray={dash} opacity={opacity} pointerEvents="none" />;
}

/* ---------- Rótulo ---------- */

export function Label({ at, children, color = '#fff', dx = 14, dy = 4, size = 12 }: { at: { x: number; y: number }; children: React.ReactNode; color?: string; dx?: number; dy?: number; size?: number }) {
  const { x, y, size: S } = usePlot();
  const px = Math.min(S - 190, x(at.x) + dx);
  return (
    <text x={px} y={Math.max(14, y(at.y) + dy)} fill={color} fontSize={size} fontWeight={500} pointerEvents="none" className="plot-label">
      {children}
    </text>
  );
}
