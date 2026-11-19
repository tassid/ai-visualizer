/**
 * Shapes shared by several algorithms: covariance ellipses and contour lines.
 */
import { contours } from 'd3-contour';
import { motion } from 'framer-motion';
import { ellipseOf, type Sym2 } from '../../core/linalg';
import { geoPath } from '../charts/geo';
import { SMOOTH } from './marks';
import { usePlot } from './Plot';

/**
 * Ellipses at 1σ and 2σ of a 2D Gaussian. The axes are the eigenvectors of
 * the covariance; Framer Motion tweens center, radii and rotation together.
 */
export function CovEllipse({ mean, cov, color, sigmas = [1, 2] }: { mean: [number, number]; cov: Sym2; color: string; sigmas?: number[] }) {
  const { x, y, size } = usePlot();
  const { r1, r2, angle } = ellipseOf(cov);
  const k = size / 2;
  return (
    <motion.g initial={false} animate={{ x: x(mean[0]), y: y(mean[1]), rotate: (-angle * 180) / Math.PI }} transition={SMOOTH} pointerEvents="none">
      {sigmas.map((m, i) => (
        <motion.ellipse
          key={m}
          initial={false}
          animate={{ rx: m * r1 * k, ry: m * r2 * k }}
          transition={SMOOTH}
          fill="none"
          stroke={color}
          strokeWidth={i === 0 ? 2 : 1}
          strokeDasharray={i === 0 ? undefined : '5 5'}
        />
      ))}
      <circle r={5} fill={color} stroke="#000" strokeWidth={1.5} />
    </motion.g>
  );
}

interface ContourProps {
  f: (x: number, y: number) => number;
  levels: { value: number; color: string; width?: number; dash?: string }[];
  resolution?: number;
}

/**
 * Level sets f(x, y) = value, computed by d3-contour on a grid over the plot.
 * Used for a kernel SVM's margins and for the optimizers' landscape.
 */
export function ContourLines({ f, levels, resolution = 64 }: ContourProps) {
  const { size } = usePlot();
  // The grid gets a border of very low values two cells wide, so every level
  // set closes outside the visible area instead of tracing the plot's edge.
  const G = resolution, P = G + 4;
  const values = new Array<number>(P * P).fill(-1e9);
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) values[(j + 2) * P + i + 2] = f(-1 + ((i + 0.5) / G) * 2, 1 - ((j + 0.5) / G) * 2);
  const gen = contours().size([P, P]);
  const cell = size / G;
  const proj = (cx: number, cy: number): [number, number] => [(cx - 2) * cell, (cy - 2) * cell];
  return (
    <g pointerEvents="none">
      {levels.map((l) => (
        <path key={l.value} d={geoPath(gen.contour(values, l.value), proj)} fill="none" stroke={l.color} strokeWidth={l.width ?? 1.5} strokeDasharray={l.dash} />
      ))}
    </g>
  );
}
