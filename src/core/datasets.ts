/**
 * Synthetic 2D datasets.
 *
 * All of them live in the square [-1, 1] x [-1, 1], which is the area drawn
 * on screen. Each one shows a property: linear separability, curved
 * boundaries, non-convex clusters, and so on.
 */
import { clamp } from './math';
import { random, randn, uniform } from './random';
import type { Point } from './types';

export type DatasetId =
  | 'linear' | 'blobs' | 'xor' | 'circles' | 'moons' | 'spiral'
  | 'clusters' | 'elongated' | 'line' | 'sine' | 'tilted' | 'few';

export interface Dataset {
  id: DatasetId;
  name: string;
  generate(): Point[];
}

const keep = (v: number) => clamp(v, -0.97, 0.97);

export const DATASETS: Record<DatasetId, Dataset> = {
  linear: {
    id: 'linear',
    name: 'Linearly separable',
    generate() {
      const out: Point[] = [];
      while (out.length < 120) {
        const x = uniform(-0.9, 0.9), y = uniform(-0.9, 0.9);
        const m = 0.7 * x - 0.5 * y + 0.1;
        if (Math.abs(m) < 0.1) continue; // leaves an empty band between the classes
        out.push({ x, y, c: m > 0 ? 1 : 0 });
      }
      return out;
    },
  },
  blobs: {
    id: 'blobs',
    name: 'Two blobs',
    generate() {
      return Array.from({ length: 140 }, (_, i) => {
        const c = i % 2;
        return { x: keep((c ? 0.38 : -0.38) + randn() * 0.2), y: keep((c ? 0.32 : -0.3) + randn() * 0.2), c };
      });
    },
  },
  xor: {
    id: 'xor',
    name: 'XOR',
    generate() {
      const out: Point[] = [];
      while (out.length < 160) {
        const x = uniform(-0.9, 0.9), y = uniform(-0.9, 0.9);
        if (Math.abs(x) < 0.07 || Math.abs(y) < 0.07) continue;
        out.push({ x, y, c: x * y > 0 ? 1 : 0 });
      }
      return out;
    },
  },
  circles: {
    id: 'circles',
    name: 'Circles',
    generate() {
      return Array.from({ length: 180 }, (_, i) => {
        const c = i % 2, t = random() * Math.PI * 2;
        const r = c ? 0.32 * Math.sqrt(random()) : 0.56 + 0.3 * random();
        return { x: r * Math.cos(t) + randn() * 0.03, y: r * Math.sin(t) + randn() * 0.03, c };
      });
    },
  },
  moons: {
    id: 'moons',
    name: 'Moons',
    generate() {
      return Array.from({ length: 180 }, (_, i) => {
        const c = i % 2, t = random() * Math.PI;
        const x = c ? 1 - Math.cos(t) : Math.cos(t);
        const y = c ? 0.5 - Math.sin(t) : Math.sin(t);
        return { x: (x - 0.5) * 0.62 + randn() * 0.04, y: (y - 0.25) * 0.62 + randn() * 0.04, c };
      });
    },
  },
  spiral: {
    id: 'spiral',
    name: 'Spiral',
    generate() {
      const out: Point[] = [], m = 110;
      for (let i = 0; i < m; i++) {
        const r = 0.06 + (i / m) * 0.82, t = 1.6 * (i / m) * 2 * Math.PI;
        for (const c of [0, 1]) {
          const a = t + c * Math.PI;
          out.push({ x: r * Math.sin(a) + randn() * 0.02, y: r * Math.cos(a) + randn() * 0.02, c });
        }
      }
      return out;
    },
  },
  clusters: {
    id: 'clusters',
    name: 'Four clusters',
    generate() {
      const centers = [[-0.5, 0.5], [0.5, 0.45], [-0.42, -0.5], [0.48, -0.42]].map(([a, b]) => [a + randn() * 0.08, b + randn() * 0.08]);
      return Array.from({ length: 220 }, (_, i) => {
        const k = i % 4;
        return { x: keep(centers[k][0] + randn() * 0.14), y: keep(centers[k][1] + randn() * 0.14), c: k % 2 };
      });
    },
  },
  elongated: {
    id: 'elongated',
    name: 'Correlated cloud',
    generate() {
      const th = 0.35 + random() * 1.2;
      return Array.from({ length: 160 }, () => {
        const u = randn() * 0.42, v = randn() * 0.11;
        return { x: keep(u * Math.cos(th) - v * Math.sin(th) + 0.05), y: keep(u * Math.sin(th) + v * Math.cos(th) - 0.03), c: 0 };
      });
    },
  },
  line: {
    id: 'line',
    name: 'Noisy line',
    generate() {
      const a = 0.3 + random() * 0.6, b = random() * 0.4 - 0.2;
      return Array.from({ length: 40 }, () => {
        const x = uniform(-0.9, 0.9);
        return { x, y: keep(a * x + b + randn() * 0.12), c: 0 };
      });
    },
  },
  tilted: {
    id: 'tilted',
    name: 'Tilted classes',
    generate() {
      // Two elongated classes with the same tilt: the features are correlated within each class.
      const th = 0.6;
      return Array.from({ length: 160 }, (_, i) => {
        const c = i % 2, u = randn() * 0.4, v = randn() * 0.09;
        const cx = c ? 0.12 : -0.12, cy = c ? -0.16 : 0.16;
        return { x: keep(cx + u * Math.cos(th) - v * Math.sin(th)), y: keep(cy + u * Math.sin(th) + v * Math.cos(th)), c };
      });
    },
  },
  few: {
    id: 'few',
    name: 'Few samples of a curve',
    generate() {
      return Array.from({ length: 10 }, () => {
        const x = uniform(-0.9, 0.9);
        return { x, y: keep(0.55 * Math.sin(3 * x) + 0.15 * x + randn() * 0.04), c: 0 };
      });
    },
  },
  sine: {
    id: 'sine',
    name: 'Sine curve',
    generate() {
      return Array.from({ length: 50 }, () => {
        const x = uniform(-0.9, 0.9);
        return { x, y: keep(0.6 * Math.sin(3.2 * x) + randn() * 0.08), c: 0 };
      });
    },
  },
};
