/**
 * Conjuntos de dados sintéticos em 2D.
 *
 * Todos vivem no quadrado [-1, 1] x [-1, 1], que é o espaço desenhado na tela.
 * Cada um foi escolhido para mostrar uma propriedade: separabilidade linear,
 * fronteiras curvas, grupos não convexos, etc.
 */
import { clamp } from './math';
import { random, randn, uniform } from './random';
import type { Point } from './types';

export type DatasetId =
  | 'linear' | 'blobs' | 'xor' | 'circles' | 'moons' | 'spiral'
  | 'clusters' | 'elongated' | 'line' | 'sine';

export interface Dataset {
  id: DatasetId;
  name: string;
  description: string;
  generate(): Point[];
}

const keep = (v: number) => clamp(v, -0.97, 0.97);

export const DATASETS: Record<DatasetId, Dataset> = {
  linear: {
    id: 'linear',
    name: 'Linearmente separável',
    description: 'Existe uma reta que separa perfeitamente as classes.',
    generate() {
      const out: Point[] = [];
      while (out.length < 120) {
        const x = uniform(-0.9, 0.9), y = uniform(-0.9, 0.9);
        const m = 0.7 * x - 0.5 * y + 0.1;
        if (Math.abs(m) < 0.1) continue; // abre uma faixa vazia entre as classes
        out.push({ x, y, c: m > 0 ? 1 : 0 });
      }
      return out;
    },
  },
  blobs: {
    id: 'blobs',
    name: 'Duas nuvens',
    description: 'Duas gaussianas com alguma sobreposição.',
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
    description: 'Quadrantes alternados: impossível separar com uma reta.',
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
    name: 'Círculos',
    description: 'Um disco dentro de um anel.',
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
    name: 'Luas',
    description: 'Duas meias-luas encaixadas, o clássico do scikit-learn.',
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
    name: 'Espiral',
    description: 'Duas espirais entrelaçadas: difícil até para redes neurais.',
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
    name: 'Grupos (4)',
    description: 'Quatro grupos arredondados, ideal para K-Means.',
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
    name: 'Nuvem alongada',
    description: 'Dados correlacionados com uma direção dominante.',
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
    name: 'Reta com ruído',
    description: 'y = a·x + b + ruído gaussiano.',
    generate() {
      const a = 0.3 + random() * 0.6, b = random() * 0.4 - 0.2;
      return Array.from({ length: 60 }, () => {
        const x = uniform(-0.9, 0.9);
        return { x, y: keep(a * x + b + randn() * 0.12), c: 0 };
      });
    },
  },
  sine: {
    id: 'sine',
    name: 'Curva (seno)',
    description: 'Relação não linear: uma reta não basta.',
    generate() {
      return Array.from({ length: 70 }, () => {
        const x = uniform(-0.9, 0.9);
        return { x, y: keep(0.6 * Math.sin(3.2 * x) + randn() * 0.08), c: 0 };
      });
    },
  },
};
