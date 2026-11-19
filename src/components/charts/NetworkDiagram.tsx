/**
 * Diagrama de uma rede neural densa, inspirado no TensorFlow Playground.
 *
 * Cada neurônio é um quadradinho que mostra seu mapa de ativação: o valor que
 * ele produz para cada ponto do plano de entrada. É a forma mais direta de
 * ver que cada camada oculta "dobra" o espaço de um jeito diferente.
 *
 * Camadas:
 *   canvas  → mapas de ativação (milhares de pixels, rápido)
 *   svg     → conexões (cor = sinal do peso, espessura = magnitude) e molduras
 */
import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import { divergingColor } from '../../core/colors';

export interface NetworkSnapshot {
  sizes: number[];
  /** weights[l][i][j]: peso da conexão do neurônio j (camada l) para o neurônio i (camada l + 1). */
  weights: ArrayLike<number>[][];
  /** maps[l][k]: grade G x G com a ativação do neurônio k da camada l. */
  maps: Float64Array[][];
  /** Intervalo de cada camada para normalizar as cores. */
  ranges: [number, number][];
  grid: number;
}

interface Props {
  net: NetworkSnapshot;
  width: number;
  height: number;
  version: number;
}

export function NetworkDiagram({ net, width, height, version }: Props) {
  const { sizes, weights, maps, ranges, grid: G } = net;
  const cols = sizes.length;
  const top = 34;
  const avail = height - top - 12;
  const colX = (l: number) => 60 + (l * (width - 120)) / (cols - 1);
  const box = (l: number) => Math.min(34, avail / sizes[l] - 5);
  const nodeY = (l: number, k: number) => top + avail / 2 + (k - (sizes[l] - 1) / 2) * (box(l) + 5);

  // Desenha os mapas de ativação no canvas.
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = width * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const tile = document.createElement('canvas');
    tile.width = tile.height = G;
    const tctx = tile.getContext('2d')!;
    const img = tctx.createImageData(G, G);
    for (let l = 0; l < cols; l++) {
      const [lo, hi] = ranges[l];
      const s = box(l);
      for (let k = 0; k < sizes[l]; k++) {
        const m = maps[l][k];
        let max = hi;
        if (!Number.isFinite(hi)) { max = 1e-9; for (const v of m) max = Math.max(max, v); } // ReLU: normaliza pelo máximo
        for (let q = 0; q < G * G; q++) {
          const [r, g, b] = divergingColor((m[q] - lo) / (max - lo));
          img.data[q * 4] = r; img.data[q * 4 + 1] = g; img.data[q * 4 + 2] = b; img.data[q * 4 + 3] = 255;
        }
        tctx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(tile, colX(l) - s / 2, nodeY(l, k) - s / 2, s, s);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, width, height]);

  // Normaliza a espessura das conexões pelo maior peso de cada camada.
  const edges: React.ReactNode[] = [];
  for (let l = 0; l < weights.length; l++) {
    let max = 1e-9;
    for (const row of weights[l]) for (let j = 0; j < row.length; j++) max = Math.max(max, Math.abs(row[j]));
    weights[l].forEach((row, i) => {
      for (let j = 0; j < row.length; j++) {
        const w = row[j], a = Math.abs(w) / max;
        edges.push(
          <line
            key={`${l}-${i}-${j}`}
            className="edge"
            x1={colX(l) + box(l) / 2} y1={nodeY(l, j)}
            x2={colX(l + 1) - box(l + 1) / 2} y2={nodeY(l + 1, i)}
            stroke={w > 0 ? `rgba(78,205,196,${0.1 + a * 0.8})` : `rgba(255,107,107,${0.1 + a * 0.8})`}
            strokeWidth={0.4 + a * 3}
          />,
        );
      }
    });
  }

  const layerName = (l: number) => (l === 0 ? 'Entrada' : l === cols - 1 ? 'Saída' : `Oculta ${l}`);

  return (
    <div className="network" style={{ aspectRatio: `${width} / ${height}` }}>
      <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg">
        <g>{edges}</g>
      </svg>
      <canvas ref={canvasRef} />
      <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg">
        {sizes.map((n, l) => (
          <g key={l}>
            <text x={colX(l)} y={18} textAnchor="middle" className="node-label muted">{layerName(l)} ({n})</text>
            {Array.from({ length: n }, (_, k) => (
              <motion.rect
                key={`${l}-${k}-${n}`}
                initial={{ opacity: 0, scale: 0.4 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: l * 0.06 + k * 0.015 }}
                style={{ transformOrigin: `${colX(l)}px ${nodeY(l, k)}px` }}
                x={colX(l) - box(l) / 2} y={nodeY(l, k) - box(l) / 2}
                width={box(l)} height={box(l)}
                fill="none" stroke="rgba(255,255,255,.3)" rx={2}
              />
            ))}
          </g>
        ))}
        <text x={colX(0) - box(0) / 2 - 10} y={nodeY(0, 0) + 4} textAnchor="middle" className="node-label muted">x</text>
        <text x={colX(0) - box(0) / 2 - 10} y={nodeY(0, 1) + 4} textAnchor="middle" className="node-label muted">y</text>
      </svg>
    </div>
  );
}
