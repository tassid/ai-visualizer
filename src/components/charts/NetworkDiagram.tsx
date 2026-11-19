/**
 * Diagram of a dense neural network, after the 3Blue1Brown neural network
 * series and the TensorFlow Playground.
 *
 * Each neuron is a square showing its activation map: the value it outputs
 * for every point of the input plane. It is the most direct way to see that
 * each hidden layer bends the space in its own way.
 *
 * When you hover the plot, the network runs a forward pass on that point:
 * each neuron's outline lights up with its activation, and edges carry
 * weight × activation, the signal that actually flows to the next layer.
 *
 * Layers:
 *   canvas → activation maps (thousands of pixels, fast)
 *   svg    → edges and outlines
 */
import { useEffect, useRef } from 'react';
import { COLORS, divergingColor, rgbCss } from '../../core/colors';

export interface NetworkSnapshot {
  sizes: number[];
  /** weights[l][i][j]: weight from neuron j (layer l) to neuron i (layer l + 1). */
  weights: ArrayLike<number>[][];
  /** maps[l][k]: G x G grid with the activation of neuron k in layer l. */
  maps: Float64Array[][];
  /** Output range of each layer, used to color it. */
  ranges: [number, number][];
  grid: number;
  /** Activations for the hovered point, if any. */
  probe: Float64Array[] | null;
}

interface Props {
  net: NetworkSnapshot;
  width: number;
  height: number;
  version: number;
}

export function NetworkDiagram({ net, width, height, version }: Props) {
  const { sizes, weights, maps, ranges, grid: G, probe } = net;
  const cols = sizes.length;
  const top = 34;
  const avail = height - top - 12;
  const colX = (l: number) => 60 + (l * (width - 120)) / (cols - 1);
  const box = (l: number) => Math.min(34, avail / sizes[l] - 5);
  const nodeY = (l: number, k: number) => top + avail / 2 + (k - (sizes[l] - 1) / 2) * (box(l) + 5);
  const norm = (l: number, v: number, max: number) => (v - ranges[l][0]) / ((Number.isFinite(ranges[l][1]) ? ranges[l][1] : max) - ranges[l][0]);

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
      const s = box(l);
      for (let k = 0; k < sizes[l]; k++) {
        const m = maps[l][k];
        let max = 1e-9;
        for (const v of m) max = Math.max(max, v); // used by ReLU, which has no upper bound
        for (let q = 0; q < G * G; q++) {
          const [r, g, b] = divergingColor(norm(l, m[q], max));
          img.data[q * 4] = r; img.data[q * 4 + 1] = g; img.data[q * 4 + 2] = b; img.data[q * 4 + 3] = 255;
        }
        tctx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(tile, colX(l) - s / 2, nodeY(l, k) - s / 2, s, s);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, width, height]);

  // Edge thickness is normalized by the largest |weight| (or |signal|) in each layer.
  const edges: React.ReactNode[] = [];
  for (let l = 0; l < weights.length; l++) {
    const signal = (i: number, j: number) => weights[l][i][j] * (probe ? probe[l][j] : 1);
    let max = 1e-9;
    weights[l].forEach((row, i) => { for (let j = 0; j < row.length; j++) max = Math.max(max, Math.abs(signal(i, j))); });
    weights[l].forEach((row, i) => {
      for (let j = 0; j < row.length; j++) {
        const v = signal(i, j), a = Math.abs(v) / max;
        edges.push(
          <line
            key={`${l}-${i}-${j}`}
            className="edge"
            x1={colX(l) + box(l) / 2} y1={nodeY(l, j)}
            x2={colX(l + 1) - box(l + 1) / 2} y2={nodeY(l + 1, i)}
            stroke={v > 0 ? COLORS.blue : COLORS.red}
            strokeOpacity={0.08 + a * 0.85}
            strokeWidth={0.4 + a * 3}
          />,
        );
      }
    });
  }

  const layerName = (l: number) => (l === 0 ? 'Input' : l === cols - 1 ? 'Output' : `Hidden ${l}`);

  return (
    <div className="network" style={{ aspectRatio: `${width} / ${height}` }}>
      <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg"><g>{edges}</g></svg>
      <canvas ref={canvasRef} />
      <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg">
        {sizes.map((n, l) => (
          <g key={l}>
            <text x={colX(l)} y={18} textAnchor="middle" className="node-label muted">{layerName(l)}</text>
            {Array.from({ length: n }, (_, k) => {
              const s = box(l);
              const lit = probe ? rgbCss(divergingColor(norm(l, probe[l][k], 1))) : 'rgba(255,255,255,.25)';
              return <rect key={k} x={colX(l) - s / 2} y={nodeY(l, k) - s / 2} width={s} height={s} rx={2} fill="none" stroke={lit} strokeWidth={probe ? 2.5 : 1} />;
            })}
          </g>
        ))}
        <text x={colX(0) - box(0) / 2 - 10} y={nodeY(0, 0) + 4} textAnchor="middle" className="node-label math">x</text>
        <text x={colX(0) - box(0) / 2 - 10} y={nodeY(0, 1) + 4} textAnchor="middle" className="node-label math">y</text>
        {probe && (
          <text x={colX(cols - 1) + box(cols - 1) / 2 + 8} y={nodeY(cols - 1, 0) + 4} className="node-label">
            {probe[cols - 1][0].toFixed(2)}
          </text>
        )}
      </svg>
    </div>
  );
}
