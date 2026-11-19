/**
 * Decision background drawn on a canvas.
 *
 * The canvas is low resolution (e.g. 72 x 72 pixels) and stretched by CSS.
 * The browser interpolates while scaling up, which gives a smooth gradient
 * for free and keeps the cost low: 72² = 5,184 model evaluations per frame
 * instead of 560² = 313,600.
 */
import { useEffect, useRef } from 'react';
import { hexToRgb, COLORS, type RGBA } from '../../core/colors';

const BG = hexToRgb(COLORS.bg);

interface Props {
  color: (x: number, y: number) => RGBA;
  version: number;
  resolution?: number;
}

export function Heatmap({ color, version, resolution = 72 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const R = resolution;
    canvas.width = R;
    canvas.height = R;
    const ctx = canvas.getContext('2d')!;
    const img = ctx.createImageData(R, R);
    for (let j = 0; j < R; j++) {
      for (let i = 0; i < R; i++) {
        // center of pixel (i, j) in world coordinates
        const wx = -1 + ((i + 0.5) / R) * 2;
        const wy = 1 - ((j + 0.5) / R) * 2;
        const [r, g, b, a] = color(wx, wy);
        const k = (j * R + i) * 4;
        // manual alpha compositing over the background
        img.data[k] = BG[0] * (1 - a) + r * a;
        img.data[k + 1] = BG[1] * (1 - a) + g * a;
        img.data[k + 2] = BG[2] * (1 - a) + b * a;
        img.data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, resolution]);

  return <canvas ref={ref} className="heatmap" />;
}
