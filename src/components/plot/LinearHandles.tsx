/**
 * Handles that let you set a linear model by hand.
 *
 * The line is w · x + b = 0. Two handles control it:
 *   base: the point of the line closest to the origin; dragging it slides
 *         the line (changes b) without turning it
 *   tip:  the end of the arrow w drawn from the base; dragging it turns the
 *         line and, for models where |w| matters, changes the steepness
 *
 * Dragging writes into the model state. If training is running, the next
 * gradient step starts from where you left the line.
 */
import { COLORS } from '../../core/colors';
import { Arrow, Handle } from './marks';

export interface LinearModel {
  w: [number, number];
  b: number;
}

interface Props {
  s: LinearModel;
  /**
   * 'direction': the arrow has a fixed length and only sets the direction
   * (the perceptron does not care about |w|).
   * 'vector': the arrow length is |w| · scale, so dragging also sets |w|.
   */
  mode: 'direction' | 'vector';
  scale?: number;
}

export function LinearHandles({ s, mode, scale = 0.06 }: Props) {
  const [w1, w2] = s.w;
  const norm = Math.hypot(w1, w2) || 1e-9;
  const d = -s.b / norm;
  const base: [number, number] = [(w1 / norm) * d, (w2 / norm) * d];
  const len = mode === 'direction' ? 0.24 : Math.max(0.05, Math.min(0.7, norm * scale));
  const tip: [number, number] = [base[0] + (w1 / norm) * len, base[1] + (w2 / norm) * len];
  const inView = Math.abs(base[0]) < 0.95 && Math.abs(base[1]) < 0.95;
  if (!inView) return null;

  return (
    <>
      <Arrow from={base} to={tip} color={COLORS.yellow} />
      <Handle
        title="Drag to slide the line"
        at={{ x: base[0], y: base[1] }}
        color={COLORS.yellow}
        onDrag={(px, py) => { s.b = -(w1 * px + w2 * py); }}
      />
      <Handle
        title="Drag to turn the line"
        at={{ x: tip[0], y: tip[1] }}
        visible={false}
        onDrag={(px, py) => {
          const vx = px - base[0], vy = py - base[1];
          const vlen = Math.hypot(vx, vy);
          if (vlen < 0.02) return;
          s.w = mode === 'direction' ? [(vx / vlen) * norm, (vy / vlen) * norm] : [vx / scale, vy / scale];
          s.b = -(s.w[0] * base[0] + s.w[1] * base[1]); // keep the line through the base
        }}
      />
    </>
  );
}
