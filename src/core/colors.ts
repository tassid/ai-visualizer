/**
 * Palette taken from Manim, the animation engine behind 3Blue1Brown.
 * Class A is red, class B is blue, and yellow marks whatever the model is
 * doing right now (the decision line, the point being updated).
 */

export type RGBA = [number, number, number, number];

export const COLORS = {
  bg: '#0c0c0e',
  blue: '#58C4DD',
  red: '#FC6255',
  yellow: '#FFFF00',
  green: '#83C167',
  teal: '#5CD0B3',
  gold: '#F0AC5F',
  purple: '#9A72AC',
  maroon: '#C55F73',
  grey: '#888888',
  greyDark: '#3d3d40',
  text: '#ECECEC',
  muted: '#9a9a9a',
  grid: '#29ABCA',
};

export const CLUSTER_PALETTE = [COLORS.blue, COLORS.yellow, COLORS.green, COLORS.red, COLORS.purple, COLORS.gold, COLORS.teal, COLORS.maroon];

export const hexToRgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [n >> 16, (n >> 8) & 255, n & 255];
};

const A = hexToRgb(COLORS.red);
const B = hexToRgb(COLORS.blue);
const MID: [number, number, number] = [34, 34, 38];

export const classColor = (c: number) => (c ? COLORS.blue : COLORS.red);

/**
 * Background tint for a probability p of class B.
 * Opacity grows with confidence |p - 0.5|, so uncertain regions stay dark.
 */
export function binaryColor(p: number): RGBA {
  const t = Math.max(0, Math.min(1, p));
  const a = 0.05 + 0.22 * Math.abs(t - 0.5) * 2;
  return [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t, a];
}

/** Opaque diverging scale: red, dark grey in the middle, blue. */
export function divergingColor(p: number): RGBA {
  const t = Math.max(0, Math.min(1, p));
  const [from, to, s] = t < 0.5 ? [A, MID, t * 2] : [MID, B, (t - 0.5) * 2];
  return [from[0] + (to[0] - from[0]) * s, from[1] + (to[1] - from[1]) * s, from[2] + (to[2] - from[2]) * s, 1];
}

export const rgbCss = ([r, g, b]: RGBA | [number, number, number]) => `rgb(${r | 0},${g | 0},${b | 0})`;
