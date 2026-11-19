/** Paleta e funções de cor compartilhadas entre canvas e SVG. */

export type RGBA = [number, number, number, number];

export const COLORS = {
  bg: '#0e1118',
  classA: '#ff6b6b',
  classB: '#4ecdc4',
  accent: '#ffd166',
  violet: '#a78bfa',
  blue: '#60a5fa',
  neutral: '#5b6475',
  noise: '#2a3040',
  text: '#e6e9ef',
  muted: '#8a93a6',
};

export const CLUSTER_PALETTE = ['#ffd166', '#4ecdc4', '#ff6b6b', '#a78bfa', '#60a5fa', '#f472b6', '#34d399', '#fb923c', '#e879f9', '#facc15'];

export const hexToRgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [n >> 16, (n >> 8) & 255, n & 255];
};

const A = hexToRgb(COLORS.classA);
const B = hexToRgb(COLORS.classB);
const MID: [number, number, number] = [38, 43, 56];

export const classColor = (c: number) => (c ? COLORS.classB : COLORS.classA);

/**
 * Cor do fundo de decisão para uma probabilidade p da classe B.
 * A opacidade cresce com a confiança |p - 0,5|, então regiões incertas ficam escuras.
 */
export function binaryColor(p: number): RGBA {
  const t = Math.max(0, Math.min(1, p));
  const a = 0.1 + 0.42 * Math.abs(t - 0.5) * 2;
  return [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t, a];
}

/** Escala divergente opaca: vermelho, cinza no meio, verde-água. */
export function divergingColor(p: number): RGBA {
  const t = Math.max(0, Math.min(1, p));
  const [from, to, s] = t < 0.5 ? [A, MID, t * 2] : [MID, B, (t - 0.5) * 2];
  return [from[0] + (to[0] - from[0]) * s, from[1] + (to[1] - from[1]) * s, from[2] + (to[2] - from[2]) * s, 1];
}

export const rgba = (hex: string, a: number) => {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
};
