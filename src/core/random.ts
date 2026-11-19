/**
 * Random numbers.
 *
 * The whole project calls `random()` instead of `Math.random()`, so tests can
 * call `seed(42)` and get the same data every run.
 */

let generator: () => number = Math.random;

/** Uniform in [0, 1). */
export const random = (): number => generator();

/** Fixes the seed using mulberry32, a small and fast generator. */
export function seed(s: number): void {
  let a = s >>> 0;
  generator = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Goes back to Math.random. */
export function unseed(): void {
  generator = Math.random;
}

/** Uniform in [a, b). */
export const uniform = (a: number, b: number): number => a + (b - a) * random();

/** Integer in [0, n). */
export const randInt = (n: number): number => Math.floor(random() * n);

/**
 * Standard normal N(0, 1) by the Box-Muller transform:
 * if U1 and U2 are uniform, sqrt(-2 ln U1) · cos(2π U2) is normal.
 */
export function randn(): number {
  let u = 0;
  while (u === 0) u = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/** Fisher-Yates shuffle, in place, O(n). */
export function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = randInt(i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** [0, 1, ..., n-1] */
export const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i);
