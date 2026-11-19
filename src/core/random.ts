/**
 * Números aleatórios.
 *
 * Todo o projeto usa `random()` em vez de `Math.random()` diretamente.
 * Assim os testes podem chamar `seed(42)` e obter sempre os mesmos dados,
 * o que torna os resultados reproduzíveis (uma boa prática em ML).
 */

let generator: () => number = Math.random;

/** Número uniforme em [0, 1). */
export const random = (): number => generator();

/**
 * Fixa a semente usando o algoritmo mulberry32: um gerador pequeno,
 * rápido e com boa distribuição para visualizações e testes.
 */
export function seed(s: number): void {
  let a = s >>> 0;
  generator = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Volta a usar Math.random. */
export function unseed(): void {
  generator = Math.random;
}

/** Uniforme em [a, b). */
export const uniform = (a: number, b: number): number => a + (b - a) * random();

/** Inteiro uniforme em [0, n). */
export const randInt = (n: number): number => Math.floor(random() * n);

/**
 * Normal padrão N(0, 1) pela transformação de Box-Muller:
 * se U1 e U2 são uniformes, sqrt(-2 ln U1) · cos(2π U2) é normal.
 */
export function randn(): number {
  let u = 0;
  while (u === 0) u = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/** Embaralhamento de Fisher-Yates, in place, O(n). */
export function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = randInt(i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** [0, 1, ..., n-1] */
export const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i);
