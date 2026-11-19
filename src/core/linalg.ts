/**
 * Small linear algebra: 2x2 matrices for Gaussians and ellipses, and a
 * Cholesky solver for Gaussian processes. Written out by hand on purpose.
 */

/** Symmetric 2x2 matrix [[a, b], [b, c]] stored as [a, b, c]. */
export type Sym2 = [number, number, number];

export const det2 = ([a, b, c]: Sym2) => a * c - b * b;

export function inv2(S: Sym2): Sym2 {
  const d = det2(S) || 1e-12;
  return [S[2] / d, -S[1] / d, S[0] / d];
}

/** Quadratic form (x - μ)ᵀ S⁻¹ (x - μ), the squared Mahalanobis distance. */
export function mahalanobis2(x: number, y: number, mu: [number, number], S: Sym2): number {
  const [ia, ib, ic] = inv2(S);
  const dx = x - mu[0], dy = y - mu[1];
  return ia * dx * dx + 2 * ib * dx * dy + ic * dy * dy;
}

/** Density of a 2D Gaussian N(μ, S) at (x, y). */
export function gauss2(x: number, y: number, mu: [number, number], S: Sym2): number {
  return Math.exp(-0.5 * mahalanobis2(x, y, mu, S)) / (2 * Math.PI * Math.sqrt(Math.max(det2(S), 1e-12)));
}

/**
 * The ellipse of one standard deviation: its axes are the eigenvectors of S
 * and its radii are the square roots of the eigenvalues.
 */
export function ellipseOf(S: Sym2): { r1: number; r2: number; angle: number } {
  const [a, b, c] = S;
  const tr = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  return { r1: Math.sqrt(Math.max(tr + d, 0)), r2: Math.sqrt(Math.max(tr - d, 0)), angle: 0.5 * Math.atan2(2 * b, a - c) };
}

/**
 * Cholesky factorization A = L Lᵀ for a symmetric positive definite matrix.
 * O(n³/3). The standard way to solve the linear systems of a Gaussian process.
 */
export function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
  }
  return L;
}

/** Solves L z = b (forward substitution). */
export function forwardSolve(L: number[][], b: number[]): number[] {
  const z = new Array<number>(b.length);
  for (let i = 0; i < b.length; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k];
    z[i] = s / L[i][i];
  }
  return z;
}

/** Solves Lᵀ x = z (back substitution). */
export function backSolve(L: number[][], z: number[]): number[] {
  const n = z.length, x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}
