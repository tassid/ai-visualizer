# Adding an algorithm

Example: a nearest centroid classifier (each class becomes the mean of its points).

## 1. Create the folder

```
src/algorithms/nearest-centroid/
├── model.ts
├── model.test.ts
└── index.tsx
```

## 2. Write the model as a generator (`model.ts`)

```ts
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface NcState {
  sums: [number, number, number][]; // [sum of x, sum of y, count] per class
  current: number;
}

export const centroid = (s: NcState, c: number) => {
  const [sx, sy, n] = s.sums[c];
  return n ? { x: sx / n, y: sy / n } : null;
};

export function* train(data: Point[], _p: unknown, s: NcState): Generator<StepInfo, StepInfo> {
  for (let i = 0; i < data.length; i++) {
    const p = data[i];
    s.current = i;
    s.sums[p.c][0] += p.x;
    s.sums[p.c][1] += p.y;
    s.sums[p.c][2]++;
    yield { line: 2, message: `Point ${i + 1} joins the mean of class ${p.c ? 'B' : 'A'}.` };
  }
  return { line: 3, message: 'Centroids ready.' };
}

export function createNc(data: Point[]): Trainer<NcState> {
  const state: NcState = { sums: [[0, 0, 0], [0, 0, 0]], current: -1 };
  return { state, steps: train(data, null, state) };
}
```

Rules:

1. Everything the drawing needs lives in `state`.
2. Each `yield` names a pseudocode line (index from 0) and a short message.
3. Use `random()`, `randn()` and `shuffle()` from `core/random.ts`, never `Math.random()`, so the tests are reproducible.

## 3. Test a property (`model.test.ts`)

```ts
import { describe, expect, it } from 'vitest';
import { runToEnd } from '../../engine/useTrainer';
import { centroid, createNc } from './model';

describe('nearest centroid', () => {
  it('the centroid is the class mean', () => {
    const t = createNc([{ x: 0, y: 0, c: 1 }, { x: 1, y: 1, c: 1 }]);
    runToEnd(t);
    expect(centroid(t.state, 1)).toEqual({ x: 0.5, y: 0.5 });
  });
});
```

## 4. Draw and describe it (`index.tsx`)

Use `Plot`, `Points`, `Ring`, `Handle` and the other marks in `components/plot/`. `logistic-regression/index.tsx` is a short model to copy. A `Handle` makes any part of the state draggable: give it a position and an `onDrag` that writes into the state.

Fill in `equation` with the formula in TeX and `docs` with the description, steps, complexity, pseudocode and exercises. Inline math in the text goes between `$...$`.

## 5. Register it

Add the definition to `src/algorithms/index.ts`. Its card appears on the home page and its page is built automatically.

## 6. Check

```bash
npm test
npm run build
```
