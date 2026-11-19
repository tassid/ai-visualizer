# Como adicionar um algoritmo

Exemplo: um classificador de centroide mais próximo (cada classe vira a média dos seus pontos).

## 1. Crie a pasta

```
src/algorithms/nearest-centroid/
├── model.ts
├── model.test.ts
└── index.tsx
```

## 2. Escreva o modelo como generator (`model.ts`)

```ts
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface NcState {
  sums: [number, number, number][]; // [soma x, soma y, contagem] por classe
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
    yield { line: 2, message: `Ponto ${i + 1} soma na média da classe ${p.c ? 'B' : 'A'}.` };
  }
  return { line: 3, message: 'Centroides prontos.' };
}

export function createNc(data: Point[]): Trainer<NcState> {
  const state: NcState = { sums: [[0, 0, 0], [0, 0, 0]], current: -1 };
  return { state, steps: train(data, null, state) };
}
```

Regras:

1. Todo o estado que o desenho precisa fica em `state`.
2. Cada `yield` informa a linha do pseudocódigo (índice começando em 0) e uma mensagem curta.
3. Use `random()`, `randn()` e `shuffle()` de `core/random.ts`, nunca `Math.random()`, para que os testes sejam reproduzíveis.

## 3. Teste uma propriedade (`model.test.ts`)

```ts
import { describe, expect, it } from 'vitest';
import { runToEnd } from '../../engine/useTrainer';
import { centroid, createNc } from './model';

describe('centroide mais próximo', () => {
  it('o centroide é a média da classe', () => {
    const t = createNc([{ x: 0, y: 0, c: 1 }, { x: 1, y: 1, c: 1 }]);
    runToEnd(t);
    expect(centroid(t.state, 1)).toEqual({ x: 0.5, y: 0.5 });
  });
});
```

## 4. Desenhe e descreva (`index.tsx`)

Use `Plot`, `Points`, `Ring` e as outras marcas de `components/plot/`. Veja `logistic-regression/index.tsx` como modelo curto. Preencha `docs` com descrição, passos, complexidade, pseudocódigo e exercícios.

## 5. Registre

Adicione a definição em `src/algorithms/index.ts`. Pronto: o card aparece na página inicial e a página do algoritmo é gerada automaticamente.

## 6. Verifique

```bash
npm test
npm run build
```
