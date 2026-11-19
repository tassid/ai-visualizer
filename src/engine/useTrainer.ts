/**
 * O motor de animação.
 *
 * Ele guarda o generator do algoritmo e chama `.next()` no ritmo escolhido.
 * Como o estado do modelo é um objeto mutável, o React não percebe mudanças
 * sozinho; por isso existe `version`, que incrementa a cada quadro em que
 * algum passo rodou e força uma nova renderização.
 *
 * Fluxo de um quadro (requestAnimationFrame):
 *   1. acumula tempo × velocidade em `budget`
 *   2. executa floor(budget) passos, respeitando um limite de tempo
 *   3. incrementa `version` para redesenhar
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { AlgorithmDef, Params, Point, StepInfo, Trainer } from '../core/types';
import { FRAME_BUDGET_MS, SPEEDS } from './speeds';

export interface TrainerController<S> {
  state: S;
  step: StepInfo | null;
  stepsTaken: number;
  done: boolean;
  playing: boolean;
  version: number;
  speedIndex: number;
  setSpeedIndex(i: number): void;
  play(): void;
  pause(): void;
  toggle(): void;
  stepOnce(): void;
  reset(): void;
}

export function useTrainer<S>(def: AlgorithmDef<S>, data: Point[], params: Params): TrainerController<S> {
  const create = useCallback((): Trainer<S> => def.create(data.map((p) => ({ ...p })), params), [def, data, params]);

  const trainer = useRef<Trainer<S>>(null as unknown as Trainer<S>);
  const meta = useRef({ step: null as StepInfo | null, steps: 0, done: false });
  // Recria o treinador na mesma renderização em que dados ou parâmetros mudam,
  // para que o visual nunca combine dados novos com um estado antigo.
  const createdWith = useRef<unknown>(null);
  if (createdWith.current !== create) {
    trainer.current = create();
    meta.current = { step: null, steps: 0, done: false };
    createdWith.current = create;
  }

  const [version, setVersion] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIndex, setSpeedIndex] = useState(def.defaultSpeed);

  /** Executa até `n` passos. Retorna true se o generator terminou. */
  const advance = useCallback((n: number, deadline = Infinity) => {
    const m = meta.current;
    for (let i = 0; i < n && !m.done; i++) {
      const r = trainer.current.steps.next();
      if (r.done) {
        m.done = true;
        if (r.value) m.step = r.value;
      } else {
        m.step = r.value;
        m.steps++;
      }
      if (performance.now() > deadline) break;
    }
    return m.done;
  }, []);

  const reset = useCallback(() => {
    trainer.current = create();
    meta.current = { step: null, steps: 0, done: false };
    setPlaying(false);
    setVersion((v) => v + 1);
  }, [create]);

  // Ao trocar dados ou parâmetros, para a animação.
  useEffect(() => {
    setPlaying(false);
  }, [create]);

  // Laço de animação.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let budget = 0;
    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      budget += SPEEDS[speedIndex] * dt;
      const n = Math.floor(budget);
      budget -= n;
      if (n > 0) {
        const finished = advance(n, performance.now() + FRAME_BUDGET_MS);
        setVersion((v) => v + 1);
        if (finished) {
          setPlaying(false);
          return;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [playing, speedIndex, advance]);

  const play = useCallback(() => {
    if (meta.current.done) {
      trainer.current = create();
      meta.current = { step: null, steps: 0, done: false };
    }
    setPlaying(true);
  }, [create]);

  const pause = useCallback(() => setPlaying(false), []);

  const stepOnce = useCallback(() => {
    setPlaying(false);
    advance(1);
    setVersion((v) => v + 1);
  }, [advance]);

  return {
    state: trainer.current.state,
    step: meta.current.step,
    stepsTaken: meta.current.steps,
    done: meta.current.done,
    playing,
    version,
    speedIndex,
    setSpeedIndex,
    play,
    pause,
    toggle: () => (playing ? pause() : play()),
    stepOnce,
    reset,
  };
}

/**
 * Roda um algoritmo de forma síncrona, sem animação.
 * Usado nas miniaturas da página inicial e nos testes.
 */
export function runToEnd<S>(trainer: Trainer<S>, maxSteps = 100_000): { state: S; steps: number; last: StepInfo | null } {
  let last: StepInfo | null = null;
  let steps = 0;
  for (; steps < maxSteps; steps++) {
    const r = trainer.steps.next();
    if (r.done) {
      if (r.value) last = r.value;
      break;
    }
    last = r.value;
  }
  return { state: trainer.state, steps, last };
}
