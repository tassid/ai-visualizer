/**
 * The animation engine.
 *
 * It holds the algorithm's generator and calls `.next()` at the chosen pace.
 * The model state is a mutable object, so React does not see changes on its
 * own; `version` increments every frame in which a step ran and forces a
 * new render.
 *
 * One frame (requestAnimationFrame):
 *   1. add elapsed time × speed to `budget`
 *   2. run floor(budget) steps, within a time limit
 *   3. bump `version` so the screen redraws
 *
 * Replay: when the data or the parameters change after a model has trained
 * (for example while you drag a point), the new model trains again as fast
 * as the frame budget allows, so the picture follows your hand.
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
  /** Call after editing the state by hand (dragging a handle) to redraw. */
  touch(): void;
}

export function useTrainer<S>(def: AlgorithmDef<S>, data: Point[], params: Params): TrainerController<S> {
  const create = useCallback((): Trainer<S> => def.create(data.map((p) => ({ ...p })), params), [def, data, params]);

  const trainer = useRef<Trainer<S>>(null as unknown as Trainer<S>);
  const meta = useRef({ step: null as StepInfo | null, steps: 0, done: false });
  const playingRef = useRef(false);
  const replay = useRef(false);

  // Rebuild the trainer in the same render where data or params change,
  // so the visual never pairs new data with an old state.
  const createdWith = useRef<unknown>(null);
  if (createdWith.current !== create) {
    const hadTrained = createdWith.current !== null && (meta.current.done || playingRef.current || replay.current);
    trainer.current = create();
    meta.current = { step: null, steps: 0, done: false };
    replay.current = hadTrained;
    createdWith.current = create;
  }

  const [version, setVersion] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIndex, setSpeedIndex] = useState(def.defaultSpeed);
  playingRef.current = playing;

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

  useEffect(() => {
    setPlaying(replay.current);
  }, [create]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let budget = 0;
    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      let n: number;
      if (replay.current) n = Infinity;
      else {
        budget += SPEEDS[speedIndex] * dt;
        n = Math.floor(budget);
        budget -= n;
      }
      if (n > 0) {
        const finished = advance(n, performance.now() + FRAME_BUDGET_MS);
        setVersion((v) => v + 1);
        if (finished) {
          replay.current = false;
          setPlaying(false);
          return;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [playing, speedIndex, advance]);

  const reset = useCallback(() => {
    trainer.current = create();
    meta.current = { step: null, steps: 0, done: false };
    replay.current = false;
    setPlaying(false);
    setVersion((v) => v + 1);
  }, [create]);

  const play = useCallback(() => {
    replay.current = false;
    if (meta.current.done) {
      trainer.current = create();
      meta.current = { step: null, steps: 0, done: false };
    }
    setPlaying(true);
  }, [create]);

  const pause = useCallback(() => {
    replay.current = false;
    setPlaying(false);
  }, []);

  const stepOnce = useCallback(() => {
    replay.current = false;
    setPlaying(false);
    advance(1);
    setVersion((v) => v + 1);
  }, [advance]);

  const touch = useCallback(() => setVersion((v) => v + 1), []);

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
    touch,
  };
}

/**
 * Runs an algorithm synchronously, without animation.
 * Used by the home page thumbnails and by the tests.
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
