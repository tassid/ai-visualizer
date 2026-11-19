/**
 * Tabular Q-learning (Watkins, 1989) on a grid world.
 *
 * The agent does not know the map. It only sees the cell it is in, picks a
 * move, and gets a reward: +1 at the goal, -1 in a pit, a small cost per
 * step otherwise. Q(s, a) estimates the total discounted reward of taking
 * move a in cell s and acting well afterward. After every move:
 *   Q(s, a) ← Q(s, a) + α [ r + γ maxₐ' Q(s', a') - Q(s, a) ]
 * The bracket is the temporal-difference error: how much better or worse the
 * step went than the table predicted.
 *
 * Moves are ε-greedy: usually the best known move, sometimes a random one,
 * so the agent keeps exploring.
 */
import { randInt, random } from '../../core/random';
import type { Point, StepInfo, Trainer } from '../../core/types';

export interface QParams {
  alpha: number;
  gamma: number;
  epsilon: number;
  episodes: number;
}

export const N = 8;
export const ACTIONS: [number, number][] = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left (row, col)
export const ACTION_NAMES = ['up', 'right', 'down', 'left'];
export const START = (N - 1) * N;            // bottom-left
export const GOAL = N - 1;                   // top-right
export const PITS = [2 * N + 5, 5 * N + 3, 3 * N + 2];
const DEFAULT_WALLS = [1 * N + 1, 1 * N + 2, 1 * N + 3, 3 * N + 5, 4 * N + 5, 5 * N + 5, 6 * N + 1, 6 * N + 2];

export interface QState {
  Q: Float64Array;
  walls: Uint8Array;
  agent: number;
  episode: number;
  step: number;
  visits: Uint32Array;
  last: { s: number; a: number; r: number; s2: number; before: number; after: number; target: number } | null;
  /** Total reward of each finished episode. */
  history: number[];
  /** Steps taken in each finished episode. */
  lengths: number[];
}

export const reward = (s: number) => (s === GOAL ? 1 : PITS.includes(s) ? -1 : -0.02);
export const terminal = (s: number) => s === GOAL || PITS.includes(s);
export const maxQ = (Q: Float64Array, s: number) => Math.max(Q[s * 4], Q[s * 4 + 1], Q[s * 4 + 2], Q[s * 4 + 3]);

export function greedy(Q: Float64Array, s: number): number {
  const best = maxQ(Q, s);
  const ties = [0, 1, 2, 3].filter((a) => Q[s * 4 + a] === best);
  return ties[randInt(ties.length)];
}

export function move(walls: Uint8Array, s: number, a: number): number {
  const r = Math.floor(s / N) + ACTIONS[a][0], c = (s % N) + ACTIONS[a][1];
  if (r < 0 || r >= N || c < 0 || c >= N) return s;
  const s2 = r * N + c;
  return walls[s2] ? s : s2;
}

export function* train(_data: Point[], P: QParams, st: QState): Generator<StepInfo, StepInfo> {
  for (let ep = 1; ep <= P.episodes; ep++) {                            // line 0
    let s = START, total = 0;                                           // line 1
    st.agent = s;
    st.episode = ep;
    for (let t = 1; t <= 200; t++) {                                    // line 2
      const a = random() < P.epsilon ? randInt(4) : greedy(st.Q, s);    // line 3
      const s2 = move(st.walls, s, a);                                  // line 4
      const r = reward(s2);
      const target = r + (terminal(s2) ? 0 : P.gamma * maxQ(st.Q, s2)); // line 5
      const before = st.Q[s * 4 + a];
      st.Q[s * 4 + a] += P.alpha * (target - before);                   // line 6
      st.last = { s, a, r, s2, before, after: st.Q[s * 4 + a], target };
      st.visits[s2]++;
      total += r;
      st.agent = s2;
      st.step = t;
      s = s2;
      if (terminal(s2)) {
        yield { line: 6, message: `Episode ${ep} ends ${s2 === GOAL ? 'at the goal' : 'in a pit'} after ${t} moves, return ${total.toFixed(2)}.` };
        break;
      }
      yield { line: 6, message: `Episode ${ep}, move ${t}: ${ACTION_NAMES[a]}. TD error ${(target - before).toFixed(3)}.` };
    }
    st.history.push(total);
    st.lengths.push(st.step);
  }
  return { line: 7, message: `${P.episodes} episodes. The arrows are the greedy policy the table now encodes.` };
}

export function createQ(_data: Point[], P: QParams): Trainer<QState> {
  const walls = new Uint8Array(N * N);
  DEFAULT_WALLS.forEach((w) => (walls[w] = 1));
  const st: QState = { Q: new Float64Array(N * N * 4), walls, agent: START, episode: 0, step: 0, visits: new Uint32Array(N * N), last: null, history: [], lengths: [] };
  return { state: st, steps: train(_data, P, st) };
}
