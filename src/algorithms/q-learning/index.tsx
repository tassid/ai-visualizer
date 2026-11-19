import { motion } from 'framer-motion';
import { COLORS, divergingColor, rgbCss } from '../../core/colors';
import { fmt, tex } from '../../core/math';
import type { AlgorithmDef, VisualProps } from '../../core/types';
import { Plot, usePlot } from '../../components/plot/Plot';
import { useEdit } from '../../engine/editing';
import { SUTTON, paper } from '../books';
import { ACTIONS, createQ, GOAL, maxQ, N, PITS, START, terminal, type QParams, type QState } from './model';

/** Grid world: cell color is the learned value max Q, arrows are the greedy policy. Click a cell to toggle a wall. */
function World({ s, compact }: { s: QState; compact?: boolean }) {
  const { size } = usePlot();
  const edit = useEdit();
  const cs = size / N;
  const cells = [];
  for (let k = 0; k < N * N; k++) {
    const r = Math.floor(k / N), c = k % N, x0 = c * cs, y0 = r * cs;
    const v = maxQ(s.Q, k);
    const wall = s.walls[k] === 1, isGoal = k === GOAL, isPit = PITS.includes(k);
    const fill = wall ? '#2a2a2e' : isGoal ? COLORS.blue : isPit ? COLORS.red : s.visits[k] || k === START ? rgbCss(divergingColor(0.5 + v / 2)) : 'transparent';
    cells.push(
      <g key={k}>
        <rect
          x={x0 + 1} y={y0 + 1} width={cs - 2} height={cs - 2} rx={3}
          fill={fill} fillOpacity={isGoal || isPit || wall ? 0.9 : 0.55}
          style={{ cursor: edit && !compact && !isGoal && !isPit && k !== START ? 'pointer' : undefined }}
          onClick={edit && !compact && !isGoal && !isPit && k !== START ? () => { s.walls[k] = s.walls[k] ? 0 : 1; edit.touch(); } : undefined}
        />
        {isGoal && <text x={x0 + cs / 2} y={y0 + cs / 2 + 7} textAnchor="middle" className="grid-label">+1</text>}
        {isPit && <text x={x0 + cs / 2} y={y0 + cs / 2 + 7} textAnchor="middle" className="grid-label">−1</text>}
        {!compact && !wall && !terminal(k) && s.visits[k] > 0 && <PolicyArrow s={s} k={k} cx={x0 + cs / 2} cy={y0 + cs / 2} len={cs * 0.28} />}
      </g>,
    );
  }
  const ar = Math.floor(s.agent / N), ac = s.agent % N;
  return (
    <g>
      {cells}
      <motion.circle initial={false} animate={{ cx: ac * cs + cs / 2, cy: ar * cs + cs / 2 }} transition={{ duration: 0.12 }} r={cs * 0.22} fill={COLORS.yellow} stroke="#000" strokeWidth={2} pointerEvents="none" />
    </g>
  );
}

function PolicyArrow({ s, k, cx, cy, len }: { s: QState; k: number; cx: number; cy: number; len: number }) {
  let a = 0;
  for (let i = 1; i < 4; i++) if (s.Q[k * 4 + i] > s.Q[k * 4 + a]) a = i;
  const [dr, dc] = ACTIONS[a];
  const tx = cx + dc * len, ty = cy + dr * len;
  const ang = Math.atan2(dr, dc), h = 6;
  return (
    <g pointerEvents="none" opacity={0.85}>
      <line x1={cx - dc * len * 0.6} y1={cy - dr * len * 0.6} x2={tx} y2={ty} stroke="#fff" strokeWidth={1.5} />
      <polygon points={`${tx + Math.cos(ang) * h},${ty + Math.sin(ang) * h} ${tx + Math.cos(ang + 2.4) * h},${ty + Math.sin(ang + 2.4) * h} ${tx + Math.cos(ang - 2.4) * h},${ty + Math.sin(ang - 2.4) * h}`} fill="#fff" />
    </g>
  );
}

function QVisual({ state: s, compact }: VisualProps<QState>) {
  return (
    <Plot>
      <World s={s} compact={compact} />
    </Plot>
  );
}

export const qLearning: AlgorithmDef<QState> = {
  id: 'q-learning',
  name: 'Q-learning',
  group: 'Reinforcement learning',
  kind: 'reinforcement',
  noData: true,
  datasets: [],
  defaultDataset: 'line',
  defaultSpeed: 8,
  thumbnailSteps: 3000,
  hint: 'Click any cell to add or remove a wall, even while the agent is learning, and watch the policy route around it.',
  params: [
    { key: 'alpha', label: 'Learning rate α', type: 'select', options: [0.1, 0.3, 0.5, 1], value: 0.5 },
    { key: 'gamma', label: 'Discount γ', type: 'select', options: [0.8, 0.9, 0.95, 0.99], value: 0.95 },
    { key: 'epsilon', label: 'Exploration ε', type: 'select', options: [0, 0.05, 0.1, 0.3], value: 0.1 },
    { key: 'episodes', label: 'Episodes', type: 'range', min: 50, max: 1000, step: 50, value: 300 },
  ],
  create: (data, params) => createQ(data, params as unknown as QParams),
  Visual: QVisual,
  history: { label: 'Return per episode', get: (s) => s.history },
  equation: (s, _d, params) => {
    const head = `Q(s,a) \\leftarrow Q(s,a) + \\alpha\\big[\\,r + \\gamma\\max_{a'}Q(s',a') - Q(s,a)\\,\\big]`;
    const l = s.last;
    if (!l) return head;
    return `${head} \\qquad ${tex(l.before, 3)} + ${params.alpha}\\,[\\,${tex(l.target, 3)} - ${tex(l.before, 3)}\\,] = {\\color{${COLORS.yellow}}${tex(l.after, 3)}}`;
  },
  metrics: (s) => [
    { label: 'Episode', value: s.episode },
    { label: 'Move', value: s.step },
    { label: 'Value at start', value: fmt(maxQ(s.Q, START), 3) },
    { label: 'Last episode length', value: s.lengths.length ? s.lengths[s.lengths.length - 1] : 'n/a' },
  ],
  docs: {
    description:
      'Reinforcement learning has no labeled examples: an agent acts, sees what happens and gets rewards. Here the yellow agent starts in the bottom left corner and must find the +1 goal while avoiding the −1 pits, paying a small cost per move. Q-learning fills in a table $Q(s, a)$, the expected future reward of each move from each cell. Each cell is colored by its best value, from red (bad) to blue (good), and the arrows show the move the table currently prefers. Value spreads backward from the goal, one episode at a time.',
    steps: [
      'Start with $Q(s,a) = 0$ everywhere.',
      'In cell $s$, pick the best known move, or a random one with probability $\\varepsilon$.',
      'Observe the reward $r$ and the next cell $s\'$.',
      'Nudge $Q(s,a)$ toward $r + \\gamma\\max_{a\'}Q(s\',a\')$.',
      'At the goal or a pit, start a new episode.',
    ],
    complexity: [['Per step', 'O(|A|)'], ['Memory', 'O(|S|\\,|A|)']],
    complexityNote: 'The table grows with the number of states, which is why large problems replace it with a neural network (deep Q-learning).',
    pseudocode: [
      'Q(s, a) ← 0 for all s, a',
      'for each episode: s ← start',
      '    repeat until s is terminal:',
      '        a ← ε-greedy action from Q(s, ·)',
      '        take a, observe r and s\'',
      '        target ← r + γ max_a\' Q(s\', a\')',
      '        Q(s, a) ← Q(s, a) + α (target - Q(s, a)) ;  s ← s\'',
      'policy(s) = argmax_a Q(s, a)',
    ],
    exercises: [
      'Set ε = 0. Does the agent still find the goal? Why is exploration needed?',
      'Compare γ = 0.8 and γ = 0.99. How far from the goal does value reach in each case?',
      'Wall off the shortest path halfway through training. How long until the policy adapts?',
      'Implement SARSA, which uses the action actually taken next instead of the max, and compare the paths near the pits.',
    ],
    theory: [
      'The setting is a Markov decision process: states $s$, actions $a$, transition probabilities $p(s\'\\mid s,a)$ and rewards $r$. The goal is a policy $\\pi$ that maximizes the expected discounted return $G_t = \\sum_{k\\ge0}\\gamma^k r_{t+k+1}$, with $0 \\le \\gamma < 1$ trading immediate against future reward. The optimal action-value function satisfies the Bellman optimality equation',
      '$$Q^*(s,a) = \\mathbb{E}\\big[\\,r + \\gamma\\max_{a\'}Q^*(s\',a\')\\ \\big|\\ s, a\\,\\big],$$',
      'and once $Q^*$ is known, acting greedily with respect to it is optimal. If the transition probabilities were known, value iteration would solve this equation by repeated substitution. Q-learning does it without a model: it replaces the expectation with the one transition it just experienced and moves part of the way, with step $\\alpha$, toward that sample. That is stochastic approximation of a fixed point.',
      'The bracketed quantity $\\delta = r + \\gamma\\max_{a\'}Q(s\',a\') - Q(s,a)$ is the temporal-difference error. Learning from it bootstraps: each estimate is updated from other estimates, which is why value first appears next to the goal and then propagates backward episode by episode.',
      'Q-learning is off-policy: it learns the value of the greedy policy while behaving $\\varepsilon$-greedily. Watkins and Dayan proved it converges to $Q^*$ with probability 1, provided every state-action pair keeps being visited and the step sizes satisfy $\\sum\\alpha_t = \\infty$, $\\sum\\alpha_t^2 < \\infty$. SARSA, its on-policy cousin, uses the action actually taken next; it learns safer paths that keep away from the pits, since it accounts for its own exploration.',
    ],
    references: [
      SUTTON('Chapter 3, Finite Markov decision processes, and Chapter 6, Temporal-difference learning (Section 6.5, Q-learning)'),
      paper('Watkins, Learning from delayed rewards, PhD thesis, University of Cambridge (1989)'),
      paper('Watkins and Dayan, Q-learning, Machine Learning 8 (1992)'),
      paper('Mnih et al., Human-level control through deep reinforcement learning, Nature 518 (2015)'),
    ],
  },
};
