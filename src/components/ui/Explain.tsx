/** The explanatory parts of an algorithm page: current step, live formula, pseudocode, complexity. */
import { motion } from 'framer-motion';
import type { AlgorithmDocs, StepInfo } from '../../core/types';
import { Tex } from './Tex';

/**
 * Pseudocode with the current line highlighted. The highlight uses
 * `layoutId`: Framer Motion sees the same element in a new place and slides
 * it between lines.
 */
export function Pseudocode({ lines, active }: { lines: string[]; active: number | null }) {
  return (
    <pre className="pseudo">
      {lines.map((l, i) => (
        <div key={i} className={'pline' + (i === active ? ' active' : '')}>
          {i === active && <motion.div layoutId="pseudo-highlight" className="pline-bg" transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }} />}
          <span className="lno">{i + 1}</span>
          <span className="ltext">{l}</span>
        </div>
      ))}
    </pre>
  );
}

export function StepMessage({ step, done }: { step: StepInfo | null; done: boolean }) {
  return (
    <p className={'step-msg' + (done ? ' done' : '')} aria-live="polite">
      {step?.message ?? 'Press play, or step through one update at a time.'}
    </p>
  );
}

export function Equation({ tex }: { tex: string | null }) {
  if (!tex) return null;
  return <div className="equation"><Tex display>{tex}</Tex></div>;
}

export function ComplexityTable({ docs }: { docs: AlgorithmDocs }) {
  return (
    <>
      <table className="cx">
        <tbody>
          {docs.complexity.map(([k, v]) => (
            <tr key={k}><td>{k}</td><td><Tex>{v}</Tex></td></tr>
          ))}
        </tbody>
      </table>
      {docs.complexityNote && <p className="small">{docs.complexityNote}</p>}
    </>
  );
}
