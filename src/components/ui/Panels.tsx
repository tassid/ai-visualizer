/** Painéis de informação: métricas, pseudocódigo, complexidade e mensagem do passo. */
import { motion } from 'framer-motion';
import type { AlgorithmDocs, Metric, StepInfo } from '../../core/types';

export function MetricsPanel({ metrics, status }: { metrics: Metric[]; status: 'pronto' | 'executando' | 'pausado' | 'concluído' }) {
  return (
    <div className="box">
      <span className={`status status-${status}`}>{status}</span>
      <div className="box-title">Métricas</div>
      <dl className="metrics">
        {metrics.map((m) => (
          <div key={m.label} className="metric">
            <dt>{m.label}</dt>
            <dd>{m.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/**
 * Pseudocódigo com a linha atual destacada.
 * O destaque usa `layoutId`: o Framer Motion percebe que o mesmo elemento
 * mudou de lugar e anima o deslocamento entre as linhas.
 */
export function Pseudocode({ lines, active }: { lines: string[]; active: number | null }) {
  return (
    <pre className="pseudo">
      {lines.map((l, i) => (
        <div key={i} className={'pline' + (i === active ? ' active' : '')}>
          {i === active && <motion.div layoutId="pseudo-highlight" className="pline-bg" transition={{ type: 'spring', stiffness: 500, damping: 40 }} />}
          <span className="lno">{i + 1}</span>
          <span className="ltext">{l}</span>
        </div>
      ))}
    </pre>
  );
}

export function StepMessage({ step, done }: { step: StepInfo | null; done: boolean }) {
  const text = step?.message ?? 'Aperte play ou dê um passo para começar.';
  return (
    <div className={'step-msg' + (done ? ' done' : '')}>
      {/* A chave é a linha do pseudocódigo, não o texto: em velocidade alta o texto
          muda a cada quadro e uma animação por mensagem nunca terminaria. */}
      <motion.span key={step?.line ?? -1} initial={{ opacity: 0.3, x: -4 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.2 }}>
        {text}
      </motion.span>
    </div>
  );
}

export function ComplexityTable({ docs }: { docs: AlgorithmDocs }) {
  return (
    <>
      <table className="cx">
        <tbody>
          {docs.complexity.map(([k, v]) => (
            <tr key={k}><td>{k}</td><td>{v}</td></tr>
          ))}
        </tbody>
      </table>
      {docs.complexityNote && <p className="small">{docs.complexityNote}</p>}
    </>
  );
}
