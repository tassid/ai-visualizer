/** Controls next to the plot: playback, data, parameters, metrics. */
import { motion } from 'framer-motion';
import { COLORS } from '../../core/colors';
import { DATASETS, type DatasetId } from '../../core/datasets';
import type { Metric, ParamSpec, Params, ParamValue } from '../../core/types';
import { SPEEDS } from '../../engine/speeds';
import type { TrainerController } from '../../engine/useTrainer';

const press = { scale: 0.97 };

export function Playback({ t }: { t: TrainerController<unknown> }) {
  const label = t.playing ? 'Pause' : t.done ? 'Replay' : t.stepsTaken ? 'Resume' : 'Play';
  return (
    <section className="side-section">
      <div className="btns">
        <motion.button whileTap={press} className="primary" onClick={t.toggle}>{label}</motion.button>
        <motion.button whileTap={press} onClick={t.stepOnce} disabled={t.done}>Step</motion.button>
        <motion.button whileTap={press} onClick={t.reset}>Reset</motion.button>
      </div>
      <div className="field">
        <label htmlFor="speed">Speed <span>{SPEEDS[t.speedIndex]} steps/s</span></label>
        <input id="speed" type="range" min={0} max={SPEEDS.length - 1} step={1} value={t.speedIndex} onChange={(e) => t.setSpeedIndex(+e.target.value)} />
      </div>
    </section>
  );
}

interface DataProps {
  options: DatasetId[];
  value: DatasetId;
  onChange(id: DatasetId): void;
  onRegenerate(): void;
  onClear(): void;
  brush?: { value: number; onChange(c: number): void };
}

export function DataControls({ options, value, onChange, onRegenerate, onClear, brush }: DataProps) {
  return (
    <section className="side-section">
      <div className="field">
        <label htmlFor="dataset">Dataset</label>
        <div className="row">
          <select id="dataset" value={value} onChange={(e) => onChange(e.target.value as DatasetId)}>
            {options.map((id) => <option key={id} value={id}>{DATASETS[id].name}</option>)}
          </select>
          <button onClick={onRegenerate}>New sample</button>
          <button onClick={onClear}>Clear</button>
        </div>
      </div>
      {brush && (
        <div className="field">
          <label id="brush-label">New points</label>
          <div className="segmented" role="radiogroup" aria-labelledby="brush-label">
            {[0, 1].map((c) => (
              <button key={c} role="radio" aria-checked={brush.value === c} className={brush.value === c ? 'on' : ''} onClick={() => brush.onChange(c)}>
                <span className="dot" style={{ background: c ? COLORS.blue : COLORS.red }} />Class {c ? 'B' : 'A'}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export function ParamControls({ specs, values, onChange }: { specs: ParamSpec[]; values: Params; onChange(key: string, v: ParamValue): void }) {
  if (!specs.length) return null;
  return (
    <section className="side-section">
      {specs.map((p) => (
        <div className="field" key={p.key}>
          <label htmlFor={`p-${p.key}`}>{p.label} <span>{String(values[p.key])}</span></label>
          {p.type === 'range' ? (
            <input id={`p-${p.key}`} type="range" min={p.min} max={p.max} step={p.step} value={Number(values[p.key])} onChange={(e) => onChange(p.key, +e.target.value)} />
          ) : (
            <select id={`p-${p.key}`} value={String(values[p.key])} onChange={(e) => onChange(p.key, typeof p.value === 'number' ? +e.target.value : e.target.value)}>
              {p.options.map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}
            </select>
          )}
          {p.help && <div className="help">{p.help}</div>}
        </div>
      ))}
    </section>
  );
}

export function Metrics({ metrics }: { metrics: Metric[] }) {
  return (
    <section className="side-section">
      <dl className="metrics">
        {metrics.map((m) => (
          <div key={m.label} className="metric">
            <dt>{m.label}</dt>
            <dd>{m.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
