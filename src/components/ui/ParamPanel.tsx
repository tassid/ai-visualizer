import type { ParamSpec, Params, ParamValue } from '../../core/types';

/** Gera os controles a partir da lista declarativa `params` de cada algoritmo. */
export function ParamPanel({ specs, values, onChange }: { specs: ParamSpec[]; values: Params; onChange(key: string, v: ParamValue): void }) {
  if (!specs.length) return null;
  return (
    <div className="box">
      <div className="box-title">Parâmetros</div>
      {specs.map((p) => (
        <div className="field" key={p.key}>
          <label title={p.help}>{p.label} <span>{String(values[p.key])}</span></label>
          {p.type === 'range' ? (
            <input type="range" min={p.min} max={p.max} step={p.step} value={Number(values[p.key])} onChange={(e) => onChange(p.key, +e.target.value)} />
          ) : (
            <select
              value={String(values[p.key])}
              onChange={(e) => onChange(p.key, typeof p.value === 'number' ? +e.target.value : e.target.value)}
            >
              {p.options.map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}
            </select>
          )}
          {p.help && <div className="help">{p.help}</div>}
        </div>
      ))}
    </div>
  );
}
