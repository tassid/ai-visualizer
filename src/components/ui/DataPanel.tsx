import { COLORS } from '../../core/colors';
import { DATASETS, type DatasetId } from '../../core/datasets';

interface Props {
  options: DatasetId[];
  value: DatasetId;
  onChange(id: DatasetId): void;
  onRegenerate(): void;
  onClear(): void;
  /** Só aparece em classificação: classe dos pontos adicionados com clique. */
  brush?: { value: number; onChange(c: number): void };
}

export function DataPanel({ options, value, onChange, onRegenerate, onClear, brush }: Props) {
  return (
    <div className="box">
      <div className="box-title">Dados</div>
      <div className="row">
        <select value={value} onChange={(e) => onChange(e.target.value as DatasetId)}>
          {options.map((id) => <option key={id} value={id}>{DATASETS[id].name}</option>)}
        </select>
        <button onClick={onRegenerate} title="Sortear novos pontos">Novos</button>
      </div>
      <p className="small">{DATASETS[value].description}</p>
      {brush && (
        <div className="row brush">
          {[0, 1].map((c) => (
            <button key={c} className={brush.value === c ? 'on' : ''} onClick={() => brush.onChange(c)}>
              <span className="dot" style={{ background: c ? COLORS.classB : COLORS.classA }} /> Classe {c ? 'B' : 'A'}
            </button>
          ))}
        </div>
      )}
      <button className="wide" onClick={onClear}>Limpar pontos</button>
    </div>
  );
}
