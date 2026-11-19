/**
 * An algorithm's page, built entirely from its AlgorithmDef. No algorithm
 * needs a page of its own.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { DATASETS, type DatasetId } from '../core/datasets';
import { defaultParams, type AlgorithmDef, type Hover, type Params, type ParamValue, type Point } from '../core/types';
import { EditContext, type EditApi } from '../engine/editing';
import { linkTo } from '../engine/useHashRoute';
import { useTrainer } from '../engine/useTrainer';
import { LineChart } from '../components/charts/LineChart';
import { ComplexityTable, Equation, Pseudocode, StepMessage } from '../components/ui/Explain';
import { DataControls, Metrics, ParamControls, Playback } from '../components/ui/Sidebar';
import { Prose } from '../components/ui/Tex';

const FOLDERS: Record<string, string> = {};
const folderOf = (id: string) => FOLDERS[id] ?? id;

export function AlgorithmPage({ def }: { def: AlgorithmDef }) {
  const [datasetId, setDatasetId] = useState<DatasetId>(def.defaultDataset);
  const [data, setData] = useState<Point[]>(() => DATASETS[def.defaultDataset].generate());
  const [params, setParams] = useState<Params>(() => defaultParams(def.params));
  const [brush, setBrush] = useState(1);
  const [hover, setHover] = useState<Hover>(null);

  const t = useTrainer(def, data, params);
  const { Visual } = def;
  const Aux = def.aux?.Component;

  const regenerate = useCallback((id: DatasetId) => setData(DATASETS[id].generate()), []);
  const setParam = (key: string, v: ParamValue) => setParams((p) => ({ ...p, [key]: v }));

  const edit: EditApi = { data, setData, brush: def.kind === 'classification' ? brush : null, onHover: setHover, touch: t.touch };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'SELECT' || tag === 'BUTTON' || (tag === 'INPUT' && e.key !== ' ')) return;
      if (e.key === ' ') { e.preventDefault(); t.toggle(); }
      else if (e.key === 'ArrowRight') t.stepOnce();
      else if (e.key === 'r' || e.key === 'R') t.reset();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [t]);

  const metrics = useMemo(() => def.metrics(t.state, data, params), [def, t.state, data, params, t.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const equation = def.equation ? def.equation(t.state, data, params, hover) : null;
  const visualProps = { state: t.state, data, params, step: t.step, hover, done: t.done, version: t.version };

  return (
    <EditContext.Provider value={edit}>
      <a className="back" href={linkTo('')}>All algorithms</a>
      <h1>{def.name}</h1>

      <div className="layout">
        <div className="stage">
          <Visual {...visualProps} />
          <Equation tex={equation} />
          <StepMessage step={t.step} done={t.done} />
          <p className="hint">
            Click to add a point, drag to move one, shift-click to delete. {def.hint}
          </p>
          {Aux && (
            <figure className="aux">
              <figcaption>{def.aux!.title}</figcaption>
              <Aux {...visualProps} />
            </figure>
          )}
        </div>

        <aside className="side">
          <Playback t={t} />
          <DataControls
            options={def.datasets}
            value={datasetId}
            onChange={(id) => { setDatasetId(id); regenerate(id); }}
            onRegenerate={() => regenerate(datasetId)}
            onClear={() => setData([])}
            brush={def.kind === 'classification' ? { value: brush, onChange: setBrush } : undefined}
          />
          <ParamControls specs={def.params} values={params} onChange={setParam} />
          <Metrics metrics={metrics} />
          {def.history && (
            <figure className="side-section">
              <figcaption>{def.history.label}</figcaption>
              <LineChart values={def.history.get(t.state)} />
            </figure>
          )}
          <p className="keys"><kbd>Space</kbd> play or pause, <kbd>→</kbd> step, <kbd>R</kbd> reset</p>
        </aside>
      </div>

      <section className="info">
        <div>
          <h2>How it works</h2>
          <p><Prose text={def.docs.description} /></p>
          <ol>{def.docs.steps.map((s) => <li key={s}><Prose text={s} /></li>)}</ol>
          <h2>Exercises</h2>
          <ol className="exercises">{def.docs.exercises.map((s) => <li key={s}>{s}</li>)}</ol>
        </div>
        <div>
          <h2>Pseudocode</h2>
          <Pseudocode lines={def.docs.pseudocode} active={t.step?.line ?? null} />
          <h2>Complexity</h2>
          <ComplexityTable docs={def.docs} />
          <p className="small">
            Source: <a href={`https://github.com/tassid/ai-visualizer/tree/main/src/algorithms/${folderOf(def.id)}`}>src/algorithms/{folderOf(def.id)}</a>
          </p>
        </div>
      </section>
    </EditContext.Provider>
  );
}
