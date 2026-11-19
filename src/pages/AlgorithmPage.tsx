/**
 * Página de um algoritmo. Monta tudo a partir da definição (AlgorithmDef):
 * nenhum algoritmo precisa de uma página própria.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { DATASETS, type DatasetId } from '../core/datasets';
import { defaultParams, type AlgorithmDef, type Params, type ParamValue, type Point } from '../core/types';
import { useTrainer } from '../engine/useTrainer';
import { linkTo } from '../engine/useHashRoute';
import { Controls } from '../components/ui/Controls';
import { DataPanel } from '../components/ui/DataPanel';
import { ParamPanel } from '../components/ui/ParamPanel';
import { ComplexityTable, MetricsPanel, Pseudocode, StepMessage } from '../components/ui/Panels';
import { LineChart } from '../components/charts/LineChart';

export function AlgorithmPage({ def }: { def: AlgorithmDef }) {
  const [datasetId, setDatasetId] = useState<DatasetId>(def.defaultDataset);
  const [data, setData] = useState<Point[]>(() => DATASETS[def.defaultDataset].generate());
  const [params, setParams] = useState<Params>(() => defaultParams(def.params));
  const [brush, setBrush] = useState(1);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);

  const t = useTrainer(def, data, params);
  const { Visual } = def;
  const Aux = def.aux?.Component;

  const regenerate = useCallback((id: DatasetId) => setData(DATASETS[id].generate()), []);
  const setParam = (key: string, v: ParamValue) => setParams((p) => ({ ...p, [key]: v }));
  const addPoint = (x: number, y: number) => setData((d) => [...d, { x, y, c: def.kind === 'classification' ? brush : 0 }]);

  // Atalhos de teclado.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'SELECT' || (tag === 'INPUT' && e.key !== ' ')) return;
      if (e.key === ' ') { e.preventDefault(); t.toggle(); }
      else if (e.key === 'ArrowRight') t.stepOnce();
      else if (e.key === 'r' || e.key === 'R') t.reset();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [t]);

  const status = t.done ? 'concluído' : t.playing ? 'executando' : t.stepsTaken ? 'pausado' : 'pronto';
  const metrics = useMemo(() => def.metrics(t.state, data, params), [def, t.state, data, params, t.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const visualProps = { state: t.state, data, params, step: t.step, hover, done: t.done, version: t.version };

  return (
    <>
      <div className="crumb"><a href={linkTo('')}>Algoritmos</a> / {def.group}</div>
      <div className="titlebar">
        <h1>{def.name}</h1>
        <span className="pill">{def.tag}</span>
      </div>

      <div className="layout">
        <div className="stage">
          <div className="plot-host" onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            addPoint(((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2);
          }}
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setHover({ x: ((e.clientX - r.left) / r.width) * 2 - 1, y: 1 - ((e.clientY - r.top) / r.height) * 2 });
            }}
            onMouseLeave={() => setHover(null)}
          >
            <Visual {...visualProps} />
          </div>
          <StepMessage step={t.step} done={t.done} />
          <p className="hint">
            {def.kind === 'classification' ? 'Clique no gráfico para adicionar pontos da classe selecionada.' : 'Clique no gráfico para adicionar pontos.'} {def.hint}{' '}
            Atalhos: <kbd>espaço</kbd> play/pausa, <kbd>→</kbd> um passo, <kbd>R</kbd> reiniciar.
          </p>
          {Aux && (
            <motion.div className="aux" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
              <div className="box-title">{def.aux!.title}</div>
              <div className="aux-frame"><Aux {...visualProps} /></div>
            </motion.div>
          )}
        </div>

        <aside className="side">
          <Controls t={t} />
          <DataPanel
            options={def.datasets}
            value={datasetId}
            onChange={(id) => { setDatasetId(id); regenerate(id); }}
            onRegenerate={() => regenerate(datasetId)}
            onClear={() => setData([])}
            brush={def.kind === 'classification' ? { value: brush, onChange: setBrush } : undefined}
          />
          <ParamPanel specs={def.params} values={params} onChange={setParam} />
          <MetricsPanel metrics={metrics} status={status} />
          {def.history && (
            <div className="box">
              <div className="box-title">{def.history.label}</div>
              <LineChart values={def.history.get(t.state)} />
            </div>
          )}
        </aside>
      </div>

      <section className="info">
        <div className="box">
          <h3>Como funciona</h3>
          <p>{def.docs.description}</p>
          <ol>{def.docs.steps.map((s) => <li key={s}>{s}</li>)}</ol>
          <h3 className="mt">Para estudar</h3>
          <ol className="exercises">{def.docs.exercises.map((s) => <li key={s}>{s}</li>)}</ol>
        </div>
        <div className="box">
          <h3>Pseudocódigo</h3>
          <Pseudocode lines={def.docs.pseudocode} active={t.step?.line ?? null} />
          <h3 className="mt">Complexidade</h3>
          <ComplexityTable docs={def.docs} />
          <p className="small">
            Código: <a href={`https://github.com/tassid/ai-visualizer/tree/main/src/algorithms/${folderOf(def.id)}`}>src/algorithms/{folderOf(def.id)}</a>
          </p>
        </div>
      </section>
    </>
  );
}

const FOLDERS: Record<string, string> = {
  'regressao-linear': 'linear-regression',
  'regressao-logistica': 'logistic-regression',
  'arvore-decisao': 'decision-tree',
};
const folderOf = (id: string) => FOLDERS[id] ?? id;
