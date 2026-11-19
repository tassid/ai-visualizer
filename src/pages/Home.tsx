import { useEffect, useMemo, useState } from 'react';
import { ALGORITHMS, GROUPS, findAlgorithm } from '../algorithms';
import { DATASETS } from '../core/datasets';
import { defaultParams, type AlgorithmDef } from '../core/types';
import { linkTo } from '../engine/useHashRoute';
import { runToEnd } from '../engine/useTrainer';
import { Tex } from '../components/ui/Tex';

/** Runs the algorithm without animation and draws the result small. */
function Thumbnail({ def }: { def: AlgorithmDef }) {
  const snap = useMemo(() => {
    const data = DATASETS[def.defaultDataset].generate();
    const params = defaultParams(def.params);
    const trainer = def.create(data, params);
    const { steps } = runToEnd(trainer, def.thumbnailSteps ?? 500);
    return { data, params, state: trainer.state, steps };
  }, [def]);
  const { Visual } = def;
  return (
    <div className="thumb">
      <Visual state={snap.state} data={snap.data} params={snap.params} step={null} hover={null} done version={snap.steps} compact />
    </div>
  );
}

/** A network training live on the spiral. */
function HeroDemo() {
  const def = findAlgorithm('mlp')!;
  const [model] = useState(() => {
    const data = DATASETS.spiral.generate();
    const params = { ...defaultParams(def.params), arch: '16-16', epochs: 600 };
    return { data, params, trainer: def.create(data, params) };
  });
  const [version, setVersion] = useState(0);
  useEffect(() => {
    let raf = 0, alive = true, epochs = 0;
    const tick = () => {
      if (!alive) return;
      for (let i = 0; i < 4 && epochs < 600; i++, epochs++) model.trainer.steps.next();
      setVersion((v) => v + 1);
      if (epochs < 600) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [model]);
  const { Visual } = def;
  return (
    <a className="hero-demo" href={linkTo('mlp')} aria-label="Open the multilayer perceptron">
      <Visual state={model.trainer.state} data={model.data} params={model.params} step={null} hover={null} done={false} version={version} compact />
    </a>
  );
}

export function Home() {
  return (
    <>
      <section className="hero">
        <div>
          <h1>Machine learning algorithms, one step at a time</h1>
          <p>
            A learning algorithm finds patterns in data instead of following rules written by hand. Each one makes its own assumptions: some draw lines, some bend the plane with layers of neurons, some just remember the examples they have seen.
          </p>
          <p>
            Every model here runs in your browser, written from scratch. Press play to watch it train, step through single updates, drag the data around, or grab the model itself and move it.
          </p>
        </div>
        <HeroDemo />
      </section>

      {GROUPS.map((g) => (
        <section key={g} className="group">
          <h2>{g}</h2>
          <div className="cards">
            {ALGORITHMS.filter((a) => a.group === g).map((a) => (
              <a key={a.id} className="card" href={linkTo(a.id)}>
                <Thumbnail def={a} />
                <span className="card-name">{a.name}</span>
              </a>
            ))}
          </div>
        </section>
      ))}

      <section className="about">
        <h2>Complexity</h2>
        <p>
          Algorithms are compared by how their time and memory grow with the input, written in <a href="https://www.programiz.com/dsa/asymptotic-notations">asymptotic notation</a>: <Tex>O</Tex>, <Tex>\Theta</Tex> and <Tex>\Omega</Tex> are the upper, tight and lower bounds. Here <Tex>n</Tex> is the number of examples and <Tex>d</Tex> the number of features.
        </p>
        <p>
          Learning algorithms have two costs, training and prediction, and they often trade one for the other. k-nearest neighbors trains in <Tex>O(1)</Tex> because it only stores the data, then pays <Tex>O(n\,d)</Tex> for every prediction. A neural network is the opposite.
        </p>
      </section>

      <footer>
        <p>
          Built by Tassiane Anzolin. Inspired by <a href="https://github.com/Myphz/sortvisualizer">Sort Visualizer</a> and the visual style of <a href="https://www.3blue1brown.com">3Blue1Brown</a>. Animations use Framer Motion, charts use D3, math uses KaTeX.
        </p>
      </footer>
    </>
  );
}
