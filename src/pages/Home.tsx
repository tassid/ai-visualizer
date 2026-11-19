import { motion } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import { ALGORITHMS, GROUPS, findAlgorithm } from '../algorithms';
import { DATASETS } from '../core/datasets';
import { defaultParams, type AlgorithmDef } from '../core/types';
import { linkTo } from '../engine/useHashRoute';
import { runToEnd } from '../engine/useTrainer';

/** Roda o algoritmo sem animação e desenha o resultado em miniatura. */
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

/** Uma MLP treinando ao vivo na espiral, como capa do site. */
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
    <motion.div className="hero-demo" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
      <Visual state={model.trainer.state} data={model.data} params={model.params} step={null} hover={null} done={false} version={version} compact />
    </motion.div>
  );
}

const container = { hidden: {}, show: { transition: { staggerChildren: 0.05 } } };
const item = { hidden: { opacity: 0, y: 18 }, show: { opacity: 1, y: 0 } };

export function Home() {
  return (
    <>
      <section className="hero">
        <div>
          <div className="crumb">Visualizador de aprendizado de máquina</div>
          <h1>Veja os algoritmos de IA aprendendo, passo a passo.</h1>
          <p>
            Algoritmos de aprendizado de máquina são métodos que permitem a um computador aprender padrões a partir de dados, em vez de ter cada regra programada à mão. Eles estão por trás de recomendações, diagnósticos, visão computacional e modelos de linguagem.
          </p>
          <p>
            Existem centenas de métodos diferentes, cada um com suas próprias suposições sobre os dados. Alguns traçam retas, outros dobram o espaço com camadas de neurônios, outros apenas lembram dos exemplos vistos. Entender o que cada um faz por dentro fica muito mais fácil quando dá para ver.
          </p>
          <p><a href="#" onClick={(e) => { e.preventDefault(); document.getElementById('algoritmos')?.scrollIntoView({ behavior: 'smooth' }); }}>Escolha um algoritmo ↓</a></p>
        </div>
        <HeroDemo />
      </section>

      <h2>Tipos de aprendizado</h2>
      <div className="cats">
        <div className="cat"><b>Supervisionado: classificação</b><span>Os dados têm rótulos discretos (classe A ou B). O modelo aprende uma fronteira de decisão que separa as classes.</span></div>
        <div className="cat"><b>Supervisionado: regressão</b><span>Os rótulos são números contínuos. O modelo aprende uma função que prevê o valor a partir das entradas.</span></div>
        <div className="cat"><b>Não supervisionado</b><span>Não há rótulos. O algoritmo procura estrutura sozinho: grupos (agrupamento) ou direções importantes (redução de dimensionalidade).</span></div>
      </div>

      <h2>Complexidade</h2>
      <p>
        Assim como nos algoritmos de ordenação, os métodos de aprendizado são comparados pela complexidade de tempo e de espaço, escritas com <a href="https://www.programiz.com/dsa/asymptotic-notations">notação assintótica</a>: O, Θ e Ω representam, respectivamente, o limite superior, o limite justo e o limite inferior. As expressões usam <code>n</code> para o número de exemplos, <code>d</code> para o número de atributos e, quando necessário, parâmetros do modelo, como <code>k</code> vizinhos ou <code>P</code> pesos.
      </p>
      <p>
        Em aprendizado de máquina há ainda uma distinção importante: o custo de <b>treino</b> e o custo de <b>predição</b>. O KNN treina em O(1), porque só guarda os dados, mas paga O(n·d) a cada previsão. Uma rede neural é o oposto: treino caro, predição rápida.
      </p>

      <h2 id="algoritmos">Algoritmos</h2>
      {GROUPS.map((g) => (
        <section key={g}>
          <div className="group-title">{g}</div>
          <motion.div className="cards" variants={container} initial="hidden" animate="show">
            {ALGORITHMS.filter((a) => a.group === g).map((a) => (
              <motion.a key={a.id} className="card" href={linkTo(a.id)} variants={item} whileHover={{ y: -4 }}>
                <Thumbnail def={a} />
                <div className="t">{a.name}</div>
                <span className="tag">{a.tag}</span>
                <div className="d">{a.summary}</div>
              </motion.a>
            ))}
          </motion.div>
        </section>
      ))}

      <h2>Sobre</h2>
      <p>
        Inspirado no <a href="https://github.com/Myphz/sortvisualizer">Sort Visualizer</a>. Cada algoritmo foi escrito do zero em TypeScript, sem bibliotecas de ML, como um <i>generator</i>: cada <code>yield</code> é um passo real do treino e a linha correspondente do pseudocódigo acende durante a animação. As animações usam Framer Motion e os gráficos usam D3. O código foi pensado para ser lido: veja o <a href="https://github.com/tassid/ai-visualizer">repositório</a> e o guia de estudo.
      </p>
      <footer>
        <div><b>AI VISUALIZER</b>Tassiane Anzolin</div>
        <div>
          <b>Referências</b>
          <a href="https://github.com/Myphz/sortvisualizer">Sort Visualizer</a><br />
          <a href="https://playground.tensorflow.org">TensorFlow Playground</a><br />
          <a href="https://www.deeplearningbook.org">Deep Learning (Goodfellow, Bengio e Courville)</a><br />
          <a href="https://hastie.su.domains/ElemStatLearn/">The Elements of Statistical Learning</a>
        </div>
      </footer>
    </>
  );
}
