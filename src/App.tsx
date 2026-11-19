/**
 * Raiz da aplicação: cabeçalho e troca de páginas com transição animada.
 * AnimatePresence mantém a página antiga na tela até a animação de saída terminar.
 */
import { AnimatePresence, motion } from 'framer-motion';
import { findAlgorithm } from './algorithms';
import { linkTo, useHashRoute } from './engine/useHashRoute';
import { AlgorithmPage } from './pages/AlgorithmPage';
import { Home } from './pages/Home';

export function App() {
  const route = useHashRoute();
  const algo = findAlgorithm(route);

  return (
    <>
      <header className="topbar">
        <div className="wrap">
          <a className="brand" href={linkTo('')}><i />AI VISUALIZER</a>
          <nav>
            <a href={linkTo('')}>Início</a>
            <a href={linkTo('')} onClick={() => setTimeout(() => document.getElementById('algoritmos')?.scrollIntoView({ behavior: 'smooth' }), 350)}>Algoritmos</a>
            <a href="https://github.com/tassid/ai-visualizer">GitHub</a>
          </nav>
        </div>
      </header>
      <AnimatePresence mode="wait">
        <motion.main
          key={algo ? algo.id : 'home'}
          className="wrap page"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          onAnimationStart={() => window.scrollTo({ top: 0 })}
        >
          {algo ? <AlgorithmPage def={algo} /> : <Home />}
        </motion.main>
      </AnimatePresence>
    </>
  );
}
