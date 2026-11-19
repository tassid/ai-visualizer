/**
 * App root: header and page switching with a short cross-fade.
 * AnimatePresence keeps the old page on screen until its exit finishes.
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
          <a className="brand" href={linkTo('')}>AI Visualizer</a>
          <nav>
            <a href="https://github.com/tassid/ai-visualizer">Source</a>
          </nav>
        </div>
      </header>
      <AnimatePresence mode="wait">
        <motion.main
          key={algo ? algo.id : 'home'}
          className="wrap page"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          onAnimationStart={() => window.scrollTo({ top: 0 })}
        >
          {algo ? <AlgorithmPage def={algo} /> : <Home />}
        </motion.main>
      </AnimatePresence>
    </>
  );
}
