import { motion } from 'framer-motion';
import { SPEEDS } from '../../engine/speeds';
import type { TrainerController } from '../../engine/useTrainer';

/** Play/pausa, passo único, reinício e velocidade. */
export function Controls({ t }: { t: TrainerController<unknown> }) {
  const label = t.playing ? '❚❚ Pausar' : t.done ? '↻ De novo' : t.stepsTaken ? '▶ Continuar' : '▶ Iniciar';
  return (
    <div className="box">
      <div className="btns">
        <motion.button whileTap={{ scale: 0.95 }} className="primary" onClick={t.toggle}>{label}</motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={t.stepOnce} disabled={t.done}>Passo</motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={t.reset}>Reiniciar</motion.button>
      </div>
      <div className="field">
        <label>Velocidade <span>{SPEEDS[t.speedIndex]} passos/s</span></label>
        <input type="range" min={0} max={SPEEDS.length - 1} step={1} value={t.speedIndex} onChange={(e) => t.setSpeedIndex(+e.target.value)} />
      </div>
    </div>
  );
}
