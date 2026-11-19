import { binaryColor, classColor, COLORS } from '../../core/colors';
import { sigmoid } from '../../core/math';
import type { VisualProps } from '../../core/types';
import { Plot } from '../../components/plot/Plot';
import { Arrow, DecisionLine, Label, Points, Ring } from '../../components/plot/marks';
import { score, type PerceptronState } from './model';

export function PerceptronVisual({ state: s, data, version, compact }: VisualProps<PerceptronState>) {
  const [w1, w2] = s.w;
  const norm = Math.hypot(w1, w2) || 1;
  // ponto da reta mais próximo da origem; a seta w sai dele, perpendicular à reta
  const d = -s.b / norm;
  const foot: [number, number] = [(w1 / norm) * d, (w2 / norm) * d];
  const cur = s.current >= 0 ? data[s.current] : null;

  return (
    <Plot interactive={false} heatmap={{ version, color: (x, y) => binaryColor(sigmoid((16 * score(s, x, y)) / norm)) }}>
      <DecisionLine w1={w1} w2={w2} b={s.b} />
      {Math.abs(foot[0]) < 0.95 && Math.abs(foot[1]) < 0.95 && (
        <Arrow from={foot} to={[foot[0] + (w1 / norm) * 0.22, foot[1] + (w2 / norm) * 0.22]} />
      )}
      <Points data={data} color={(p) => classColor(p.c)} />
      {!compact && cur && (
        <>
          <Ring at={cur} color={s.updated ? COLORS.accent : '#fff'} width={s.updated ? 3 : 1.6} />
          {s.updated && <Label at={cur} color={COLORS.accent}>erro: atualiza w</Label>}
        </>
      )}
    </Plot>
  );
}
