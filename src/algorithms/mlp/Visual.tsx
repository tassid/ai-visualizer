import { useMemo } from 'react';
import { binaryColor, classColor } from '../../core/colors';
import type { VisualProps } from '../../core/types';
import { NetworkDiagram, type NetworkSnapshot } from '../../components/charts/NetworkDiagram';
import { Plot } from '../../components/plot/Plot';
import { Points } from '../../components/plot/marks';
import type { MlpState } from './model';

export function MlpVisual({ state: s, data, version }: VisualProps<MlpState>) {
  return (
    <Plot interactive={false} heatmap={{ version, resolution: 64, color: (x, y) => binaryColor(s.net.predict(x, y)) }}>
      <Points data={data} color={(p) => classColor(p.c)} />
    </Plot>
  );
}

const GRID = 16;

/** Painel inferior: a rede com o mapa de ativação de cada neurônio. */
export function MlpNetwork({ state: s, version }: VisualProps<MlpState>) {
  const snapshot = useMemo<NetworkSnapshot>(() => {
    const net = s.net;
    const maps = net.sizes.map((n) => Array.from({ length: n }, () => new Float64Array(GRID * GRID)));
    // Uma única passada direta por ponto da grade preenche os mapas de TODOS os neurônios.
    for (let gj = 0; gj < GRID; gj++) {
      for (let gi = 0; gi < GRID; gi++) {
        const acts = net.forward(-1 + ((gi + 0.5) / GRID) * 2, 1 - ((gj + 0.5) / GRID) * 2);
        acts.forEach((layer, l) => layer.forEach((v, k) => { maps[l][k][gj * GRID + gi] = v; }));
      }
    }
    const L = net.sizes.length - 1;
    const ranges = net.sizes.map((_, l): [number, number] => (l === 0 ? [-1, 1] : l === L ? [0, 1] : net.act.range));
    return { sizes: net.sizes, weights: net.params.W, maps, ranges, grid: GRID };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, s]);

  return <NetworkDiagram net={snapshot} width={560} height={290} version={version} />;
}
