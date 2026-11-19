/**
 * Área de desenho principal: um canvas (fundo de decisão) sob um SVG (pontos,
 * retas e marcadores animados com Framer Motion).
 *
 * Por que misturar? O fundo tem milhares de pixels que mudam a cada quadro,
 * algo que o canvas desenha muito rápido. Já os elementos com significado
 * (retas, centroides, vizinhos) ficam no SVG, onde o Framer Motion consegue
 * interpolar posições suavemente entre um passo e outro.
 *
 * As escalas vêm do D3 (d3-scale): convertem coordenadas do mundo [-1, 1]
 * em pixels do SVG [0, 560]. Note que o eixo y é invertido, pois no SVG o
 * y cresce para baixo.
 */
import { scaleLinear, type ScaleLinear } from 'd3-scale';
import { createContext, useContext, useMemo, useRef, type ReactNode } from 'react';
import type { RGBA } from '../../core/colors';
import { Heatmap } from './Heatmap';

export const PLOT_SIZE = 560;

export interface PlotScales {
  x: ScaleLinear<number, number>;
  y: ScaleLinear<number, number>;
  size: number;
}

const makeScales = (size: number): PlotScales => ({
  x: scaleLinear().domain([-1, 1]).range([0, size]),
  y: scaleLinear().domain([-1, 1]).range([size, 0]),
  size,
});

const PlotContext = createContext<PlotScales>(makeScales(PLOT_SIZE));
export const usePlot = () => useContext(PlotContext);

export interface HeatmapSpec {
  color: (x: number, y: number) => RGBA;
  version: number;
  resolution?: number;
}

interface PlotProps {
  heatmap?: HeatmapSpec | null;
  onPick?: (x: number, y: number) => void;
  onHover?: (p: { x: number; y: number } | null) => void;
  children?: ReactNode;
  interactive?: boolean;
}

export function Plot({ heatmap, onPick, onHover, children, interactive = true }: PlotProps) {
  const scales = useMemo(() => makeScales(PLOT_SIZE), []);
  const ref = useRef<SVGSVGElement>(null);

  const toWorld = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * 2 - 1, y: 1 - ((e.clientY - r.top) / r.height) * 2 };
  };

  return (
    <PlotContext.Provider value={scales}>
      <div className={'plot' + (interactive ? ' interactive' : '')}>
        {heatmap && <Heatmap {...heatmap} />}
        <svg
          ref={ref}
          viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`}
          onClick={interactive && onPick ? (e) => { const p = toWorld(e); onPick(p.x, p.y); } : undefined}
          onMouseMove={interactive && onHover ? (e) => onHover(toWorld(e)) : undefined}
          onMouseLeave={interactive && onHover ? () => onHover(null) : undefined}
        >
          <Grid />
          {children}
        </svg>
      </div>
    </PlotContext.Provider>
  );
}

/** Grade de fundo a cada 0,25 e eixos em x = 0 e y = 0. */
function Grid() {
  const { x, y, size } = usePlot();
  const ticks = x.ticks(8);
  return (
    <g className="grid" pointerEvents="none">
      {ticks.map((t) => (
        <g key={t} className={t === 0 ? 'axis' : undefined}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={size} />
          <line y1={y(t)} y2={y(t)} x1={0} x2={size} />
        </g>
      ))}
    </g>
  );
}
