/**
 * Diagrama de árvore usando o layout `tree()` do d3-hierarchy.
 *
 * O D3 calcula x e y de cada nó; o Framer Motion anima a entrada de cada
 * nó novo (escala de 0 a 1) e o deslocamento dos nós existentes quando a
 * árvore cresce e o layout se reorganiza.
 */
import { hierarchy, tree } from 'd3-hierarchy';
import { AnimatePresence, motion } from 'framer-motion';
import { divergingColor } from '../../core/colors';

export interface DiagramNode {
  id: number;
  prob: number;
  label?: string;
  pending?: boolean;
  children?: DiagramNode[];
}

interface Props {
  root: DiagramNode;
  currentId?: number;
  width: number;
  height: number;
}

const spring = { type: 'spring', stiffness: 220, damping: 26 } as const;

export function TreeDiagram({ root, currentId, width, height }: Props) {
  const h = hierarchy(root, (d) => d.children);
  const leaves = h.leaves().length;
  const layout = tree<DiagramNode>().size([width - 48, height - 54]).separation(() => 1);
  const laid = layout(h);
  const nodes = laid.descendants();
  const links = laid.links();
  const r = Math.max(3.5, Math.min(10, (width - 48) / Math.max(1, leaves) / 2.6));
  const showLabels = leaves <= 14;
  const col = (p: number) => {
    const [R, G, B] = divergingColor(p);
    return `rgb(${R | 0},${G | 0},${B | 0})`;
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg">
      <g transform="translate(24, 28)">
        <AnimatePresence>
          {links.map((l) => (
            <motion.line
              key={`${l.source.data.id}-${l.target.data.id}`}
              initial={{ opacity: 0, x1: l.source.x, y1: l.source.y, x2: l.source.x, y2: l.source.y }}
              animate={{ opacity: 1, x1: l.source.x, y1: l.source.y, x2: l.target.x, y2: l.target.y }}
              exit={{ opacity: 0 }}
              transition={spring}
              stroke="rgba(255,255,255,.3)"
              strokeWidth={1.2}
            />
          ))}
          {nodes.map((n) => (
            <motion.g
              key={n.data.id}
              initial={{ x: n.parent?.x ?? n.x, y: n.parent?.y ?? n.y, scale: 0 }}
              animate={{ x: n.x, y: n.y, scale: 1 }}
              exit={{ scale: 0 }}
              transition={spring}
            >
              <circle
                r={r}
                fill={col(n.data.prob)}
                stroke={n.data.id === currentId ? 'var(--accent)' : n.data.pending ? 'rgba(255,255,255,.4)' : 'rgba(255,255,255,.75)'}
                strokeWidth={n.data.id === currentId ? 3 : 1}
                strokeDasharray={n.data.pending ? '3 3' : undefined}
              />
              {showLabels && n.data.label && (
                <text y={-r - 5} textAnchor="middle" className="node-label">{n.data.label}</text>
              )}
            </motion.g>
          ))}
        </AnimatePresence>
      </g>
    </svg>
  );
}
