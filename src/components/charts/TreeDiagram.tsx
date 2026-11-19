/**
 * Tree diagram using d3-hierarchy's `tree()` layout.
 *
 * D3 computes x and y for every node; Framer Motion grows each new node from
 * its parent and slides existing nodes when the layout changes.
 * Hovering the plot lights up the path a point takes from the root to a leaf.
 */
import { hierarchy, tree } from 'd3-hierarchy';
import { AnimatePresence, motion } from 'framer-motion';
import { COLORS, divergingColor, rgbCss } from '../../core/colors';

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
  path?: Set<number>;
  width: number;
  height: number;
}

const ease = { duration: 0.35, ease: [0.45, 0, 0.55, 1] } as const;

export function TreeDiagram({ root, currentId, path, width, height }: Props) {
  const h = hierarchy(root, (d) => d.children);
  const leaves = h.leaves().length;
  const laid = tree<DiagramNode>().size([width - 48, height - 56]).separation(() => 1)(h);
  const nodes = laid.descendants();
  const links = laid.links();
  const r = Math.max(3.5, Math.min(10, (width - 48) / Math.max(1, leaves) / 2.6));
  const showLabels = leaves <= 14;
  const onPath = (id: number) => path?.has(id) ?? false;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="aux-svg">
      <g transform="translate(24, 30)">
        <AnimatePresence>
          {links.map((l) => {
            const lit = onPath(l.source.data.id) && onPath(l.target.data.id);
            return (
              <motion.line
                key={`${l.source.data.id}-${l.target.data.id}`}
                initial={{ x1: l.source.x, y1: l.source.y, x2: l.source.x, y2: l.source.y }}
                animate={{ x1: l.source.x, y1: l.source.y, x2: l.target.x, y2: l.target.y }}
                exit={{ opacity: 0 }}
                transition={ease}
                stroke={lit ? COLORS.yellow : 'rgba(255,255,255,.3)'}
                strokeWidth={lit ? 2.5 : 1.2}
              />
            );
          })}
          {nodes.map((n) => (
            <motion.g
              key={n.data.id}
              initial={{ x: n.parent?.x ?? n.x, y: n.parent?.y ?? n.y, scale: 0 }}
              animate={{ x: n.x, y: n.y, scale: 1 }}
              exit={{ scale: 0 }}
              transition={ease}
            >
              <circle
                r={r}
                fill={rgbCss(divergingColor(n.data.prob))}
                stroke={n.data.id === currentId || onPath(n.data.id) ? COLORS.yellow : 'rgba(255,255,255,.7)'}
                strokeWidth={n.data.id === currentId || onPath(n.data.id) ? 2.5 : 1}
                strokeDasharray={n.data.pending ? '3 3' : undefined}
              />
              {showLabels && n.data.label && <text y={-r - 6} textAnchor="middle" className="node-label">{n.data.label}</text>}
            </motion.g>
          ))}
        </AnimatePresence>
      </g>
    </svg>
  );
}
