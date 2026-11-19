/**
 * TeX rendering with KaTeX. Formulas use the Computer Modern look of the
 * 3Blue1Brown videos, and \color{...} picks colors from the Manim palette.
 */
import katex from 'katex';
import { useMemo } from 'react';

export function Tex({ children, display = false }: { children: string; display?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: display, throwOnError: false }), [children, display]);
  return <span className={display ? 'tex tex-display' : 'tex'} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Text with inline math between $...$. */
export function Prose({ text }: { text: string }) {
  const parts = text.split(/(\$[^$]+\$)/g);
  return (
    <>
      {parts.map((p, i) => (p.startsWith('$') && p.endsWith('$') ? <Tex key={i}>{p.slice(1, -1)}</Tex> : <span key={i}>{p}</span>))}
    </>
  );
}
