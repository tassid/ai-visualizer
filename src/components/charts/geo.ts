/**
 * Conversão mínima de MultiPolygon (GeoJSON, saída do d3-contour) para o
 * atributo `d` de um <path> SVG. Substitui o d3-geo, que seria pesado demais
 * para só escalar coordenadas.
 */
import type { ContourMultiPolygon } from 'd3-contour';

export type Projector = (x: number, y: number) => [number, number];

export const geoIdentity = (sx: number, sy: number): Projector => (x, y) => [x * sx, y * sy];

export function geoPath(poly: ContourMultiPolygon, project: Projector): string {
  let d = '';
  for (const polygon of poly.coordinates) {
    for (const ring of polygon) {
      ring.forEach(([x, y], i) => {
        const [px, py] = project(x, y);
        d += (i ? 'L' : 'M') + px.toFixed(1) + ',' + py.toFixed(1);
      });
      d += 'Z';
    }
  }
  return d;
}
