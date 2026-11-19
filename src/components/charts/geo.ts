/**
 * Minimal conversion from a GeoJSON MultiPolygon (d3-contour's output) to an
 * SVG path string. Replaces d3-geo, which would be heavy just to scale points.
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
