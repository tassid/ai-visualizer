/**
 * Contexts that make the plot interactive.
 *
 * The algorithm page provides them; thumbnails do not. Components inside a
 * plot check for them, so the same Visual is editable on its own page and
 * static on the home page.
 */
import { createContext, useContext } from 'react';
import type { Point } from '../core/types';

export interface EditApi {
  data: Point[];
  setData(d: Point[]): void;
  /** Class given to new points; null for unlabeled data. */
  brush: number | null;
  onHover(p: { x: number; y: number } | null): void;
  /** False for algorithms without data: the plot then only reports hover. */
  editable: boolean;
  /** Redraw after a handle changed the model state. */
  touch(): void;
}

export const EditContext = createContext<EditApi | null>(null);
export const useEdit = () => useContext(EditContext);
