/**
 * Core types.
 *
 * The central idea: every algorithm is a JavaScript generator (function*).
 * Each `yield` is one step of the animation. The algorithm reads like
 * textbook pseudocode, and the animation engine only has to call `.next()`
 * at the pace the user picks.
 */
import type { ComponentType } from 'react';
import type { DatasetId } from './datasets';

/** One example on the plane. `c` is the class (0 or 1); ignored in regression and clustering. */
export interface Point {
  x: number;
  y: number;
  c: number;
}

export type Kind = 'classification' | 'regression' | 'clustering' | 'reduction' | 'reinforcement' | 'optimization';

export type ParamValue = number | string;
export type Params = Record<string, ParamValue>;

export type ParamSpec =
  | { key: string; label: string; type: 'range'; min: number; max: number; step: number; value: number; help?: string }
  | { key: string; label: string; type: 'select'; options: ParamValue[]; value: ParamValue; help?: string };

/** What each step reports. `line` is the pseudocode line to highlight. */
export interface StepInfo {
  line: number;
  message: string;
}

/** A trainer: the mutable model state and the generator that changes it. */
export interface Trainer<S> {
  state: S;
  steps: Generator<StepInfo, StepInfo | void, void>;
}

export type Hover = { x: number; y: number } | null;

/** Props every algorithm visual receives. */
export interface VisualProps<S> {
  state: S;
  data: Point[];
  params: Params;
  step: StepInfo | null;
  /** Mouse position in world coordinates ([-1, 1]), or null. */
  hover: Hover;
  done: boolean;
  /** Increments every frame in which the state changed. Use it as a memo key. */
  version: number;
  /** Thumbnail mode: no handles, no per-step highlights. */
  compact?: boolean;
}

export interface Metric {
  label: string;
  value: string | number;
}

export interface AlgorithmDocs {
  description: string;
  steps: string[];
  complexity: [string, string][];
  complexityNote?: string;
  /** Pseudocode lines; indices match StepInfo.line. */
  pseudocode: string[];
  exercises: string[];
  /** Textbook-style explanation. Paragraphs; inline math in $...$, display math in $$...$$. */
  theory: string[];
  /** Books chapters and original papers. */
  references: Reference[];
}

export interface Reference {
  text: string;
  url?: string;
}

export interface AlgorithmDef<S = any> {
  id: string;
  name: string;
  group: string;
  kind: Kind;
  datasets: DatasetId[];
  defaultDataset: DatasetId;
  /** Index into SPEEDS (engine/speeds.ts). */
  defaultSpeed: number;
  params: ParamSpec[];
  create(data: Point[], params: Params): Trainer<S>;
  Visual: ComponentType<VisualProps<S>>;
  aux?: { title: string; Component: ComponentType<VisualProps<S>> };
  metrics(state: S, data: Point[], params: Params): Metric[];
  history?: { label: string; get(state: S): number[] };
  /** The model's key formula, filled with live numbers (TeX). */
  equation?(state: S, data: Point[], params: Params, hover: Hover): string | null;
  /** What can be dragged or hovered, beyond adding points. */
  hint?: string;
  /** The algorithm builds its own world and ignores the dataset (Q-learning, optimizers). */
  noData?: boolean;
  /** Steps to run when drawing the home page thumbnail. */
  thumbnailSteps?: number;
  docs: AlgorithmDocs;
}

export const defaultParams = (specs: ParamSpec[]): Params =>
  Object.fromEntries(specs.map((p) => [p.key, p.value]));
