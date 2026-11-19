/**
 * Tipos centrais do projeto.
 *
 * A ideia principal: cada algoritmo é escrito como um *generator* de JavaScript
 * (function*). Cada `yield` é um passo da animação. Isso deixa o código do
 * algoritmo parecido com o pseudocódigo de um livro, e o motor de animação só
 * precisa chamar `.next()` no ritmo escolhido pelo usuário.
 */
import type { ComponentType } from 'react';
import type { DatasetId } from './datasets';

/** Um exemplo no plano. `c` é a classe (0 ou 1); em regressão e agrupamento é ignorada. */
export interface Point {
  x: number;
  y: number;
  c: number;
}

export type Kind = 'classification' | 'regression' | 'clustering' | 'reduction';

export type ParamValue = number | string;
export type Params = Record<string, ParamValue>;

export type ParamSpec =
  | { key: string; label: string; type: 'range'; min: number; max: number; step: number; value: number; help?: string }
  | { key: string; label: string; type: 'select'; options: ParamValue[]; value: ParamValue; help?: string };

/**
 * O que cada passo informa para a interface.
 * `line` é o índice da linha do pseudocódigo que deve ficar destacada.
 */
export interface StepInfo {
  line: number;
  message: string;
}

/** Um treinador: o estado mutável do modelo e o generator que o modifica. */
export interface Trainer<S> {
  state: S;
  steps: Generator<StepInfo, StepInfo | void, void>;
}

/** Props recebidas pelos componentes visuais de cada algoritmo. */
export interface VisualProps<S> {
  state: S;
  data: Point[];
  params: Params;
  step: StepInfo | null;
  /** Posição do mouse em coordenadas do mundo ([-1, 1]), ou null. */
  hover: { x: number; y: number } | null;
  done: boolean;
  /** Incrementa a cada quadro em que o estado mudou. Útil como dependência de memo. */
  version: number;
  /** Miniatura (cards da página inicial): desliga detalhes caros. */
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
  /** Linhas do pseudocódigo; os índices correspondem a StepInfo.line. */
  pseudocode: string[];
  /** Ideias de exercícios para quem está estudando. */
  exercises: string[];
}

export interface AlgorithmDef<S = any> {
  id: string;
  name: string;
  group: string;
  tag: string;
  kind: Kind;
  summary: string;
  datasets: DatasetId[];
  defaultDataset: DatasetId;
  /** Índice em SPEEDS (engine/speeds.ts). */
  defaultSpeed: number;
  params: ParamSpec[];
  create(data: Point[], params: Params): Trainer<S>;
  Visual: ComponentType<VisualProps<S>>;
  aux?: { title: string; Component: ComponentType<VisualProps<S>> };
  metrics(state: S, data: Point[], params: Params): Metric[];
  history?: { label: string; get(state: S): number[] };
  hint?: string;
  /** Quantos passos rodar para gerar a miniatura da página inicial. */
  thumbnailSteps?: number;
  docs: AlgorithmDocs;
}

export const defaultParams = (specs: ParamSpec[]): Params =>
  Object.fromEntries(specs.map((p) => [p.key, p.value]));
