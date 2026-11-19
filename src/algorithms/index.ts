/**
 * Registro de algoritmos. Para adicionar um novo, crie a pasta com
 * model.ts + index.tsx (veja docs/NOVO_ALGORITMO.md) e inclua aqui.
 * A ordem desta lista é a ordem da página inicial.
 */
import type { AlgorithmDef } from '../core/types';
import { dbscan } from './dbscan';
import { decisionTree } from './decision-tree';
import { kmeans } from './kmeans';
import { knn } from './knn';
import { linearRegression } from './linear-regression';
import { logisticRegression } from './logistic-regression';
import { mlp } from './mlp';
import { naiveBayes } from './naive-bayes';
import { pca } from './pca';
import { perceptron } from './perceptron';
import { randomForest } from './random-forest';
import { svm } from './svm';

export const ALGORITHMS: AlgorithmDef[] = [
  perceptron,
  mlp,
  linearRegression,
  logisticRegression,
  svm,
  naiveBayes,
  knn,
  decisionTree,
  randomForest,
  kmeans,
  dbscan,
  pca,
];

export const GROUPS = [...new Set(ALGORITHMS.map((a) => a.group))];

export const findAlgorithm = (id: string) => ALGORITHMS.find((a) => a.id === id);
