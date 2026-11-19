/**
 * Algorithm registry. To add one, create its folder with model.ts and
 * index.tsx (see docs/NEW_ALGORITHM.md) and list it here. The order of this
 * list is the order on the home page.
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
