/**
 * Algorithm registry. To add one, create its folder with model.ts and
 * index.tsx (see docs/NEW_ALGORITHM.md) and list it here. The order of this
 * list is the order on the home page; groups appear in order of first use.
 */
import type { AlgorithmDef } from '../core/types';
import { adaboost } from './adaboost';
import { dbscan } from './dbscan';
import { decisionTree } from './decision-tree';
import { discriminantAnalysis } from './discriminant-analysis';
import { gaussianProcess } from './gaussian-process';
import { gmm } from './gmm';
import { gradientBoosting } from './gradient-boosting';
import { hierarchical } from './hierarchical';
import { kernelSvm } from './kernel-svm';
import { kmeans } from './kmeans';
import { knn } from './knn';
import { linearRegression } from './linear-regression';
import { logisticRegression } from './logistic-regression';
import { meanShift } from './mean-shift';
import { mlp } from './mlp';
import { naiveBayes } from './naive-bayes';
import { optimizers } from './optimizers';
import { pca } from './pca';
import { perceptron } from './perceptron';
import { qLearning } from './q-learning';
import { randomForest } from './random-forest';
import { som } from './som';
import { svm } from './svm';

export const ALGORITHMS: AlgorithmDef[] = [
  // supervised
  perceptron,
  mlp,
  linearRegression,
  logisticRegression,
  discriminantAnalysis,
  naiveBayes,
  svm,
  kernelSvm,
  gaussianProcess,
  knn,
  decisionTree,
  randomForest,
  adaboost,
  gradientBoosting,
  // unsupervised
  kmeans,
  gmm,
  hierarchical,
  dbscan,
  meanShift,
  pca,
  som,
  // beyond
  qLearning,
  optimizers,
];

export const GROUPS = [...new Set(ALGORITHMS.map((a) => a.group))];

export const findAlgorithm = (id: string) => ALGORITHMS.find((a) => a.id === id);
