import { ESL, ISLR, paper } from '../books';

export const theory = {
  theory: [
    'Averaging reduces variance. If $B$ predictors each have variance $\\sigma^2$ and pairwise correlation $\\rho$, their average has variance',
    '$$\\rho\\,\\sigma^2 + \\frac{1-\\rho}{B}\\,\\sigma^2.$$',
    'The second term vanishes as $B$ grows, but the first does not: correlation sets a floor. Bagging (bootstrap aggregating) trains each tree on a bootstrap sample, $n$ draws with replacement, which decorrelates the trees somewhat. Random forests go further by letting each split consider only $m$ randomly chosen features out of $d$, typically $m = \\sqrt{d}$ for classification. A strong feature then cannot dominate the top of every tree, $\\rho$ drops, and the floor drops with it.',
    'Each tree is grown deep and unpruned, so it has low bias and high variance; the averaging takes care of the variance. Adding trees never causes overfitting in the sense of the error going up: the forest error converges as $B$ grows, and the main cost of more trees is time.',
    'The probability that a given point is never drawn into a bootstrap sample is $(1 - 1/n)^n \\to e^{-1} \\approx 0.368$. For each point, about a third of the trees never saw it, and their averaged vote is an honest prediction for that point. The out-of-bag error built this way is almost identical to leave-one-out cross-validation and comes for free.',
    'Forests also give a measure of feature importance: the total decrease in Gini impurity from splits on each feature, or the drop in out-of-bag accuracy when a feature\'s values are randomly permuted.',
  ],
  references: [
    ISLR('Section 8.2, Bagging, random forests, boosting'),
    ESL('Chapter 15, Random forests, and Section 8.7, Bagging'),
    paper('Breiman, Bagging predictors, Machine Learning 24 (1996)'),
    paper('Breiman, Random forests, Machine Learning 45 (2001)'),
  ],
};
