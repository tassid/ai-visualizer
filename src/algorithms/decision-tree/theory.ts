import { ESL, ISLR, MITCHELL, paper } from '../books';

export const theory = {
  theory: [
    'A classification tree partitions the input space into rectangles $R_1, \\ldots, R_M$ and predicts in each the share of each class. Finding the best tree of a given size is NP-hard, so CART grows it greedily: at each node, choose the split (feature $j$, threshold $s$) that most reduces the weighted impurity of the two children,',
    '$$\\Delta = G(\\text{parent}) - \\frac{n_L}{n}G(L) - \\frac{n_R}{n}G(R), \\qquad G(p) = \\sum_k p_k(1-p_k) = 2p(1-p)\\ \\text{for two classes}.$$',
    'The Gini index is the expected error of a classifier that guesses labels at random in proportion to the node\'s class shares. Entropy, $-\\sum_k p_k\\log p_k$, behaves almost the same and is used by ID3 and C4.5. Both are strictly concave, which matters: misclassification rate is not, and it often sees no gain in splits that clearly help, so it is a poor criterion for growing.',
    'Grown to full depth, a tree fits every training point and has very high variance. The CART answer is to grow a large tree and then prune it by cost complexity: for each $\\alpha$, find the subtree $T$ minimizing $\\sum_{m=1}^{|T|} n_m G_m + \\alpha|T|$, and pick $\\alpha$ by cross-validation. The maximum depth and minimum size used here are simpler stopping rules with the same goal.',
    'Trees are easy to read, handle mixed feature types and need no scaling, since only the order of values matters. Their weaknesses are instability, where a small change in the data can change the first split and so the whole tree, and axis-aligned boundaries, which need many steps to approximate a diagonal line. Ensembles such as random forests and boosting keep the strengths and fix both weaknesses.',
  ],
  references: [
    ISLR('Chapter 8, Tree-based methods'),
    ESL('Section 9.2, Tree-based methods'),
    MITCHELL('Chapter 3, Decision tree learning'),
    paper('Breiman, Friedman, Olshen and Stone, Classification and Regression Trees (1984)'),
    paper('Quinlan, Induction of decision trees, Machine Learning 1 (1986)'),
  ],
};
