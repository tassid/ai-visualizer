import { ESL, ISLR, MITCHELL, paper } from '../books';

export const theory = {
  theory: [
    'k-nearest neighbors estimates the class probability at a point by the share of each class among its $k$ closest training examples $N_k(\\mathbf{x})$:',
    '$$\\hat P(y = 1\\mid\\mathbf{x}) = \\frac{1}{k}\\sum_{i\\in N_k(\\mathbf{x})} y_i.$$',
    'It is a nonparametric method: there is no fixed set of parameters, and the model grows with the data. The Bayes classifier, which picks the class with the highest true posterior, is the best any classifier can do. Cover and Hart proved in 1967 that as $n \\to \\infty$ the error of 1-NN is at most twice the Bayes error, and if $k \\to \\infty$ with $k/n \\to 0$ the k-NN error converges to the Bayes error.',
    'The choice of $k$ is the bias and variance trade-off in its purest form. With $k = 1$ the boundary passes around every training point: zero training error, high variance. With large $k$ the estimate averages over a wide region: low variance, but the boundary is pulled toward a straight line and fine structure is lost. Its effective number of parameters is roughly $n/k$. The leave-one-out estimate shown here is a cheap and nearly unbiased way to choose $k$.',
    'k-NN suffers most from the curse of dimensionality. In $d$ dimensions, to capture a fraction $r$ of uniformly spread data a neighborhood must cover a fraction $r^{1/d}$ of each axis: for $d = 10$ and $r = 0.01$ that is 63% of the range. Neighbors stop being local, and distances concentrate so that the nearest and farthest points are almost equally far.',
  ],
  references: [
    ISLR('Section 2.2.3, The classification setting (k-nearest neighbors)'),
    ESL('Chapter 13, Prototype methods and nearest-neighbors, and Section 2.5, Local methods in high dimensions'),
    MITCHELL('Chapter 8, Instance-based learning'),
    paper('Cover and Hart, Nearest neighbor pattern classification, IEEE Transactions on Information Theory 13 (1967)'),
  ],
};
