import { ESL, ISLR, PRML, paper } from '../books';

export const theory = {
  theory: [
    'k-means looks for $k$ centers $\\boldsymbol{\\mu}_1, \\ldots, \\boldsymbol{\\mu}_k$ and an assignment $c(i)$ of each point that minimize the within-cluster sum of squares,',
    '$$J = \\sum_{i=1}^{n}\\big\\lVert\\mathbf{x}_i - \\boldsymbol{\\mu}_{c(i)}\\big\\rVert^2.$$',
    'Lloyd\'s algorithm is coordinate descent on $J$. With the centers fixed, the best assignment sends each point to its nearest center. With the assignments fixed, $J$ is a sum of separate quadratics and the best center is the mean of its points, since $\\sum_i\\lVert\\mathbf{x}_i - \\boldsymbol{\\mu}\\rVert^2$ is minimized at $\\boldsymbol{\\mu} = \\bar{\\mathbf{x}}$. Neither step can increase $J$, and there are finitely many assignments, so the algorithm stops. Finding the global minimum, however, is NP-hard even for $k = 2$, and Lloyd\'s algorithm finds only a local one.',
    'Initialization therefore matters. k-means++ picks the first center uniformly at random and each next center with probability proportional to $D(\\mathbf{x})^2$, the squared distance to the nearest center already chosen. Arthur and Vassilvitskii proved that this alone gives an expected cost within a factor $O(\\log k)$ of the optimum.',
    'k-means is the limit of a Gaussian mixture model in which every component has the same spherical covariance $\\sigma^2 I$ and $\\sigma \\to 0$: soft responsibilities become hard assignments, and the EM algorithm becomes Lloyd\'s algorithm. That explains its implicit assumptions: clusters that are round, of similar size and similar spread. The moons and circles break all three.',
    'Choosing $k$ is outside the algorithm. Common tools are the elbow in $J$ as a function of $k$, the silhouette coefficient and the gap statistic.',
  ],
  references: [
    ISLR('Section 12.4.1, K-means clustering'),
    ESL('Section 14.3.6, K-means'),
    PRML('Section 9.1, K-means clustering'),
    paper('Lloyd, Least squares quantization in PCM, IEEE Transactions on Information Theory 28 (1982)'),
    paper('Arthur and Vassilvitskii, k-means++: the advantages of careful seeding, SODA (2007)'),
  ],
};
