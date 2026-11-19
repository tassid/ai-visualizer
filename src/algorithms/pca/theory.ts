import { ESL, ISLR, PRML, paper } from '../books';

export const theory = {
  theory: [
    'Center the data so that each feature has mean zero, and let $\\Sigma = \\frac{1}{n}X^\\top X$ be the sample covariance. Projecting onto a unit vector $\\mathbf{u}$ gives the scores $X\\mathbf{u}$, whose variance is',
    '$$\\operatorname{Var}(\\mathbf{u}) = \\frac{1}{n}\\lVert X\\mathbf{u}\\rVert^2 = \\mathbf{u}^\\top\\Sigma\\,\\mathbf{u}.$$',
    'Maximizing this under the constraint $\\lVert\\mathbf{u}\\rVert = 1$ with a Lagrange multiplier gives $\\Sigma\\mathbf{u} = \\lambda\\mathbf{u}$: the stationary directions are the eigenvectors of $\\Sigma$, and the variance along each is its eigenvalue. The first principal component is the eigenvector with the largest eigenvalue $\\lambda_1$; the second maximizes variance among directions orthogonal to the first, and so on. The fraction of variance explained by the first $k$ components is $\\sum_{j\\le k}\\lambda_j / \\sum_j\\lambda_j$.',
    'There is an equivalent view: PCA finds the $k$-dimensional subspace that minimizes the mean squared distance from the points to their projections. Since every point\'s squared length splits into the part along the subspace and the part perpendicular to it, maximizing the first is the same as minimizing the second. In the animation, the yellow segments are those perpendicular errors.',
    'In practice PCA is computed from the singular value decomposition $X = U S V^\\top$: the columns of $V$ are the principal directions and $\\lambda_j = s_j^2/n$. This avoids forming $X^\\top X$, which squares the condition number. PCA is sensitive to scale, so features measured in different units are usually standardized first. Probabilistic PCA (Tipping and Bishop) recasts it as a Gaussian latent variable model, which connects it to factor analysis.',
  ],
  references: [
    ISLR('Section 12.2, Principal components analysis'),
    ESL('Section 14.5.1, Principal components'),
    PRML('Chapter 12, Continuous latent variables'),
    paper('Pearson, On lines and planes of closest fit to systems of points in space, Philosophical Magazine 2 (1901)'),
    paper('Hotelling, Analysis of a complex of statistical variables into principal components, Journal of Educational Psychology 24 (1933)'),
    paper('Shlens, A tutorial on principal component analysis (2014)', 'https://arxiv.org/abs/1404.1100'),
  ],
};
