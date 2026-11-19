import { ISLR, MITCHELL, PRML, paper } from '../books';

export const theory = {
  theory: [
    'Bayes\' theorem turns a model of how each class generates data into a classifier:',
    '$$P(k\\mid\\mathbf{x}) = \\frac{\\pi_k\\,p(\\mathbf{x}\\mid k)}{\\sum_j \\pi_j\\,p(\\mathbf{x}\\mid j)}.$$',
    'The prior $\\pi_k$ is the share of class $k$; the class-conditional density $p(\\mathbf{x}\\mid k)$ is the hard part. In $d$ dimensions a full model needs to capture every dependency between features. Naive Bayes assumes they are independent given the class, so the density factors into one-dimensional pieces, $p(\\mathbf{x}\\mid k) = \\prod_j p(x_j\\mid k)$. In the Gaussian version each piece is $\\mathcal{N}(x_j;\\ \\mu_{kj}, \\sigma^2_{kj})$, estimated by the sample mean and variance.',
    'Taking logs, the decision compares $\\log\\pi_k - \\sum_j \\big[\\log\\sigma_{kj} + (x_j - \\mu_{kj})^2/(2\\sigma_{kj}^2)\\big]$ across classes. That is quadratic in $\\mathbf{x}$, so the boundary is a conic section (an ellipse, a parabola or a hyperbola). If both classes share the same variances, the quadratic terms cancel and the boundary becomes a line.',
    'The independence assumption is rarely true, yet naive Bayes often classifies well. Domingos and Pazzani explained why: classification only needs the largest posterior to be on the right class, not the probabilities to be calibrated, and that can survive large errors in the estimates. Where the assumption hurts most is with strongly correlated features, which is what linear and quadratic discriminant analysis fix by modeling full covariance matrices.',
    'The means and variances here are computed online with Welford\'s algorithm. The naive formula $\\frac{1}{n}\\sum x^2 - \\bar x^2$ subtracts two large, nearly equal numbers and loses precision; Welford\'s update avoids that and never stores the data.',
  ],
  references: [
    ISLR('Section 4.4, Generative models for classification (including naive Bayes)'),
    PRML('Section 4.2, Probabilistic generative models'),
    MITCHELL('Chapter 6, Bayesian learning'),
    paper('Domingos and Pazzani, On the optimality of the simple Bayesian classifier under zero-one loss, Machine Learning 29 (1997)'),
    paper('Welford, Note on a method for calculating corrected sums of squares and products, Technometrics 4 (1962)'),
  ],
};
