import { ESL, ISLR, PRML } from '../books';

export const theory = {
  theory: [
    'Linear regression assumes $y = \\mathbf{w}\\cdot\\boldsymbol{\\phi}(x) + b + \\varepsilon$, where $\\boldsymbol{\\phi}(x) = (x, x^2, \\ldots, x^d)$ are the features and $\\varepsilon$ is noise. Least squares picks the coefficients that minimize the residual sum of squares. Stacking the features into a design matrix $X$ (with a column of ones for $b$), the objective and its gradient are',
    '$$\\mathrm{RSS}(\\boldsymbol{\\beta}) = \\lVert \\mathbf{y} - X\\boldsymbol{\\beta} \\rVert^2, \\qquad \\nabla\\,\\mathrm{RSS} = -2X^\\top(\\mathbf{y} - X\\boldsymbol{\\beta}).$$',
    'Setting the gradient to zero gives the normal equations $X^\\top X\\boldsymbol{\\beta} = X^\\top\\mathbf{y}$, so $\\hat{\\boldsymbol{\\beta}} = (X^\\top X)^{-1}X^\\top\\mathbf{y}$ whenever $X^\\top X$ is invertible. Geometrically, $X\\hat{\\boldsymbol{\\beta}}$ is the orthogonal projection of $\\mathbf{y}$ onto the column space of $X$: the residual vector is perpendicular to every feature.',
    'If the noise is Gaussian, $\\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)$, then minimizing RSS is the same as maximizing the likelihood, since the log-likelihood is $-\\mathrm{RSS}/(2\\sigma^2)$ plus a constant. Squared error is therefore not arbitrary: it is what Gaussian noise implies.',
    'Gradient descent reaches the same answer iteratively. The RSS is a convex quadratic, its contours are ellipses, and their shape is set by the eigenvalues of $X^\\top X$. Descent converges for a step size $\\eta < 2/\\lambda_\\text{max}$ (scaled by $2/n$ for the mean), and its speed depends on the condition number $\\lambda_\\text{max}/\\lambda_\\text{min}$: long thin ellipses make it zigzag. That is what the path on the landscape shows when you raise the learning rate.',
    'Raising the polynomial degree lowers the training error but raises the variance of the fit, which is the bias and variance trade-off. Ridge regression adds $\\lambda\\lVert\\mathbf{w}\\rVert^2$ to the objective, giving $\\hat{\\boldsymbol{\\beta}} = (X^\\top X + \\lambda I)^{-1}X^\\top\\mathbf{y}$, which shrinks the coefficients and tames high degrees.',
  ],
  references: [
    ISLR('Chapter 3, Linear regression, and Chapter 6, Linear model selection and regularization'),
    ESL('Chapter 3, Linear methods for regression'),
    PRML('Chapter 3, Linear models for regression'),
  ],
};
