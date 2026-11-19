import { ESL, ISLR, PRML } from '../books';

export const theory = {
  theory: [
    'Logistic regression models the log-odds of class 1 as a linear function of the inputs:',
    '$$\\log\\frac{p(\\mathbf{x})}{1 - p(\\mathbf{x})} = \\mathbf{w}\\cdot\\mathbf{x} + b \\quad\\Longleftrightarrow\\quad p(\\mathbf{x}) = \\sigma(\\mathbf{w}\\cdot\\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}\\cdot\\mathbf{x} + b)}}.$$',
    'Each unit of $\\mathbf{w}\\cdot\\mathbf{x}$ multiplies the odds by $e$. The boundary $p = 0.5$ is the line $\\mathbf{w}\\cdot\\mathbf{x} + b = 0$, and the dashed lines where $p = 0.25$ and $0.75$ sit at $\\mathbf{w}\\cdot\\mathbf{x} + b = \\mp\\ln 3$, a distance $\\ln 3/\\lVert\\mathbf{w}\\rVert$ from it. So $\\lVert\\mathbf{w}\\rVert$ measures how sharp, or how confident, the model is.',
    'The weights maximize the likelihood of the labels, $\\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$. Its negative log is the cross-entropy, and since $\\sigma\'(z) = \\sigma(z)(1-\\sigma(z))$ the gradient simplifies neatly:',
    '$$\\mathcal{L}(\\mathbf{w}) = -\\sum_i \\big[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\big], \\qquad \\nabla\\mathcal{L} = \\sum_i (p_i - y_i)\\,\\mathbf{x}_i.$$',
    'The Hessian $\\sum_i p_i(1-p_i)\\,\\mathbf{x}_i\\mathbf{x}_i^\\top$ is positive semidefinite, so the loss is convex and has no bad local minima. Newton\'s method on this problem is called iteratively reweighted least squares, which is how statistics packages fit it.',
    'When the classes are perfectly separable the maximum likelihood estimate does not exist: scaling $\\mathbf{w}$ up always lowers the loss, and the weights diverge. An L2 penalty $\\frac{\\lambda}{2}\\lVert\\mathbf{w}\\rVert^2$ fixes this and corresponds to a Gaussian prior on $\\mathbf{w}$.',
  ],
  references: [
    ISLR('Chapter 4, Classification (Section 4.3, Logistic regression)'),
    ESL('Section 4.4, Logistic regression'),
    PRML('Section 4.3, Probabilistic discriminative models'),
  ],
};
