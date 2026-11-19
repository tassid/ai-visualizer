import { ESL, ISLR, PRML, paper } from '../books';

export const theory = {
  theory: [
    'For separable data, the maximum margin classifier solves',
    '$$\\min_{\\mathbf{w}, b}\\ \\tfrac{1}{2}\\lVert\\mathbf{w}\\rVert^2 \\quad\\text{subject to}\\quad y_i(\\mathbf{w}\\cdot\\mathbf{x}_i + b) \\ge 1 \\ \\text{for all } i.$$',
    'The constraints fix the scale of $\\mathbf{w}$ so that the closest points have $|f| = 1$; the distance between the two margin lines is then $2/\\lVert\\mathbf{w}\\rVert$, and minimizing $\\lVert\\mathbf{w}\\rVert$ maximizes it. The soft margin version of Cortes and Vapnik allows violations with slack variables $\\xi_i \\ge 0$ and a cost $C\\sum_i\\xi_i$. Eliminating the slacks gives the unconstrained form used here, with $\\lambda$ playing the role of $1/(nC)$:',
    '$$\\min_{\\mathbf{w}, b}\\ \\frac{\\lambda}{2}\\lVert\\mathbf{w}\\rVert^2 + \\frac{1}{n}\\sum_i \\max\\big(0,\\ 1 - y_i(\\mathbf{w}\\cdot\\mathbf{x}_i + b)\\big).$$',
    'The hinge loss is convex but not differentiable at the kink, so the update uses a subgradient: points with $y_i f(\\mathbf{x}_i) < 1$ contribute $-y_i\\mathbf{x}_i$, all others contribute nothing. With a step size decaying like $1/(\\lambda t)$ this is the Pegasos algorithm, which reaches accuracy $\\epsilon$ in $O(1/(\\lambda\\epsilon))$ iterations.',
    'The Lagrangian dual shows why only some points matter. The optimal weights are $\\mathbf{w} = \\sum_i \\alpha_i y_i\\mathbf{x}_i$, and the Karush-Kuhn-Tucker conditions force $\\alpha_i = 0$ for every point strictly outside the margin. The points with $\\alpha_i > 0$ are the support vectors. The dual also depends on the data only through inner products $\\mathbf{x}_i\\cdot\\mathbf{x}_j$, which is the opening for the kernel trick (see the kernel SVM page).',
    'Compared with logistic regression, the hinge loss is exactly zero for confident correct points, while the cross-entropy never is. That is why the SVM line depends on a few points near the boundary and the logistic line on all of them.',
  ],
  references: [
    ISLR('Chapter 9, Support vector machines'),
    ESL('Chapter 12, Support vector machines and flexible discriminants'),
    PRML('Chapter 7, Sparse kernel machines'),
    paper('Cortes and Vapnik, Support-vector networks, Machine Learning 20 (1995)'),
    paper('Shalev-Shwartz, Singer, Srebro and Cotter, Pegasos: primal estimated sub-gradient solver for SVM, Mathematical Programming 127 (2011)'),
  ],
};
