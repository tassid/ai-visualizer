import { ESL, PRML, paper } from '../books';

export const theory = {
  theory: [
    'The perceptron is a linear threshold unit. With labels $y \\in \\{-1, +1\\}$ it predicts $\\hat y = \\operatorname{sign}(\\mathbf{w}\\cdot\\mathbf{x} + b)$. Geometrically, $\\mathbf{w}$ is the normal vector of the line $\\mathbf{w}\\cdot\\mathbf{x} + b = 0$, and $(\\mathbf{w}\\cdot\\mathbf{x} + b)/\\lVert\\mathbf{w}\\rVert$ is the signed distance from a point to that line. A point is misclassified exactly when $y\\,(\\mathbf{w}\\cdot\\mathbf{x} + b) \\le 0$.',
    'The learning rule is stochastic gradient descent on the perceptron criterion, which only counts misclassified points $\\mathcal{M}$ by how far they are on the wrong side:',
    '$$E(\\mathbf{w}, b) = -\\sum_{i \\in \\mathcal{M}} y_i\\,(\\mathbf{w}\\cdot\\mathbf{x}_i + b), \\qquad \\nabla_{\\mathbf{w}} E = -\\sum_{i \\in \\mathcal{M}} y_i\\,\\mathbf{x}_i.$$',
    'Taking one misclassified point at a time gives $\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta\\,y_i\\,\\mathbf{x}_i$. Each update makes $y_i\\,\\mathbf{w}\\cdot\\mathbf{x}_i$ larger by $\\eta\\lVert\\mathbf{x}_i\\rVert^2$, so the line turns toward the right side of that point, although it can undo the work done for other points.',
    'Novikoff\'s theorem (1962) bounds the damage. If some unit vector $\\mathbf{w}^*$ separates the data with margin $\\gamma$, meaning $y_i\\,\\mathbf{w}^*\\cdot\\mathbf{x}_i \\ge \\gamma$ for all $i$, and every $\\lVert\\mathbf{x}_i\\rVert \\le R$, then the perceptron makes at most $(R/\\gamma)^2$ mistakes. The proof tracks two quantities: $\\mathbf{w}\\cdot\\mathbf{w}^*$ grows by at least $\\gamma$ per mistake while $\\lVert\\mathbf{w}\\rVert^2$ grows by at most $R^2$, and Cauchy-Schwarz forbids both from continuing forever.',
    'The theorem says nothing when the data is not separable, and then the algorithm never settles. It also says nothing about which separating line it finds: any one will do, which is why the margin-maximizing SVM was developed later. Minsky and Papert showed in 1969 that a single perceptron cannot compute XOR, which stalled neural network research until backpropagation made multilayer networks trainable.',
  ],
  references: [
    PRML('Section 4.1.7, The perceptron algorithm'),
    ESL('Section 4.5, Separating hyperplanes'),
    paper('Rosenblatt, The perceptron: a probabilistic model for information storage and organization in the brain, Psychological Review 65 (1958)'),
    paper('Novikoff, On convergence proofs on perceptrons, Symposium on the Mathematical Theory of Automata (1962)'),
    paper('Minsky and Papert, Perceptrons (1969)'),
  ],
};
