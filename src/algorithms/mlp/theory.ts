import { DLBOOK, ESL, PRML, paper } from '../books';

export const theory = {
  theory: [
    'A multilayer perceptron is a composition of affine maps and elementwise nonlinearities. With $\\mathbf{a}^{(0)} = \\mathbf{x}$, each layer computes',
    '$$\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}, \\qquad \\mathbf{a}^{(l)} = g\\big(\\mathbf{z}^{(l)}\\big),$$',
    'and the output layer uses a sigmoid so that $\\hat p = \\sigma(z^{(L)})$ is a probability. Without the nonlinearity $g$ the whole network would collapse into one affine map, no more expressive than logistic regression. With it, the universal approximation theorem (Cybenko 1989, Hornik 1991) says one hidden layer with enough units can approximate any continuous function on a compact set. Depth does not add expressive power in that sense, but it can represent many functions with far fewer units.',
    'Training minimizes the average cross-entropy $\\mathcal{L} = -\\frac{1}{n}\\sum_i [y_i\\log\\hat p_i + (1-y_i)\\log(1-\\hat p_i)]$. Backpropagation is the chain rule applied in reverse order, reusing the forward pass. Define the error signal $\\boldsymbol{\\delta}^{(l)} = \\partial\\mathcal{L}/\\partial\\mathbf{z}^{(l)}$. For a sigmoid output with cross-entropy, $\\delta^{(L)} = \\hat p - y$, and then',
    '$$\\boldsymbol{\\delta}^{(l-1)} = \\big(W^{(l)\\top}\\boldsymbol{\\delta}^{(l)}\\big) \\odot g\'\\big(\\mathbf{z}^{(l-1)}\\big), \\qquad \\frac{\\partial\\mathcal{L}}{\\partial W^{(l)}} = \\boldsymbol{\\delta}^{(l)}\\,\\mathbf{a}^{(l-1)\\top}, \\qquad \\frac{\\partial\\mathcal{L}}{\\partial \\mathbf{b}^{(l)}} = \\boldsymbol{\\delta}^{(l)}.$$',
    'The whole gradient costs about as much as one more forward pass, which is what makes training large networks feasible. The loss surface is not convex: there are many minima and saddle points, and the result depends on the initialization. Xavier initialization (variance $1/n_\\text{in}$) and He initialization (variance $2/n_\\text{in}$ for ReLU) keep the scale of $\\boldsymbol{\\delta}$ roughly constant across layers, so gradients neither vanish nor explode.',
    'Optimizers differ in how they turn gradients into steps. Momentum averages past gradients, which damps oscillation across narrow valleys. Adam keeps running estimates of the first and second moments of each gradient coordinate and divides by the square root of the second, so every weight gets its own step size. Its bias correction, dividing by $1-\\beta^t$, compensates for the moments starting at zero.',
  ],
  references: [
    DLBOOK('Chapter 6, Deep feedforward networks, and Chapter 8, Optimization for training deep models'),
    PRML('Chapter 5, Neural networks'),
    ESL('Chapter 11, Neural networks'),
    paper('Rumelhart, Hinton and Williams, Learning representations by back-propagating errors, Nature 323 (1986)'),
    paper('Kingma and Ba, Adam: a method for stochastic optimization, ICLR (2015)', 'https://arxiv.org/abs/1412.6980'),
    paper('3Blue1Brown, Neural networks (video series)', 'https://www.3blue1brown.com/topics/neural-networks'),
  ],
};
