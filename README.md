# AI Visualizer

Interactive, step-by-step visualizations of machine learning algorithms. Like [Sort Visualizer](https://github.com/Myphz/sortvisualizer), but for models: you watch a perceptron, a neural network, an SVM, k-means and others train, one update at a time. The look follows [3Blue1Brown](https://www.3blue1brown.com): Manim colors, a number plane, Computer Modern math.

**Live:** https://tassid.github.io/ai-visualizer/

Every algorithm is written from scratch in TypeScript, with no ML library, and meant to be read alongside the animation.

## What you can do

1. Play, pause, or step through single updates. The pseudocode highlights the line that just ran, and the model's formula updates with live numbers.
2. Edit the data: click to add a point, drag to move one, shift-click to delete. A trained model retrains immediately, so the boundary follows your hand.
3. Move the model itself: drag the line of a perceptron, logistic regression or SVM, fit a regression line by hand, place k-means centroids or mixture components, turn the PCA axis, pick the optimizers' starting point, build walls in the Q-learning world.
4. Probe it: hover the plane to send a point through the neural network, trace its path down the decision tree, count the forest's votes, or see the k nearest neighbors.

## Algorithms

23 algorithms, each with an animation, a live formula, pseudocode, a textbook-style theory section with derivations, exercises and a reading list (chapters of ISLR, ESL, Bishop, Goodfellow, Rasmussen and Williams, Sutton and Barto, and the original papers).

| Group | Algorithm | What the animation shows |
| --- | --- | --- |
| Neural networks | Perceptron | The line correcting itself on each mistake, with the weight vector |
| Neural networks | Multilayer perceptron | The boundary, and every neuron's activation map |
| Linear, probabilistic, kernel | Linear regression | Squared errors as squares, polynomial fits, loss landscape |
| Linear, probabilistic, kernel | Logistic regression | Probability, boundary, the p = 0.25 and 0.75 lines |
| Linear, probabilistic, kernel | LDA and QDA | Full covariance ellipses, linear or quadratic boundary |
| Linear, probabilistic, kernel | Gaussian naive Bayes | Axis-aligned class ellipses estimated point by point |
| Linear, probabilistic, kernel | Linear SVM | Margin, support vectors, hinge loss |
| Linear, probabilistic, kernel | Kernel SVM | RBF bumps, margins as contour lines, support vector weights |
| Linear, probabilistic, kernel | Gaussian process regression | Posterior mean, uncertainty band and sampled functions |
| Neighbors, trees, ensembles | k-nearest neighbors | A sweep of the plane with neighbors and radius |
| Neighbors, trees, ensembles | Decision tree | Gini splits level by level, and the tree itself |
| Neighbors, trees, ensembles | Random forest | Bootstrap samples, averaged trees, out-of-bag error |
| Neighbors, trees, ensembles | AdaBoost | Stumps voting, point weights growing on mistakes |
| Neighbors, trees, ensembles | Gradient boosting | Trees fit to residuals, residuals shrinking |
| Clustering | k-means | Assignment, centroids gliding, Voronoi regions |
| Clustering | Gaussian mixture (EM) | Soft responsibilities, ellipses, rising log-likelihood |
| Clustering | Hierarchical clustering | Merges with four linkages and a growing dendrogram |
| Clustering | DBSCAN | Core, border and noise points, the expansion queue |
| Clustering | Mean shift | Every point climbing the density to its peak |
| Dimensionality reduction | PCA | A turning line, projections, variance by angle |
| Dimensionality reduction | Self-organizing map | A neuron grid unfolding over the data |
| Reinforcement learning | Q-learning | A grid world, learned values and the greedy policy |
| Optimization | SGD, Momentum, RMSProp, Adam | Four optimizers racing on contour maps |

## Stack

React 19, TypeScript and Vite. **Framer Motion** animates what moves (lines, centroids, tree nodes, the pseudocode highlight). **D3** computes scales, chart paths, the tree layout (`d3-hierarchy`) and the loss contours (`d3-contour`). **KaTeX** renders the math. **Vitest** tests the models.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173/ai-visualizer/
npm test           # model tests
npm run build      # type check and production build into dist/
```

Every push to `main` deploys to GitHub Pages (`.github/workflows/deploy.yml`). The first time, set **Settings → Pages → Source** to **GitHub Actions**.

## How the code is organized

```
src/
├── core/                  types, math, random numbers, colors, datasets
├── engine/                animation engine (useTrainer), editing context, routing
├── components/
│   ├── plot/              main plot: canvas background + SVG marks and drag handles
│   ├── charts/            LineChart, TreeDiagram, NetworkDiagram, LossLandscape
│   └── ui/                controls, pseudocode, live equation, KaTeX
├── algorithms/
│   ├── books.ts           the references shared by the theory sections
│   ├── perceptron/
│   │   ├── model.ts       the algorithm, as a generator
│   │   ├── model.test.ts  tests of its mathematical properties
│   │   ├── theory.ts      textbook-style explanation and reading list
│   │   └── index.tsx      what to draw, parameters, formula, text, pseudocode
│   ├── mlp/               + network.ts (from scratch) and optimizers.ts (SGD, Momentum, Adam)
│   ├── shared/tree.ts     Gini, best split, prediction (used by tree and forest)
│   └── ...
└── pages/                 Home and AlgorithmPage (one page for every algorithm)
```

### The core idea: algorithms as generators

Each algorithm is a `function*`. Each `yield` is one animation step and names the pseudocode line to highlight:

```ts
export function* train(data, P, s) {
  for (let epoch = 0; epoch < P.epochs; epoch++) {
    for (const i of shuffle(order)) {
      const f = score(s, p.x, p.y);
      if (y * f <= 0) {
        s.w[0] += P.lr * y * p.x;
        s.w[1] += P.lr * y * p.y;
        s.b += P.lr * y;
        yield { line: 6, message: 'Wrong side: w moves toward this point.' };
      } else {
        yield { line: 4, message: 'Correct: nothing changes.' };
      }
    }
  }
}
```

The code reads like textbook pseudocode, and the engine (`src/engine/useTrainer.ts`) only calls `.next()` at the chosen pace. The tests run the same generators without any UI.

More in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Studying with it

1. [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md): a suggested order, what to look for, and questions.
2. [docs/NEW_ALGORITHM.md](docs/NEW_ALGORITHM.md): how to add an algorithm of your own.
3. Every algorithm page ends with exercises.

## License

MIT. Built by Tassiane Anzolin.
