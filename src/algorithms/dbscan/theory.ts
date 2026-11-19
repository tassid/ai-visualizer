import { paper } from '../books';

export const theory = {
  theory: [
    'DBSCAN defines clusters as dense regions separated by sparse ones. With the $\\varepsilon$-neighborhood $N_\\varepsilon(\\mathbf{p}) = \\{\\mathbf{q} : \\lVert\\mathbf{p}-\\mathbf{q}\\rVert \\le \\varepsilon\\}$, a point is a core point when $|N_\\varepsilon(\\mathbf{p})| \\ge \\text{minPts}$. A point $\\mathbf{q}$ is directly density-reachable from a core point $\\mathbf{p}$ if it lies in $N_\\varepsilon(\\mathbf{p})$, and density-reachable if a chain of such steps leads from $\\mathbf{p}$ to $\\mathbf{q}$.',
    'Reachability is not symmetric, because a border point can be reached but cannot reach anything. Two points are density-connected if both are density-reachable from some common core point, and that relation is symmetric. A cluster is a maximal set of density-connected points. Every point that belongs to no cluster is noise.',
    'The algorithm is a graph search. Build, implicitly, a graph whose vertices are points and whose edges join each core point to its $\\varepsilon$-neighbors; the clusters are the connected components reachable from core points, found by breadth-first search from any unvisited core point. Core points always end up in the same cluster regardless of visiting order. A border point within reach of two clusters joins whichever reaches it first, which is the only order-dependent part.',
    'Each neighborhood query costs $O(n)$ by brute force, giving $O(n^2)$ overall. With a spatial index such as a k-d tree or a grid of cells of side $\\varepsilon$, each query becomes much cheaper in low dimensions. Choosing the parameters is the hard part. A rule of thumb from the original paper is to plot each point\'s distance to its $k$-th nearest neighbor, sorted, with $k = \\text{minPts} - 1$, and set $\\varepsilon$ at the knee of that curve. Because a single $\\varepsilon$ is used everywhere, DBSCAN struggles with clusters of very different densities, the motivation for OPTICS and HDBSCAN.',
  ],
  references: [
    paper('Ester, Kriegel, Sander and Xu, A density-based algorithm for discovering clusters in large spatial databases with noise, KDD (1996)'),
    paper('Schubert, Sander, Ester, Kriegel and Xu, DBSCAN revisited, revisited: why and how you should (still) use DBSCAN, ACM Transactions on Database Systems 42 (2017)'),
    paper('Campello, Moulavi and Sander, Density-based clustering based on hierarchical density estimates (HDBSCAN), PAKDD (2013)'),
  ],
};
