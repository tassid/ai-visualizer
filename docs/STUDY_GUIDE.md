# Study guide

A suggested order for studying the algorithms with the visualizer and the code. For each one: what to watch, where the key idea lives in the code, and questions to answer.

## Part 1: learning by fitting a line

### 1. Linear regression
File: `src/algorithms/linear-regression/model.ts`

The red squares are the squared errors; the loss is their average area. Fit the line by hand with the two handles first, then press play and watch gradient descent walk down the loss landscape. That same mechanism drives almost all of deep learning.

1. Set the learning rate to 1.2. What happens to the path?
2. Why are the contours ellipses rather than circles? (Hint: compare the spread of x with the constant 1 that multiplies b.)
3. Read `normalEquation()` and compare it with the gradient answer.

### 2. Perceptron
File: `src/algorithms/perceptron/model.ts`

The oldest of them all. It only learns when it is wrong.

1. On XOR, why do the mistakes never reach zero?
2. The arrow (the vector w) is always perpendicular to the line. Why?

### 3. Logistic regression
File: `src/algorithms/logistic-regression/model.ts`

A perceptron with a soft output. Compare its update rule with linear regression's: it is identical, with the prediction passed through a sigmoid.

1. On separable data, the arrow keeps growing. Why? What does the L2 penalty do?

### 4. Linear SVM
File: `src/algorithms/svm/model.ts`

The best line is not any line that separates the classes: it is the one with the widest margin.

1. Drag a point that is not a support vector. Does the line move?

## Part 2: beyond lines

### 5. Multilayer perceptron
Files: `src/algorithms/mlp/network.ts`, `optimizers.ts`

Read `forward()` and `backward()` slowly: that is all of backpropagation in about 40 lines. The test "the backpropagation gradient matches the finite difference" is the classic proof that it is right.

1. Look at the first layer's activation maps: each neuron is a soft line. The next layers combine those lines.
2. Hover the plane and follow the signal through the network. Which neurons light up for the inner circle?
3. Compare SGD, Momentum and Adam on the spiral.

### 6. Naive Bayes
File: `src/algorithms/naive-bayes/model.ts`

A different approach altogether: model how each class generates data (a generative model) instead of modeling the boundary (a discriminative one).

1. Why is the boundary curved, for such a simple model?

## Part 3: no weights, no gradients

### 7. k-nearest neighbors
File: `src/algorithms/knn/model.ts`

1. Move k and note the leave-one-out accuracy. Sketch the bias and variance trade-off.

### 8. Decision tree and 9. Random forest
Files: `src/algorithms/shared/tree.ts`, `decision-tree/`, `random-forest/`

Read `bestSplit()`: the running count turns an O(n²) search into O(n log n).

1. A deep tree reaches 100% on the training data. Is that good?
2. Why is the average of many different, worse trees better than one good tree?

## Part 4: no labels

### 10. k-means, 11. DBSCAN and 12. PCA

1. Find a dataset where k-means fails and DBSCAN succeeds, and explain why.
2. In PCA, why is turning from 0° to 180° enough?
3. Show that the projected variance is uᵀΣu.

## Bigger projects

1. Add an algorithm following [NEW_ALGORITHM.md](NEW_ALGORITHM.md). Ideas: gradient boosting, an RBF kernel SVM, a Gaussian mixture fitted by EM, hierarchical clustering, t-SNE.
2. Add a three-class dataset and give the network a softmax output.
3. Split the data into training and test sets and show both accuracies.
4. Add a compare mode that runs two algorithms side by side on the same data.
