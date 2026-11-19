# Architecture

How the pieces fit, from pressing Play to a pixel on screen.

## 1. An algorithm's contract

Everything revolves around `AlgorithmDef` (`src/core/types.ts`). An algorithm declares:

1. `create(data, params)`: returns a `Trainer`, a pair `{ state, steps }`. `state` is a mutable object with the model; `steps` is the generator that changes it.
2. `Visual`: a React component that draws the state.
3. `params`: a declarative list of parameters. The controls are generated from it.
4. `equation`: the model's key formula as TeX, filled with live numbers (and with the hovered point, when there is one).
5. `metrics`, `history`, `aux`: what goes in the metrics list, the side chart and the panel under the plot.
6. `docs`: description, steps, complexity, pseudocode and exercises.

`AlgorithmPage.tsx` is generic: it knows nothing about perceptrons or trees and builds the page from the definition.

## 2. Why generators

The usual alternative is a state machine (`step()` with a `switch` over the current phase). That scatters the algorithm into pieces and moves it away from the pseudocode.

With `function*`, the natural control flow (nested loops, `if`, `continue`, `return`) survives. `yield` pauses the function mid-loop and the engine decides when to resume. Compare k-means in `src/algorithms/kmeans/model.ts` with the pseudocode on its page: same shape.

The generator's return value (`return { line, message }`) becomes the final message, for example "Converged after 4 iterations".

## 3. The engine (`src/engine/useTrainer.ts`)

```
requestAnimationFrame
  └─ add elapsed time × speed (steps per second)
       └─ run floor(total) steps, at most 28 ms per frame
            └─ bump `version` → React renders again
```

The model state is mutable, so React does not see changes by itself. The `version` counter fixes that and doubles as a cache key: the decision background (`Heatmap`) only recomputes when `version` changes.

When the data or the parameters change, the trainer is rebuilt in the same render, so the picture never pairs new data with an old model. If the old model had trained, the new one **replays**: it runs as many steps as fit in each frame until done. That is what makes the boundary follow a point while you drag it.

`runToEnd()` runs a generator to completion without animation. The home page thumbnails and the tests use it.

## 4. Interaction

`src/engine/editing.ts` defines an `EditContext` that the algorithm page provides and the thumbnails do not. Inside a plot:

1. `EditLayer` (in `Plot.tsx`) is an invisible rectangle under the marks. A click on empty space adds a point, a drag moves the nearest point, a shift-click deletes it. It also reports the hovered position.
2. `Handle` (in `marks.tsx`) is a draggable dot. Its `onDrag` writes straight into the model state, then calls `touch()` to redraw. Because the generator reads the state on every step, training resumes from wherever you left the model.
3. `LinearHandles` builds the two handles of a linear model from `w` and `b`: one slides the line, the other turns it (and, for logistic regression and the SVM, sets `|w|`).

## 5. Drawing: canvas + SVG

The plot (`components/plot/Plot.tsx`) has two layers:

1. **Canvas** (`Heatmap.tsx`): the tinted background showing the model's prediction everywhere. Drawn at low resolution (for example 72 x 72) and stretched by CSS, which gives a smooth gradient through bilinear interpolation at low cost.
2. **SVG**: the number plane, points, lines, rings, arrows and handles. Elements with meaning live here so Framer Motion can tween them and the pointer can grab them.

D3 scales (`scaleLinear`) map the world `[-1, 1]` to pixels `[0, 560]`, with the y axis flipped.

## 6. The look

The style follows 3Blue1Brown and Manim:

1. Manim's palette (`core/colors.ts`): blue `#58C4DD` and red `#FC6255` for the classes, yellow `#FFFF00` for whatever the model is doing right now.
2. A NumberPlane grid: faint blue lines, white axes with ticks, drawn in once when the page opens.
3. Computer Modern type through KaTeX's fonts, for prose and math alike. Controls use the system sans.
4. Manim's smooth ease-in-out for tweens. Motion only where it explains something: a line moving after an update, a node growing out of its parent.

## 7. Where each library comes in

**Framer Motion**

1. `motion.line` with `animate={{ x1, y1, x2, y2 }}`: decision lines glide after each update.
2. `motion.g` with `animate={{ x, y }}`: k-means centroids and tree nodes.
3. `AnimatePresence`: support vectors and DBSCAN's queue fade in and out; page transitions.
4. `layoutId`: the pseudocode highlight slides between lines.
5. `pathLength`: the number plane draws itself.

**D3** (always "D3 computes, React draws")

1. `d3-scale`: every coordinate conversion.
2. `d3-shape`: lines for charts and the regression curve.
3. `d3-hierarchy`: the decision tree layout.
4. `d3-contour` and `d3-scale-chromatic`: the loss landscape.

**KaTeX**: the live equations, the formulas in the text, the complexity table.

## 8. Tests

Each `model.test.ts` checks a mathematical property, not implementation details. For example:

1. The perceptron converges on separable data and does not on XOR.
2. The backpropagation gradient matches a finite difference (gradient check).
3. k-means inertia never increases.
4. Gradient descent for linear regression reaches the normal equation solution.
5. About 1/e of the points are left out of each bootstrap sample.
6. The PCA angle sweep finds the same component as the eigenvalues.

`seed()` in `core/random.ts` makes the tests reproducible.
