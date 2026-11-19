# Arquitetura

Este documento explica como as peças se encaixam, do clique em "Iniciar" até o pixel na tela.

## 1. O contrato de um algoritmo

Tudo gira em torno da interface `AlgorithmDef` (`src/core/types.ts`). Um algoritmo declara:

1. `create(data, params)`: devolve um `Trainer`, que é um par `{ state, steps }`. `state` é um objeto mutável com o estado do modelo; `steps` é o generator que modifica esse estado.
2. `Visual`: um componente React que recebe o `state` e desenha.
3. `params`: lista declarativa de parâmetros. A interface de controles é gerada a partir dela.
4. `metrics`, `history`, `aux`: o que mostrar no painel de métricas, no gráfico lateral e no painel inferior.
5. `docs`: descrição, passos, complexidade, pseudocódigo e exercícios.

A página `AlgorithmPage.tsx` é genérica: ela não sabe nada sobre perceptrons ou árvores, apenas monta a tela a partir da definição.

## 2. Por que generators

A alternativa comum seria escrever cada algoritmo como uma máquina de estados (`step()` com um `switch` sobre a fase atual). Isso espalha o algoritmo em pedaços e o afasta do pseudocódigo.

Com `function*`, o fluxo de controle natural (laços aninhados, `if`, `continue`, `return`) é preservado. O `yield` pausa a função no meio do laço e o motor decide quando continuar. Compare o K-Means em `src/algorithms/kmeans/model.ts` com o pseudocódigo da página: a estrutura é a mesma.

O valor de retorno do generator (`return { line, message }`) vira a mensagem final, por exemplo "Convergiu após 4 iterações".

## 3. O motor (`src/engine/useTrainer.ts`)

```
requestAnimationFrame
  └─ acumula tempo × velocidade (passos por segundo)
       └─ executa floor(acumulado) passos, com limite de 28 ms por quadro
            └─ incrementa `version` → React renderiza de novo
```

Como o estado do modelo é mutável, o React não percebe as mudanças sozinho. O contador `version` resolve isso e também serve como chave de cache: o fundo de decisão (`Heatmap`) só é recalculado quando `version` muda.

Quando dados ou parâmetros mudam, o treinador é recriado na mesma renderização, para que o desenho nunca combine dados novos com um estado antigo.

`runToEnd()` executa um generator até o fim sem animação. É usado nas miniaturas da página inicial e nos testes.

## 4. Desenho: canvas + SVG

O gráfico principal (`components/plot/Plot.tsx`) tem duas camadas:

1. **Canvas** (`Heatmap.tsx`): o fundo colorido com a previsão do modelo em cada ponto do plano. Desenhado em baixa resolução (por exemplo 72 x 72) e ampliado pelo CSS, o que dá um gradiente suave por interpolação bilinear e custa pouco.
2. **SVG**: pontos, retas, anéis e setas. Elementos com significado ficam aqui porque o Framer Motion consegue animar seus atributos.

As escalas do D3 (`scaleLinear`) convertem o mundo `[-1, 1]` em pixels `[0, 560]`, com o eixo y invertido.

## 5. Onde cada biblioteca entra

**Framer Motion**

1. `motion.line` com `animate={{ x1, y1, x2, y2 }}`: retas de decisão deslizam com mola em vez de pular (perceptron, logística, SVM).
2. `motion.g` com `animate={{ x, y }}`: centroides do K-Means e nós da árvore.
3. `AnimatePresence`: vetores de suporte e itens da fila do DBSCAN aparecem e somem com animação; troca de páginas.
4. `layoutId`: o destaque do pseudocódigo desliza entre as linhas.
5. `variants` com `staggerChildren`: entrada em cascata dos cards da página inicial.

**D3** (sempre no padrão "D3 calcula, React desenha")

1. `d3-scale`: todas as conversões de coordenadas.
2. `d3-shape`: `line` e `area` nos gráficos de perda e no ajuste da regressão.
3. `d3-hierarchy`: layout do diagrama da árvore de decisão.
4. `d3-contour` + `d3-scale-chromatic`: curvas de nível da paisagem de perda.

## 6. Testes

Cada `model.test.ts` testa uma propriedade matemática do algoritmo, não detalhes de implementação. Exemplos:

1. O perceptron converge em dados separáveis e não converge no XOR.
2. O gradiente da retropropagação bate com a diferença finita (gradient check).
3. A inércia do K-Means nunca aumenta.
4. A descida de gradiente da regressão linear chega à solução da equação normal.
5. Cerca de 1/e dos pontos fica fora de cada amostra bootstrap.
6. A busca por ângulo do PCA encontra o mesmo componente que os autovalores.

`seed()` em `core/random.ts` deixa os testes reproduzíveis.
