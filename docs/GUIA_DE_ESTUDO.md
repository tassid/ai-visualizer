# Guia de estudo

Uma ordem sugerida para estudar os algoritmos usando o visualizador e o código. Para cada um: o que observar na animação, onde está a ideia principal no código e perguntas para responder.

## Parte 1: aprender ajustando uma reta

### 1. Regressão linear
Arquivo: `src/algorithms/linear-regression/model.ts`

Observe a paisagem de perda com grau 1. O ponto amarelo desce o vale seguindo o gradiente. Esse é o mesmo mecanismo de quase tudo em deep learning.

1. Aumente a taxa para 1,2. O que acontece com o caminho?
2. Por que as curvas de nível são elipses e não círculos? (Dica: escala de x vs. escala de 1.)
3. Leia `normalEquation()` e compare com a solução por gradiente.

### 2. Perceptron
Arquivo: `src/algorithms/perceptron/model.ts`

O mais antigo de todos. Só aprende quando erra.

1. No XOR, por que os erros nunca chegam a zero?
2. A seta (vetor w) é sempre perpendicular à reta. Por quê?

### 3. Regressão logística
Arquivo: `src/algorithms/logistic-regression/model.ts`

O perceptron com uma saída suave. Compare a regra de atualização com a da regressão linear: é idêntica, trocando a previsão pela sigmoide.

1. Em dados separáveis, os pesos crescem para sempre. Por quê? O que a regularização L2 faz?

### 4. SVM linear
Arquivo: `src/algorithms/svm/model.ts`

A melhor reta não é qualquer reta que separa: é a de maior margem.

1. Apague (mentalmente) os pontos que não são vetores de suporte. A reta mudaria?

## Parte 2: além das retas

### 5. MLP
Arquivos: `src/algorithms/mlp/network.ts`, `optimizers.ts`

Leia `forward()` e `backward()` com calma: é a retropropagação inteira em cerca de 40 linhas. O teste `o gradiente da retropropagação bate com a diferença finita` é a forma clássica de provar que ela está certa.

1. Olhe os mapas de ativação da primeira camada: cada neurônio é uma "reta suave". As camadas seguintes combinam essas retas.
2. Quantos neurônios no mínimo resolvem os círculos?
3. Compare SGD, Momentum e Adam na espiral.

### 6. Naive Bayes
Arquivo: `src/algorithms/naive-bayes/model.ts`

Uma abordagem completamente diferente: modelar como cada classe gera os dados (modelo generativo) em vez de modelar a fronteira (modelo discriminativo).

1. Por que a fronteira é curva, mesmo sendo um modelo tão simples?

## Parte 3: sem pesos, sem gradiente

### 7. KNN
Arquivo: `src/algorithms/knn/model.ts`

1. Varie k e anote a acurácia leave-one-out. Desenhe a curva de viés e variância.

### 8. Árvore de decisão e 9. Floresta aleatória
Arquivos: `src/algorithms/shared/tree.ts`, `decision-tree/`, `random-forest/`

Leia `bestSplit()`: o truque da contagem acumulada transforma O(n²) em O(n log n).

1. Uma árvore profunda chega a 100% no treino. Isso é bom?
2. Por que a média de árvores ruins e diferentes é melhor que uma árvore boa?

## Parte 4: sem rótulos

### 10. K-Means, 11. DBSCAN e 12. PCA

1. Ache um conjunto em que o K-Means falha e o DBSCAN acerta, e explique por quê.
2. No PCA, por que basta girar de 0° a 180°?
3. Mostre que a variância projetada é uᵀΣu.

## Projetos maiores

1. Implemente um algoritmo novo seguindo [NOVO_ALGORITMO.md](NOVO_ALGORITMO.md). Sugestões: gradient boosting, SVM com kernel RBF, mistura de gaussianas (EM), agrupamento hierárquico, t-SNE.
2. Adicione um conjunto de dados com 3 classes e adapte a MLP para softmax.
3. Adicione divisão treino/teste e mostre a acurácia de teste ao lado da de treino.
4. Adicione um modo "comparar" que roda dois algoritmos lado a lado nos mesmos dados.
