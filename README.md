# AI Visualizer

Visualizador interativo de algoritmos de aprendizado de máquina, com animações passo a passo. Inspirado no [Sort Visualizer](https://github.com/Myphz/sortvisualizer), mas para IA: perceptron, MLP, SVM, KNN, árvores, K-Means, DBSCAN, PCA e mais.

**Demo:** https://tassid.github.io/ai-visualizer/

Cada algoritmo foi escrito do zero em TypeScript, sem bibliotecas de ML, e pensado para ser lido. O objetivo é estudar: você vê o modelo aprendendo, a linha do pseudocódigo que está rodando e o código real que faz aquilo.

## Algoritmos

| Grupo | Algoritmo | O que a animação mostra |
| --- | --- | --- |
| Redes neurais | Perceptron | A reta corrigida a cada erro, com o vetor de pesos |
| Redes neurais | MLP (backpropagation) | Fronteira de decisão e mapa de ativação de cada neurônio |
| Modelos lineares | Regressão linear | Resíduos, ajuste polinomial e paisagem de perda com curvas de nível |
| Modelos lineares | Regressão logística | Probabilidade, fronteira e linhas de p = 0,25 e 0,75 |
| Modelos lineares | SVM linear | Margem, vetores de suporte e perda hinge |
| Probabilísticos | Naive Bayes gaussiano | Elipses de cada classe estimadas ponto a ponto (Welford) |
| Instâncias e árvores | KNN | Varredura do plano com vizinhos e raio |
| Instâncias e árvores | Árvore de decisão | Cortes por Gini nível a nível e diagrama da árvore |
| Instâncias e árvores | Floresta aleatória | Bootstrap, média das árvores e erro out of bag |
| Não supervisionado | K-Means | Atribuição, centroides deslizando e regiões de Voronoi |
| Não supervisionado | DBSCAN | Núcleos, borda, ruído e a fila de expansão |
| Não supervisionado | PCA | Reta girando, projeções e variância por ângulo |

## Stack

React 19 + TypeScript + Vite, **Framer Motion** para as animações (retas, centroides, nós da árvore, destaque do pseudocódigo, transições de página), **D3** para escalas, gráficos, layout de árvore (`d3-hierarchy`) e curvas de nível (`d3-contour`), e **Vitest** para os testes dos modelos.

## Como rodar

```bash
npm install
npm run dev        # http://localhost:5173/ai-visualizer/
npm test           # testes dos algoritmos
npm run build      # verificação de tipos + build de produção em dist/
```

O deploy no GitHub Pages é automático a cada push na `main` (`.github/workflows/deploy.yml`). Na primeira vez, ative em **Settings → Pages → Source: GitHub Actions**.

## Como o código está organizado

```
src/
├── core/                  tipos, matemática, números aleatórios, cores e conjuntos de dados
├── engine/                motor de animação (useTrainer) e rotas
├── components/
│   ├── plot/              gráfico principal: canvas (fundo) + SVG (Framer Motion)
│   ├── charts/            LineChart, TreeDiagram, NetworkDiagram, LossLandscape (D3)
│   └── ui/                controles, parâmetros, métricas, pseudocódigo
├── algorithms/
│   ├── perceptron/
│   │   ├── model.ts       o algoritmo em si, como generator
│   │   ├── model.test.ts  testes que provam propriedades do algoritmo
│   │   ├── Visual.tsx     o que desenhar
│   │   └── index.ts       metadados: parâmetros, métricas, textos, pseudocódigo
│   ├── mlp/               + network.ts (rede do zero) e optimizers.ts (SGD, Momentum, Adam)
│   ├── shared/tree.ts     Gini, melhor corte, predição (usado por árvore e floresta)
│   └── ...
└── pages/                 Home e AlgorithmPage (genérica para todos os algoritmos)
```

### A ideia central: algoritmos como generators

Cada algoritmo é uma `function*`. Cada `yield` é um passo da animação e informa qual linha do pseudocódigo destacar:

```ts
export function* train(data, P, s) {
  for (let epoch = 0; epoch < P.epochs; epoch++) {
    for (const i of shuffle(order)) {
      const f = score(s, p.x, p.y);
      if (y * f <= 0) {
        s.w[0] += P.lr * y * p.x;
        s.w[1] += P.lr * y * p.y;
        s.b += P.lr * y;
        yield { line: 6, message: 'Errou: w e b se movem.' };
      } else {
        yield { line: 4, message: 'Acertou: nada muda.' };
      }
    }
  }
}
```

O código fica parecido com o pseudocódigo de um livro, e o motor (`src/engine/useTrainer.ts`) só chama `.next()` no ritmo escolhido. Os mesmos generators rodam nos testes, sem interface.

Mais detalhes em [docs/ARQUITETURA.md](docs/ARQUITETURA.md).

## Para estudar

1. [docs/GUIA_DE_ESTUDO.md](docs/GUIA_DE_ESTUDO.md): ordem sugerida, o que observar em cada algoritmo e exercícios.
2. [docs/NOVO_ALGORITMO.md](docs/NOVO_ALGORITMO.md): passo a passo para implementar um algoritmo novo (bom exercício).
3. Cada página do site tem uma seção "Para estudar" com desafios específicos.

## Licença

MIT. Feito por Tassiane Anzolin.
