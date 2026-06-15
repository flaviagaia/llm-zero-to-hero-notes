# 2. micrograd

micrograd é um motor de autograd escalar minúsculo. A lição é que toda a
"mágica" do aprendizado profundo se resume a duas coisas: um grafo de operações
e a regra da cadeia aplicada de trás para frente (backpropagation).

Cada número (`Value`) guarda como foi calculado e uma função `_backward` que sabe
empurrar o gradiente para os pais. No `backward()`, ordenamos o grafo
topologicamente e percorremos de trás para frente, acumulando gradientes. Um
detalhe que costuma confundir: quando um nó é reutilizado, os gradientes somam.
É a regra da cadeia multivariada, e é por isso que zeramos os gradientes a cada
passo de treino.

Experimento neste repo (`src/micrograd/`): reimplementei o motor e uma MLP por
cima (Neuron, Layer, MLP) e treinei no XOR, o problema clássico que um modelo
linear não resolve. A loss cai de cerca de 1.68 para perto de zero em 200 passos
e os quatro exemplos são classificados corretamente. Os testes validam os
gradientes contra diferenças finitas, então o autograd está provavelmente certo,
não só funcionando por sorte.

Rode: `python src/micrograd/demo.py` e `pytest tests/test_micrograd.py -v`.
