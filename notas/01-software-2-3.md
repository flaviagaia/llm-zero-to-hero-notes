# 1. Software 2.0 a 3.0

No ensaio "Software 2.0" (2017), Karpathy propôs uma ideia que envelheceu muito
bem: em vez de escrever regras à mão (Software 1.0), nós especificamos um
objetivo e deixamos a otimização por gradiente encontrar o programa, codificado
nos pesos de uma rede neural. O conjunto de dados vira o código-fonte; o treino
vira o compilador.

A virada recente, que ele chamou de Software 3.0, é que o prompt em linguagem
natural passa a ser uma camada de programação por cima do modelo. "Inglês é a
nova linguagem de programação." Isso reposiciona muita coisa do dia a dia: o
prompt deixa de ser um detalhe e vira artefato versionado (é a tese do post sobre
Prompt Registry), e engenharia de contexto passa a importar mais que truques de
redação.

Por que isso abre a trilha: as três camadas convivem. Embaixo, pesos aprendidos
por gradiente (Software 2.0), que vamos construir do micrograd ao nanochat. Em
cima, a camada de linguagem natural (Software 3.0). Entender a de baixo é o que
permite usar a de cima sem mágica.
