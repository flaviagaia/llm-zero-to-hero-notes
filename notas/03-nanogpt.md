# 3. nanoGPT

nanoGPT é o GPT reduzido ao essencial, treinável e legível. O valor pedagógico
está em ver que um Transformer decoder é uma pilha de blocos simples repetidos:
atenção com múltiplas cabeças, seguida de uma MLP, com conexões residuais e
normalização em cada etapa.

A peça que de fato carrega o modelo é a self-attention. Cada token gera três
vetores (query, key, value). O token "pergunta" com sua query, compara com as
keys de todos os tokens anteriores, e usa esses pesos para fazer uma média
ponderada dos values. A máscara causal garante que um token só olhe para o
passado, nunca para o futuro, que é o que torna o modelo um previsor do próximo
token. As conexões residuais dão um caminho limpo para o gradiente fluir em
redes profundas, e a normalização mantém as ativações em escala estável.

A continuidade com a etapa anterior é direta: a atenção é só mais um grafo de
multiplicações e somas, o mesmo tipo de operação que o micrograd já sabia
diferenciar. nanoGPT troca o autograd manual pelo PyTorch e adiciona paralelismo,
mas a ideia de fundo não muda.
