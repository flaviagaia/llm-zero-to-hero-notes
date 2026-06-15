# 5. nanochat

nanochat é o projeto que amarra a trilha: um ChatGPT em miniatura, do tokenizer
ao modelo conversacional, treinável por um custo da ordem de uma centena de
dólares. Karpathy o lançou como capstone do curso LLM101n, e é o ponto em que as
peças anteriores se encaixam num pipeline único.

O pipeline tem três fases. Primeiro, pré-treino: o modelo aprende a prever o
próximo token em um corpus gigante e absorve gramática, fatos e estilo. O
resultado é um "modelo base", que completa texto mas não sabe conversar. Segundo,
fine-tuning supervisionado (SFT): treina-se o base em exemplos de diálogo no
formato pergunta e resposta, ensinando o formato de assistente. Terceiro,
alinhamento por preferência (o tema da próxima nota), que refina o
comportamento.

O que nanochat deixa claro é que "treinar um LLM" não é um evento único, e sim
uma sequência de estágios com objetivos diferentes. Cada estágio reaproveita a
mesma maquinaria de gradiente e atenção, só muda o dado e a função de perda.
