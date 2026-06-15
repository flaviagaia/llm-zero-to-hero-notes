# 4. Tokenizers e minbpe

Tokenização é a etapa mais subestimada de um LLM, e a fonte de uma surpreendente
quantidade de bugs e custos. O modelo nunca vê texto: vê uma sequência de ids
inteiros. Como o preço de API é por token, o tokenizer define, literalmente,
quanto custa cada chamada.

O algoritmo padrão é Byte-Pair Encoding (BPE). Começa nos 256 bytes possíveis e,
repetidamente, funde o par de tokens adjacentes mais frequente em um novo token,
até atingir o tamanho de vocabulário desejado. Trabalhar em bytes, e não em
caracteres, é o detalhe que garante que qualquer string, com emoji, acento ou
qualquer idioma, sempre codifica e decodifica sem erro.

Experimento neste repo (`src/tokenizer/`): implementei BPE do zero (treino,
encode, decode). Treinado num texto em português, ele atinge cerca de 2x de
compressão (bytes por token) e faz roundtrip perfeito, inclusive em "café ☕ com
açúcar — R$ 5,00". Os testes garantem que, mesmo sem nenhuma fusão aprendida, o
roundtrip em bytes funciona para japonês, emoji e acentuação. É a base do "nunca
quebra" dos tokenizers modernos.

Rode: `python src/tokenizer/demo.py` e `pytest tests/test_tokenizer.py -v`.
