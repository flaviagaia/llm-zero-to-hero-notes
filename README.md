# LLM do Zero — Notas e Experimentos (trilha Zero to Hero)

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · stdlib pura (sem PyTorch, sem numpy) · 100% offline · MIT License

---

## 🇧🇷 Português

### O que é isto

Notas de estudo em português da trilha "construa um LLM do zero", inspirada no
curso **Neural Networks: Zero to Hero** do Andrej Karpathy, acompanhadas dos
**experimentos que rodei em cada etapa** — todos do zero, em Python puro, testados
e executáveis em segundos.

A trilha é cumulativa: gradientes → arquitetura → tokenização → pipeline completo →
alinhamento. Cada peça depende da anterior, exatamente como o curso ensina.

### As notas (`notas/`)

1. [Software 2.0 a 3.0](notas/01-software-2-3.md) — por que pesos são "código"
2. [micrograd](notas/02-micrograd.md) — backpropagation do zero **(experimento)**
3. [nanoGPT](notas/03-nanogpt.md) — self-attention sem mistério
4. [Tokenizers e minbpe](notas/04-tokenizers.md) — custos e bugs **(experimento)**
5. [nanochat](notas/05-nanochat.md) — pré-treino → SFT → alinhamento
6. [De State of GPT a DPO](notas/06-rlhf-dpo.md) — como um modelo cru vira assistente

### Os experimentos (`src/`), com resultados reais

**micrograd — autograd escalar do zero.** Reimplementei o motor de
backpropagation (`Value`) e uma MLP por cima (Neuron, Layer, MLP). Treinada no
XOR, a loss cai de **1.68 para ~0.00** em 200 passos e classifica os 4 exemplos
corretamente. Os testes validam os gradientes contra **diferenças finitas**.

```
python src/micrograd/demo.py
```

**BPE — tokenizer do zero (estilo minbpe).** Treino, encode e decode em bytes.
Num texto em português atinge **~2x de compressão** (bytes/token) com **roundtrip
perfeito**, inclusive em `café ☕ com açúcar — R$ 5,00`. Funciona em bytes, então
nunca quebra com emoji, acento ou outros idiomas.

```
python src/tokenizer/demo.py
```

### Execução

```
pip install -r requirements.txt   # só pytest
pytest tests/ -v                  # 9 testes (autograd + tokenizer)
```

### Estrutura

```
notas/                  # as 6 notas da trilha, em português
src/
├── micrograd/          # engine.py (autograd), nn.py (MLP), demo.py
└── tokenizer/          # bpe.py (BPE do zero), demo.py
tests/                  # test_micrograd.py, test_tokenizer.py
```

### Honestidade sobre escopo

São implementações de estudo, não de produção: micrograd é escalar (um Value por
número, sem tensores), e o BPE não tem regex de pré-tokenização nem tokens
especiais. O objetivo é a clareza da mecânica. As âncoras de produção são os
repositórios originais do Karpathy (micrograd, nanoGPT, minbpe, nanochat).

---

## 🇺🇸 English

### What this is

Portuguese study notes for the "build an LLM from scratch" trail, inspired by
Andrej Karpathy's **Neural Networks: Zero to Hero**, together with the
**experiments I ran at each step** — all from scratch, in pure Python, tested and
runnable in seconds. The trail is cumulative: gradients → architecture →
tokenization → full pipeline → alignment.

### The experiments (`src/`), with real results

- **micrograd — scalar autograd from scratch.** A reimplemented backprop engine
  (`Value`) plus an MLP. Trained on XOR, loss drops from **1.68 to ~0.00** in 200
  steps and classifies all 4 examples correctly. Tests validate gradients against
  **finite differences**.
- **BPE — tokenizer from scratch (minbpe-style).** Byte-level train/encode/decode.
  ~**2x compression** on Portuguese text with **perfect roundtrip**, including
  emoji and accents, because it works on bytes and never breaks.

### Running

```
pip install -r requirements.txt   # just pytest
pytest tests/ -v                  # 9 tests (autograd + tokenizer)
```

### Honest scope

Study implementations, not production: micrograd is scalar (no tensors), and the
BPE has no pre-tokenization regex or special tokens. The goal is clarity of
mechanics. Production anchors are Karpathy's original repos (micrograd, nanoGPT,
minbpe, nanochat).

---

Part of my LinkedIn series on building LLMs from scratch → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
