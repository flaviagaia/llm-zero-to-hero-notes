"""Treina um BPE pequeno num texto em português e mostra compressão + roundtrip.

    python src/tokenizer/demo.py
"""

from __future__ import annotations

try:
    from .bpe import BPETokenizer
except ImportError:
    from bpe import BPETokenizer

TEXTO = (
    "a tokenização é a etapa mais subestimada de um modelo de linguagem. "
    "a tokenização define o custo, porque o preço é por token, e o token nasce aqui. "
    "byte-pair encoding começa nos bytes e funde os pares mais frequentes. "
    "a sequência de tokens fica mais curta a cada fusão aprendida. "
) * 8


def main() -> None:
    tok = BPETokenizer()
    print("Treinando BPE (vocab_size=320) num texto em português...\n")
    tok.train(TEXTO, vocab_size=320, verbose=True)

    amostra = "a tokenização define o custo do modelo de linguagem."
    ids = tok.encode(amostra)
    volta = tok.decode(ids)

    print("\nAmostra:", amostra)
    print(f"Bytes UTF-8 : {len(amostra.encode('utf-8'))}")
    print(f"Tokens BPE  : {len(ids)}")
    print(f"Compressão  : {tok.compression_ratio(amostra):.2f}x bytes/token")
    print(f"Roundtrip OK: {volta == amostra}")

    # mesmo emoji e acento sobrevivem (BPE em bytes nunca quebra)
    dificil = "café ☕ com açúcar — R$ 5,00"
    print(f"\nString difícil: {dificil!r}")
    print(f"Roundtrip OK  : {tok.decode(tok.encode(dificil)) == dificil}")


if __name__ == "__main__":
    main()
