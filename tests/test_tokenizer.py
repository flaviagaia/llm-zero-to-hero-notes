"""Testes do BPE: roundtrip perfeito, compressão e tamanho de vocabulário."""

import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC / "tokenizer"))

from bpe import BPETokenizer  # noqa: E402

TEXTO = ("a tokenização define o custo do modelo de linguagem. " * 30)


def test_roundtrip_identidade():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=300)
    for s in ["a tokenização", "modelo de linguagem", "café ☕ açúcar — R$ 5,00", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_vocab_cresce_conforme_pedido():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=300)
    assert len(tok.merges) == 300 - 256
    assert len(tok.vocab) == 300


def test_bpe_comprime():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=320)
    # texto visto no treino deve comprimir bem (> 1 byte por token)
    assert tok.compression_ratio(TEXTO) > 1.5


def test_merge_reduz_comprimento():
    tok = BPETokenizer()
    tok.train(TEXTO, vocab_size=300)
    s = "modelo de linguagem"
    assert len(tok.encode(s)) < len(s.encode("utf-8"))


def test_bytes_garantem_qualquer_string():
    """Mesmo SEM treino, BPE em bytes faz roundtrip de qualquer string."""
    tok = BPETokenizer()  # vocab base (256 bytes), nenhuma fusão
    for s in ["emoji 😀🎉", "日本語", "naïve café", "\n\t mistura"]:
        assert tok.decode(tok.encode(s)) == s
