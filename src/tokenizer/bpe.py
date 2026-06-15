"""BPE (Byte-Pair Encoding) do zero — estudo do minbpe do Karpathy.

Tokenização é onde nascem custos e bugs de LLM, e quase ninguém olha. BPE é o
algoritmo por trás de GPT-2/3/4: começa nos 256 bytes e vai fundindo o par de
tokens mais frequente, repetidamente, até o tamanho de vocabulário desejado.

Trabalhar em BYTES (não em caracteres) garante que QUALQUER string — emoji,
acento, idioma — sempre codifica e decodifica sem erro. É a base do "nunca
quebra" dos tokenizers modernos.
"""

from __future__ import annotations


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Conta a frequência de cada par adjacente."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Substitui todas as ocorrências de `pair` pelo token `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}      # par -> novo id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Aprende as fusões a partir do texto, até atingir vocab_size."""
        assert vocab_size >= 256, "vocab_size precisa ser >= 256 (os bytes base)"
        n_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for k in range(n_merges):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)   # par mais frequente
            new_id = 256 + k
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"  merge {k+1}: {pair} -> {new_id} ({self.vocab[new_id]!r})")

    def encode(self, text: str) -> list[int]:
        """Texto -> lista de ids, aplicando as fusões na ordem aprendida."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # funde o par cuja fusão foi aprendida MAIS CEDO (menor id)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Lista de ids -> texto. Bytes garantem roundtrip perfeito."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        """Bytes originais / tokens — quão mais curta a sequência ficou."""
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(self.encode(text))
        return n_bytes / max(n_tokens, 1)
