"""Uma rede neural mínima sobre o motor de autograd (engine.py).

Neuron -> Layer -> MLP, exatamente a hierarquia do micrograd. O ponto
pedagógico: uma "rede neural" é só um monte de Values somados e multiplicados,
e o treino é o mesmo backward de sempre.
"""

from __future__ import annotations

import random

try:
    from .engine import Value
except ImportError:  # rodar como script solto
    from engine import Value


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, n_in: int, nonlin: bool = True, rng: random.Random | None = None) -> None:
        rng = rng or random.Random(0)
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, rng: random.Random | None = None, **kw) -> None:
        rng = rng or random.Random(0)
        self.neurons = [Neuron(n_in, rng=rng, **kw) for _ in range(n_out)]

    def __call__(self, x: list[Value]) -> list[Value] | Value:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in: int, n_outs: list[int], seed: int = 0) -> None:
        rng = random.Random(seed)
        sizes = [n_in] + n_outs
        self.layers = [
            Layer(sizes[i], sizes[i + 1], rng=rng, nonlin=(i != len(n_outs) - 1))
            for i in range(len(n_outs))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
