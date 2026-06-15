"""micrograd — um motor de autograd escalar, do zero.

Reimplementação de estudo do micrograd do Karpathy (trilha "Zero to Hero"),
com comentários em português. A ideia central, que está por trás de PyTorch e
de qualquer LLM: cada número guarda como foi calculado (o grafo), e a regra da
cadeia propaga gradientes de trás para frente (backpropagation).

Um `Value` é um escalar com:
- data: o valor numérico
- grad: d(saída)/d(este) acumulado no backward
- _backward: como empurrar o gradiente para os pais
- _prev: os Values que o geraram
"""

from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---------------------------------------------------------- operações
    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Value":
        assert isinstance(power, (int, float)), "só potência por constante"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward() -> None:
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t ** 2) * out.grad   # d/dx tanh = 1 - tanh^2
        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------- backprop
    def backward(self) -> None:
        """Ordena o grafo topologicamente e propaga gradientes de trás p/ frente."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------- açúcar sintático
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
