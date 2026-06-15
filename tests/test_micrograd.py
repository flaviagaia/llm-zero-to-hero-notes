"""Testes do micrograd: os gradientes do autograd batem com diferenças finitas."""

import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC / "micrograd"))

from engine import Value  # noqa: E402
from nn import MLP  # noqa: E402


def _numerical_grad(f, x: float, h: float = 1e-6) -> float:
    """Derivada por diferença central — a verdade de referência."""
    return (f(x + h) - f(x - h)) / (2 * h)


def test_grad_de_expressao_composta():
    """d/dx de uma expressão não trivial bate com a diferença finita."""
    def expr(xv: float) -> float:
        # f(x) = (x*2 + 1) * tanh(x) ; calculado só com floats p/ a referência
        import math
        return (xv * 2 + 1) * math.tanh(xv)

    x = Value(0.7)
    y = (x * 2 + 1) * x.tanh()
    y.backward()
    assert abs(x.grad - _numerical_grad(expr, 0.7)) < 1e-4


def test_relu_e_potencia():
    x = Value(-3.0)
    y = (x * 2).relu() + x ** 2     # ReLU zera o ramo linear; x^2 contribui
    y.backward()
    # d/dx [relu(2x) + x^2] em x=-3: relu inativo (0) + 2x = -6
    assert abs(x.grad - (-6.0)) < 1e-9


def test_gradiente_acumula_em_no_reutilizado():
    """Se um Value é usado duas vezes, os gradientes somam (regra da cadeia)."""
    x = Value(3.0)
    y = x + x          # = 2x, dy/dx = 2
    y.backward()
    assert x.grad == 2.0


def test_mlp_treina_xor():
    """A MLP reduz a loss do XOR e classifica os 4 exemplos corretamente."""
    data = [([0.0, 0.0], -1.0), ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0), ([1.0, 1.0], -1.0)]
    model = MLP(2, [4, 4, 1], seed=1)

    def loss_fn():
        total = Value(0.0)
        for xx, yy in data:
            pred = model([Value(xx[0]), Value(xx[1])])
            total = total + (pred - yy) ** 2
        return total * (1.0 / len(data))

    loss0 = loss_fn().data
    for _ in range(200):
        loss = loss_fn()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad

    assert loss.data < 0.01 * loss0   # caiu pelo menos 100x
    for xx, yy in data:
        pred = model([Value(xx[0]), Value(xx[1])]).data
        assert (pred > 0) == (yy > 0)
