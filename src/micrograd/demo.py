"""Treina uma MLP minúscula no problema XOR usando só o autograd do zero.

XOR é o "olá mundo" das redes neurais: não é linearmente separável, então um
perceptron simples falha e uma MLP com uma camada escondida resolve. Mostra o
loop completo: forward -> loss -> backward -> passo de gradiente.

    python src/micrograd/demo.py
"""

from __future__ import annotations

try:
    from .engine import Value
    from .nn import MLP
except ImportError:
    from engine import Value
    from nn import MLP

# XOR: 4 exemplos, rótulos +1 / -1
DATA = [([0.0, 0.0], -1.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], -1.0)]


def loss_fn(model: MLP) -> Value:
    """Erro quadrático médio sobre os 4 exemplos."""
    total = Value(0.0)
    for x, y in DATA:
        pred = model([Value(x[0]), Value(x[1])])
        total = total + (pred - y) ** 2
    return total * (1.0 / len(DATA))


def main() -> None:
    model = MLP(2, [4, 4, 1], seed=1)  # 2 -> 4 -> 4 -> 1
    print(f"Parâmetros treináveis: {len(model.parameters())}")
    lr = 0.1
    for step in range(200):
        loss = loss_fn(model)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        if step % 40 == 0 or step == 199:
            print(f"  passo {step:3d} | loss {loss.data:.4f}")

    print("\nPredições finais (alvo XOR):")
    for x, y in DATA:
        pred = model([Value(x[0]), Value(x[1])]).data
        ok = "✓" if (pred > 0) == (y > 0) else "✗"
        print(f"  {x} -> {pred:+.3f} (alvo {y:+.0f}) {ok}")


if __name__ == "__main__":
    main()
