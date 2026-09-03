"""Smoke example for Welvet C-ABI Python bindings."""
from __future__ import annotations

import welvet as w


def main() -> None:
    w.assert_engine_version()
    print("engine", w.engine_version())
    g = w.create_grid()
    try:
        w.place_dense(g, {"in": 8, "out": 4, "act": "relu", "dtype": w.DType.FLOAT32})
        y = w.forward(g, [0.05 * (i + 1) for i in range(8)])
        print("forward", y)
        print("train", w.train_sgd(g, [0.05 * (i + 1) for i in range(8)], [0, 0, 0, 1], 0.05))
    finally:
        w.free_grid(g)


if __name__ == "__main__":
    main()
