"""Train the single-qubit re-uploader on the circle task.

    python examples/train_circle.py
    python examples/train_circle.py --layers 6 --steps 120 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp

from eigenflow.reuploading import accuracy, circle, make_circuit, predict, train


def main() -> None:
    p = argparse.ArgumentParser(description="Data-reuploading circle classifier")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--layer", choices=("compressed", "split"), default="compressed")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    X_tr, y_tr = circle(args.train, seed=args.seed)
    X_te, y_te = circle(args.test, seed=args.seed + 1)

    print(
        f"re-uploading  layers={args.layers}  kind={args.layer}  "
        f"device=lightning.qubit  qjit=on"
    )
    params, _ = train(
        X_tr,
        y_tr,
        n_layers=args.layers,
        layer=args.layer,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        X_val=X_te,
        y_val=y_te,
    )

    circuit = make_circuit(args.layers, args.layer)
    yhat = predict(circuit, params, X_te)
    acc = float(accuracy(y_te, yhat))
    print(f"test accuracy  {acc:.3f}")

    if args.plot:
        _plot(X_te, y_te, yhat)


def _plot(X: jnp.ndarray, y: jnp.ndarray, yhat: jnp.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install 'eigenflow[plot]'", file=sys.stderr)
        return

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, labels, title in (
        (axes[0], y, "true"),
        (axes[1], yhat, "predicted"),
    ):
        inside = labels == 1
        ax.scatter(X[~inside, 0], X[~inside, 1], c="C3", s=12, label="0")
        ax.scatter(X[inside, 0], X[inside, 1], c="C0", s=12, label="1")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    path = out / "circle.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
