"""Train a compiled re-uploader.

The QNode is Catalyst @qjit on lightning.qubit. Batch + Adam stay in JAX.

Why the split: Catalyst 0.15 dies in LLVM bufferization if you take
qml.grad through a qml.for_loop over a batch. Single-sample
qml.grad(circuit) works. jax.grad / jax.vmap on the *already compiled*
QNode also work and match. So we compile the quantum kernel and let
JAX handle the classical outer product.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from eigenflow.backends import maybe_qjit
from eigenflow.reuploading.circuit import fidelity, init_params, make_circuit
from eigenflow.reuploading.data import accuracy


def _compile(n_layers: int, layer: str):
    return maybe_qjit(make_circuit(n_layers, layer), qjit=True)


def _loss_fn(circuit, params, X, y):
    z = jax.vmap(lambda x: circuit(params, x))(X)
    return jnp.mean((1.0 - fidelity(z, y)) ** 2)


def _predict(circuit, params, X):
    z = jax.vmap(lambda x: circuit(params, x))(X)
    return (z < 0.0).astype(jnp.int32)


def train(
    X: jnp.ndarray,
    y: jnp.ndarray,
    *,
    n_layers: int = 4,
    layer: str = "compressed",
    steps: int = 80,
    lr: float = 0.2,
    seed: int = 0,
    log_every: int = 10,
    X_val: jnp.ndarray | None = None,
    y_val: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, list[dict]]:
    """Fit a re-uploader. Returns (params, history). First step compiles (~seconds)."""
    circuit = _compile(n_layers, layer)
    params = init_params(n_layers, layer, seed)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(_loss_fn, argnums=1)

    history: list[dict] = []
    for t in range(steps):
        loss, grads = value_and_grad(circuit, params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if t % log_every == 0 or t == steps - 1:
            row = {"step": t, "loss": float(loss)}
            if X_val is not None and y_val is not None:
                val_loss = float(_loss_fn(circuit, params, X_val, y_val))
                val_acc = float(accuracy(y_val, _predict(circuit, params, X_val)))
                row["val_loss"] = val_loss
                row["val_acc"] = val_acc
            history.append(row)
            extra = ""
            if "val_acc" in row:
                extra = f"  val_loss {row['val_loss']:.3f}  val_acc {row['val_acc']:.3f}"
            print(f"step {t:3d}  loss {row['loss']:.3f}{extra}")

    return params, history
