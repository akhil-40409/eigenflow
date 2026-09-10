"""Train QuIRK with MSE + Adam."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax

from eigenflow.quirk.model import QuIRK


def train(
    model: QuIRK,
    X: jnp.ndarray,
    y: jnp.ndarray,
    *,
    steps: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
    log_every: int = 20,
    X_val: jnp.ndarray | None = None,
    y_val: jnp.ndarray | None = None,
) -> tuple[dict, list[dict]]:
    key = jax.random.PRNGKey(seed)
    params = model.init(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, xx, yy):
        pred = model.apply(p, xx)
        return jnp.mean((pred - yy) ** 2)

    value_and_grad = jax.value_and_grad(loss_fn)

    history: list[dict] = []
    for t in range(steps):
        loss, grads = value_and_grad(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if t % log_every == 0 or t == steps - 1:
            row = {"step": t, "loss": float(loss)}
            if X_val is not None and y_val is not None:
                row["val_loss"] = float(loss_fn(params, X_val, y_val))
            history.append(row)
            extra = f"  val {row['val_loss']:.4f}" if "val_loss" in row else ""
            print(f"step {t:3d}  mse {row['loss']:.4f}{extra}")

    return params, history


def fit_dataset(
    task_id: str,
    *,
    hidden: Sequence[int] = (4,),
    n_reps: int = 2,
    n_samples: int = 200,
    steps: int = 80,
    seed: int = 0,
    qjit: bool = True,
) -> tuple[QuIRK, dict, list[dict]]:
    from eigenflow.datasets import make_dataset

    key = jax.random.PRNGKey(seed)
    data = make_dataset(task_id, key, n_samples=n_samples)
    sizes = [data["n_features"], *hidden, 1]
    model = QuIRK(sizes, n_reps=n_reps, qjit=qjit)
    params, hist = train(
        model,
        data["x_train"],
        data["y_train"],
        steps=steps,
        seed=seed + 1,
        X_val=data["x_val"],
        y_val=data["y_val"],
    )
    return model, params, hist
